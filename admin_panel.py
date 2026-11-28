"""TestMitra AI — Admin Panel (1-Click PDF-to-Test Converter & Error Report Manager)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv, set_key
import streamlit as st

from firebase_cloudinary import (
    delete_test_from_firestore,
    fetch_app_config,
    fetch_error_reports,
    fetch_published_tests,
    publish_test_to_firestore,
    reset_firebase_app,
    resolve_error_report,
    save_app_config,
    update_test_in_firestore,
    upload_image_to_cloudinary,
)
from pdf_parser import (
    extract_answer_key_text,
    list_gemini_models,
    parse_pdf_chunk_with_gemini,
    split_pdf_into_chunks,
)

BASE_DIR = Path(__file__).resolve().parent
ENV_PATH = BASE_DIR / ".env"
SERVICE_ACCOUNT_PATH = BASE_DIR / "serviceAccountKey.json"

# Auto-restore serviceAccountKey.json from Streamlit Cloud secrets if available
if not SERVICE_ACCOUNT_PATH.exists():
    try:
        if hasattr(st, "secrets") and "firebase_service_account" in st.secrets:
            SERVICE_ACCOUNT_PATH.write_text(
                json.dumps(dict(st.secrets["firebase_service_account"]), ensure_ascii=False, indent=2),
                encoding="utf-8",
            )
    except Exception:
        pass

load_dotenv(dotenv_path=ENV_PATH)

st.set_page_config(
    page_title="TestMitra AI — Admin Panel",
    page_icon="📘",
    layout="wide",
)


def _init_session_state() -> None:
    defaults: dict[str, Any] = {
        "gemini_api_key": st.session_state.get("gemini_api_key", "") or "",
        "available_models": [],
        "selected_model": "gemini-flash-latest",
        "extracted_questions": [],
        "last_test_doc": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def _save_env_var(key: str, value: str) -> None:
    if not ENV_PATH.exists():
        ENV_PATH.touch()
    set_key(str(ENV_PATH), key, value)


def _option_label(idx: int) -> str:
    return chr(65 + idx) if idx < 26 else f"Opt{idx + 1}"


def _render_sidebar() -> tuple[str, str, dict[str, str]]:
    with st.sidebar:
        st.title("📘 TestMitra AI")
        st.caption("Admin Panel • Strict PDF-to-Test Parser")
        st.divider()

        st.subheader("1. Google Gemini API")
        env_key = _get_env("GEMINI_API_KEY")
        api_key = st.text_input(
            "Gemini API Key",
            value=st.session_state["gemini_api_key"] or env_key,
            type="password",
            help="Saved locally in admin_panel/.env",
        )
        if api_key and api_key != env_key:
            _save_env_var("GEMINI_API_KEY", api_key)
        st.session_state["gemini_api_key"] = api_key

        if st.button("Refresh Gemini Models", use_container_width=True, disabled=not api_key):
            try:
                models, default_model = list_gemini_models(api_key)
                st.session_state["available_models"] = models
                st.session_state["selected_model"] = default_model
                st.success(f"Loaded {len(models)} models.")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Failed to list models: {exc}")

        if api_key and not st.session_state["available_models"]:
            try:
                models, default_model = list_gemini_models(api_key)
                st.session_state["available_models"] = models
                st.session_state["selected_model"] = default_model
            except Exception as exc:  # noqa: BLE001
                st.error(f"Could not fetch models: {exc}")
        elif not api_key:
            st.caption("👆 Enter your Gemini API Key above to fetch all available models.")

        model_options = st.session_state["available_models"] or [st.session_state["selected_model"]]
        default_idx = (
            model_options.index(st.session_state["selected_model"])
            if st.session_state["selected_model"] in model_options
            else 0
        )
        selected_model = st.selectbox("Gemini Model (client.models.list)", model_options, index=default_idx)
        st.session_state["selected_model"] = selected_model

        st.divider()
        st.subheader("2. Cloudinary (DI/Reasoning Images)")
        cloud_name = st.text_input("Cloud Name", value=_get_env("CLOUDINARY_CLOUD_NAME"))
        upload_preset = st.text_input("Unsigned Upload Preset", value=_get_env("CLOUDINARY_UPLOAD_PRESET"))
        c_key = st.text_input("API Key (if signed)", value=_get_env("CLOUDINARY_API_KEY"), type="password")
        c_secret = st.text_input("API Secret (if signed)", value=_get_env("CLOUDINARY_API_SECRET"), type="password")
        if st.button("Save Cloudinary Settings", use_container_width=True):
            _save_env_var("CLOUDINARY_CLOUD_NAME", cloud_name)
            _save_env_var("CLOUDINARY_UPLOAD_PRESET", upload_preset)
            _save_env_var("CLOUDINARY_API_KEY", c_key)
            _save_env_var("CLOUDINARY_API_SECRET", c_secret)
            st.success("Saved to .env")

        st.divider()
        st.subheader("3. Firebase Firestore")
        has_secrets = False
        try:
            if hasattr(st, "secrets") and "firebase_service_account" in st.secrets:
                has_secrets = True
        except Exception:
            pass

        if has_secrets:
            st.success("Firebase configured via Streamlit Secrets!")
        elif SERVICE_ACCOUNT_PATH.exists():
            st.success("serviceAccountKey.json detected")
            with st.expander("Update / Replace serviceAccountKey.json"):
                uploaded_sa = st.file_uploader("Upload new serviceAccountKey.json", type=["json"], key="sa_replace")
                if uploaded_sa is not None:
                    SERVICE_ACCOUNT_PATH.write_bytes(uploaded_sa.getvalue())
                    reset_firebase_app()
                    st.success("Saved new serviceAccountKey.json!")
                    st.rerun()
        else:
            st.warning("serviceAccountKey.json not found in /admin_panel")
            uploaded_sa = st.file_uploader("Upload serviceAccountKey.json", type=["json"], key="sa_upload")
            if uploaded_sa is not None:
                SERVICE_ACCOUNT_PATH.write_bytes(uploaded_sa.getvalue())
                reset_firebase_app()
                st.success("Saved serviceAccountKey.json!")
                st.rerun()

    cloudinary_cfg = {
        "cloud_name": cloud_name,
        "upload_preset": upload_preset,
        "api_key": c_key,
        "api_secret": c_secret,
    }
    return api_key, selected_model, cloudinary_cfg


def _get_env(name: str) -> str:
    import os

    return os.getenv(name, "")


def _build_test_document(meta: dict[str, Any], questions: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "category": meta["category"],
        "exam_name": meta["exam_name"],
        "title": meta["title"],
        "duration_mins": int(meta["duration_mins"]),
        "target_questions": int(meta["target_questions"]),
        "positive_marks": float(meta["positive_marks"]),
        "negative_marks": float(meta["negative_marks"]),
        "is_free": bool(meta["is_free"]),
        "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "questions": questions,
    }


def _render_question_editor(idx: int, q: dict[str, Any], cloudinary_cfg: dict[str, str]) -> None:
    options: list[str] = list(q.get("options", []))
    c_idx = int(q.get("correct_index", -1))
    status_badge = "⚠️ Answer Missing (-1)" if c_idx == -1 else f"✅ Ans: {_option_label(c_idx)}"
    header = f"Q{idx + 1} [{q.get('subject', 'General')}] • {len(options)} options • {status_badge}"

    with st.expander(header, expanded=(c_idx == -1)):
        col1, col2 = st.columns([3, 1])
        with col1:
            q["question_hi"] = st.text_area(
                "Question (Hindi)",
                value=q.get("question_hi", ""),
                key=f"q_hi_{idx}",
                height=80,
            )
            q["question_en"] = st.text_area(
                "Question (English)",
                value=q.get("question_en", ""),
                key=f"q_en_{idx}",
                height=80,
            )
        with col2:
            q["subject"] = st.text_input("Subject", value=q.get("subject", "General"), key=f"q_subj_{idx}")
            q["image_url"] = st.text_input("Image URL (DI/Reasoning)", value=q.get("image_url", ""), key=f"q_img_{idx}")
            diag_file = st.file_uploader(
                "Upload Diagram to Cloudinary",
                type=["png", "jpg", "jpeg", "webp"],
                key=f"diag_up_{idx}",
            )
            if diag_file is not None and st.button("Upload Image", key=f"diag_btn_{idx}"):
                try:
                    url = upload_image_to_cloudinary(
                        image_bytes=diag_file.getvalue(),
                        filename=diag_file.name,
                        **cloudinary_cfg,
                    )
                    q["image_url"] = url
                    st.success("Uploaded!")
                    st.rerun()
                except Exception as exc:  # noqa: BLE001
                    st.error(f"Upload failed: {exc}")
            if q.get("image_url"):
                st.image(q["image_url"], width=180)

        st.markdown("**Dynamic Options (2, 4, 5, 6+)**")
        opt_cols_action = st.columns([1, 1, 4])
        with opt_cols_action[0]:
            if st.button("➕ Add Option", key=f"add_opt_{idx}"):
                options.append("")
                q["options"] = options
                st.rerun()
        with opt_cols_action[1]:
            if len(options) > 2 and st.button("➖ Remove Last", key=f"rem_opt_{idx}"):
                options.pop()
                q["options"] = options
                if q["correct_index"] >= len(options):
                    q["correct_index"] = -1
                st.rerun()

        updated_options: list[str] = []
        for opt_i, opt_val in enumerate(options):
            updated_options.append(
                st.text_input(f"Option {_option_label(opt_i)}", value=opt_val, key=f"q_{idx}_opt_{opt_i}")
            )
        q["options"] = updated_options

        choice_indices = [-1] + list(range(len(updated_options)))
        current_choice = q["correct_index"] if q["correct_index"] in choice_indices else -1
        q["correct_index"] = st.selectbox(
            "Correct Answer",
            options=choice_indices,
            index=choice_indices.index(current_choice),
            format_func=lambda x: "Missing (-1) — Needs Manual Selection"
            if x == -1
            else f"{_option_label(x)}: {updated_options[x][:40]}",
            key=f"q_ans_{idx}",
        )
        q["explanation"] = st.text_area(
            "Explanation",
            value=q.get("explanation", ""),
            key=f"q_exp_{idx}",
            height=70,
        )


def _render_converter_tab(api_key: str, selected_model: str, cloudinary_cfg: dict[str, str]) -> None:
    st.subheader("Test Configuration")
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        category = st.selectbox("Category", ["UP Exams", "SSC", "Railways", "Banking", "Other"])
        exam_name = st.text_input("Exam Name", value="UPSSSC PET")
    with c2:
        title = st.text_input("Test Title", value="Full Mock Test 01")
        target_questions = st.number_input("Target Question Count", min_value=1, max_value=300, value=100)
    with c3:
        duration_mins = st.number_input("Paper Timer (mins)", min_value=5, max_value=300, value=120)
        positive_marks = st.number_input("Positive Marks", min_value=0.25, max_value=10.0, value=1.0, step=0.25)
    with c4:
        negative_marks = st.number_input("Negative Marks", min_value=0.0, max_value=5.0, value=0.25, step=0.05)
        is_free = st.checkbox("Free Mock Test", value=True)

    meta = {
        "category": category,
        "exam_name": exam_name,
        "title": title,
        "target_questions": target_questions,
        "duration_mins": duration_mins,
        "positive_marks": positive_marks,
        "negative_marks": negative_marks,
        "is_free": is_free,
    }

    st.divider()
    st.subheader("Strict PDF-Only Parser (temperature = 0.0, 5-Page Chunks)")
    u1, u2 = st.columns(2)
    with u1:
        q_pdf = st.file_uploader("Upload Question Paper PDF (Required)", type=["pdf"])
    with u2:
        ans_pdf = st.file_uploader("Upload Answer Key PDF (Optional)", type=["pdf"])

    if st.button("🚀 Parse PDF Verbatim", type="primary", disabled=(q_pdf is None or not api_key)):
        try:
            answer_text = extract_answer_key_text(ans_pdf.getvalue()) if ans_pdf is not None else ""
            chunks = split_pdf_into_chunks(q_pdf.getvalue(), chunk_size=5)
            progress = st.progress(0, text="Splitting PDF into 5-page chunks...")
            all_questions: list[dict[str, Any]] = []
            for idx, (start_p, end_p, chunk_bytes) in enumerate(chunks, start=1):
                progress.progress(
                    idx / len(chunks),
                    text=f"Parsing pages {start_p}-{end_p} ({idx}/{len(chunks)}) with {selected_model}...",
                )
                batch = parse_pdf_chunk_with_gemini(
                    api_key=api_key,
                    model_name=selected_model,
                    chunk_bytes=chunk_bytes,
                    start_page=start_p,
                    end_page=end_p,
                    answer_key_text=answer_text,
                )
                all_questions.extend(batch)
            st.session_state["extracted_questions"] = all_questions
            progress.empty()
            st.success(f"Extracted {len(all_questions)} questions across {len(chunks)} PDF chunks.")
        except Exception as exc:  # noqa: BLE001
            st.error(f"PDF parsing failed: {exc}")

    questions: list[dict[str, Any]] = st.session_state["extracted_questions"]
    if not questions:
        st.info("Upload a Question Paper PDF and click **Parse PDF Verbatim** to preview, edit, and publish.")
        return

    st.divider()
    st.subheader("Mismatch Validation & Question Review")
    extracted_count = len(questions)
    target_count = int(target_questions)
    m1, m2, m3 = st.columns(3)
    m1.metric("Extracted Questions", extracted_count)
    m2.metric("Target Questions", target_count)
    missing_ans = sum(1 for q in questions if int(q.get("correct_index", -1)) == -1)
    m3.metric("Missing Answers (-1)", missing_ans)

    if extracted_count < target_count:
        st.warning(
            f"Count mismatch: Extracted {extracted_count} questions, which is {target_count - extracted_count} fewer than Target ({target_count})."
        )
    elif extracted_count > target_count:
        st.warning(
            f"Count mismatch: Extracted {extracted_count} questions, which is {extracted_count - target_count} more than Target ({target_count})."
        )
        if st.checkbox(f"Trim extra questions to first {target_count} questions", value=False):
            questions = questions[:target_count]
            st.session_state["extracted_questions"] = questions
    else:
        st.success(f"Exact match: {extracted_count} / {target_count} questions extracted.")

    filter_missing = st.checkbox("Show only questions with missing answer (-1)", value=False)
    for idx, q in enumerate(questions):
        if filter_missing and int(q.get("correct_index", -1)) != -1:
            continue
        _render_question_editor(idx, q, cloudinary_cfg)

    st.divider()
    st.subheader("🛡️ Pre-Publish Quality Scorecard (Zero-Defect Gate)")
    missing_ans = sum(1 for q in questions if int(q.get("correct_index", -1)) == -1)
    bilingual_count = sum(1 for q in questions if q.get("question_hi") and q.get("question_en"))
    total_q = len(questions)

    q_score = 100
    if missing_ans > 0:
        q_score -= min(50, missing_ans * 10)
    if total_q != int(target_questions):
        q_score -= 20
    bilingual_pct = int((bilingual_count / total_q) * 100) if total_q else 0
    if bilingual_pct < 80:
        q_score -= 15
    q_score = max(0, q_score)

    qc1, qc2, qc3, qc4 = st.columns(4)
    qc1.metric("Quality Health Score", f"{q_score}/100", delta="Ready" if q_score >= 80 else "Attention")
    qc2.metric("Answered Questions", f"{total_q - missing_ans}/{total_q}")
    qc3.metric("Bilingual Ratio", f"{bilingual_pct}%")
    qc4.metric("Total Test Marks", f"{total_q * float(positive_marks):.1f}")

    if missing_ans > 0:
        st.error(
            f"⚠️ Pre-Publish Warning: {missing_ans} question(s) have missing correct answers (-1). Please select the correct options before publishing."
        )
    elif total_q != int(target_questions):
        st.warning(f"⚠️ Target Count Mismatch: Target is {target_questions}, but extracted is {total_q}.")
    else:
        st.success("✅ 100% Quality Passed: All questions have valid answers, bilingual text, and marks mapped. Ready for live students!")

    st.divider()
    st.subheader("Export & Publish")
    test_doc = _build_test_document(meta, questions)
    json_str = json.dumps(test_doc, ensure_ascii=False, indent=2)
    safe_filename = f"{exam_name}_{title}".lower().replace(" ", "_") + ".json"

    p1, p2 = st.columns(2)
    with p1:
        st.download_button(
            "⬇️ Download Ready JSON",
            data=json_str.encode("utf-8"),
            file_name=safe_filename,
            mime="application/json",
            use_container_width=True,
        )
    with p2:
        if st.button("☁️ Publish to Firebase Firestore", type="primary", use_container_width=True):
            try:
                doc_id = publish_test_to_firestore(SERVICE_ACCOUNT_PATH, test_doc)
                st.success(f"Published to Firestore `tests` collection! Document ID: `{doc_id}`")
            except Exception as exc:  # noqa: BLE001
                st.error(f"Firestore publish failed: {exc}")


def _render_error_reports_tab() -> None:
    st.subheader("Student Error Reports (Firestore: `error_reports`)")
    has_sa = SERVICE_ACCOUNT_PATH.exists()
    try:
        if hasattr(st, "secrets") and "firebase_service_account" in st.secrets:
            has_sa = True
    except Exception:
        pass

    if not has_sa:
        st.info("Upload `serviceAccountKey.json` in the sidebar to view and resolve student Error Reports.")
        return

    if st.button("🔄 Refresh Error Reports"):
        st.rerun()

    try:
        reports = fetch_error_reports(SERVICE_ACCOUNT_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to fetch error reports: {exc}")
        return

    if not reports:
        st.success("No student error reports found.")
        return

    status_filter = st.radio("Filter by status", ["All", "open", "resolved"], horizontal=True)
    for rep in reports:
        rep_status = rep.get("status", "open")
        if status_filter != "All" and rep_status != status_filter:
            continue
        with st.expander(
            f"[{rep_status.upper()}] Test: {rep.get('test_title', rep.get('test_id', 'Unknown'))} "
            f"• Q{rep.get('question_index', '?')} • {rep.get('reason', 'Reported Issue')}"
        ):
            st.write(f"**Report ID:** `{rep.get('id')}`")
            st.write(f"**User:** {rep.get('user_id', 'Anonymous')}")
            st.write(f"**Details:** {rep.get('details', '')}")
            if rep_status != "resolved":
                note = st.text_input("Admin Resolution Note", key=f"note_{rep['id']}")
                if st.button("Mark Resolved", key=f"res_{rep['id']}"):
                    try:
                        resolve_error_report(SERVICE_ACCOUNT_PATH, rep["id"], note)
                        st.success("Marked as resolved!")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to resolve: {exc}")


def _render_manage_tests_tab() -> None:
    st.subheader("📚 Published Tests Library (Firestore: `tests`)")
    has_sa = SERVICE_ACCOUNT_PATH.exists()
    try:
        if hasattr(st, "secrets") and "firebase_service_account" in st.secrets:
            has_sa = True
    except Exception:
        pass

    if not has_sa:
        st.info("Upload `serviceAccountKey.json` in the sidebar to manage published tests.")
        return

    col_ref, col_search = st.columns([1, 3])
    with col_ref:
        if st.button("🔄 Refresh Tests"):
            st.rerun()

    try:
        all_tests = fetch_published_tests(SERVICE_ACCOUNT_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to fetch published tests: {exc}")
        return

    if not all_tests:
        st.info("No published tests found in Firestore. Parse and publish a test from the PDF Converter tab.")
        return

    total_count = len(all_tests)
    live_count = sum(1 for t in all_tests if t.get("is_active", True) is not False)
    draft_count = total_count - live_count

    m1, m2, m3 = st.columns(3)
    m1.metric("Total Published Tests", total_count)
    m2.metric("Live in Student App", live_count)
    m3.metric("Draft / Hidden", draft_count)

    with col_search:
        search_query = st.text_input("🔍 Search by title, exam, or category", value="").lower().strip()

    filtered_tests = [
        t for t in all_tests
        if not search_query
        or search_query in str(t.get("title", "")).lower()
        or search_query in str(t.get("exam_name", "")).lower()
        or search_query in str(t.get("category", "")).lower()
    ]

    st.caption(f"Showing {len(filtered_tests)} of {total_count} tests.")

    for t in filtered_tests:
        test_id = t["id"]
        title = t.get("title", "Untitled Test")
        exam_name = t.get("exam_name", "General")
        category = t.get("category", "General")
        is_active = t.get("is_active", True) is not False
        status_badge = "🟢 LIVE" if is_active else "⚪ DRAFT"
        questions = t.get("questions", [])
        q_count = len(questions)
        duration = t.get("duration_mins", 120)
        pos_marks = t.get("positive_marks", 1.0)
        neg_marks = t.get("negative_marks", 0.25)
        is_free = t.get("is_free", True)

        with st.expander(f"[{status_badge}] {exam_name} — {title} ({q_count} Qs, {duration}m) • ID: `{test_id}`"):
            action_col1, action_col2, action_col3 = st.columns([2, 2, 2])
            with action_col1:
                toggle_label = "Switch to Draft (Hide from App)" if is_active else "Switch to Live (Show in App)"
                if st.button(f"🔄 {toggle_label}", key=f"tog_{test_id}"):
                    try:
                        update_test_in_firestore(SERVICE_ACCOUNT_PATH, test_id, {"is_active": not is_active})
                        st.success(f"Status changed to {'LIVE' if not is_active else 'DRAFT'}!")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to update status: {exc}")

            with action_col2:
                backup_json = json.dumps(t, ensure_ascii=False, indent=2)
                st.download_button(
                    "💾 Download JSON Backup",
                    data=backup_json.encode("utf-8"),
                    file_name=f"{exam_name}_{title}_{test_id}.json".replace(" ", "_"),
                    mime="application/json",
                    key=f"dl_{test_id}",
                )

            with action_col3:
                del_confirm = st.checkbox(f"Confirm Delete", key=f"del_chk_{test_id}")
                if st.button(f"🗑️ Delete Test", key=f"del_btn_{test_id}", disabled=not del_confirm):
                    try:
                        delete_test_from_firestore(SERVICE_ACCOUNT_PATH, test_id)
                        st.success("Test deleted permanently from Firestore!")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Failed to delete: {exc}")

            st.markdown("---")
            st.markdown("##### ⚙️ Test Properties & Marks Scheme")
            prop_col1, prop_col2, prop_col3, prop_col4 = st.columns(4)
            with prop_col1:
                new_title = st.text_input("Title", value=title, key=f"meta_title_{test_id}")
                new_exam = st.text_input("Exam Name", value=exam_name, key=f"meta_exam_{test_id}")
            with prop_col2:
                new_category = st.text_input("Category", value=category, key=f"meta_cat_{test_id}")
                new_duration = st.number_input("Timer (mins)", min_value=5, max_value=300, value=int(duration), key=f"meta_dur_{test_id}")
            with prop_col3:
                new_pos = st.number_input("+ Marks", min_value=0.25, max_value=10.0, value=float(pos_marks), step=0.25, key=f"meta_pos_{test_id}")
                new_neg = st.number_input("- Marks", min_value=0.0, max_value=5.0, value=float(neg_marks), step=0.05, key=f"meta_neg_{test_id}")
            with prop_col4:
                new_free = st.checkbox("Free Test", value=is_free, key=f"meta_free_{test_id}")
                if st.button("💾 Save Metadata", key=f"save_meta_{test_id}"):
                    try:
                        update_test_in_firestore(
                            SERVICE_ACCOUNT_PATH,
                            test_id,
                            {
                                "title": new_title,
                                "exam_name": new_exam,
                                "category": new_category,
                                "duration_mins": new_duration,
                                "positive_marks": new_pos,
                                "negative_marks": new_neg,
                                "is_free": new_free,
                            },
                        )
                        st.success("Metadata updated successfully in Firestore!")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Update failed: {exc}")

            st.markdown("---")
            st.markdown(f"##### 📝 Question Inspector & Hotfix Editor ({q_count} Questions)")
            if not questions:
                st.caption("No questions in this test.")
            else:
                q_select_idx = st.selectbox(
                    f"Select Question to inspect / hotfix (Test: {test_id[:8]})",
                    options=list(range(len(questions))),
                    format_func=lambda i: f"Q{i + 1}: {questions[i].get('question_hi', questions[i].get('question_en', ''))[:60]}...",
                    key=f"q_sel_{test_id}",
                )
                target_q = questions[q_select_idx]
                q_col_hi, q_col_en = st.columns(2)
                with q_col_hi:
                    target_q["question_hi"] = st.text_area(
                        "Question Hindi",
                        value=target_q.get("question_hi", ""),
                        key=f"edit_hi_{test_id}_{q_select_idx}",
                    )
                with q_col_en:
                    target_q["question_en"] = st.text_area(
                        "Question English",
                        value=target_q.get("question_en", ""),
                        key=f"edit_en_{test_id}_{q_select_idx}",
                    )

                opts = target_q.get("options", [])
                st.markdown("**Options:**")
                opt_cols = st.columns(min(len(opts), 4) or 1)
                for opt_i, opt_val in enumerate(opts):
                    col_idx = opt_i % len(opt_cols)
                    with opt_cols[col_idx]:
                        opts[opt_i] = st.text_input(
                            f"Opt {_option_label(opt_i)}",
                            value=opt_val,
                            key=f"edit_opt_{test_id}_{q_select_idx}_{opt_i}",
                        )
                target_q["options"] = opts

                ans_col1, ans_col2 = st.columns([1, 2])
                with ans_col1:
                    choice_opts = [-1] + list(range(len(opts)))
                    curr_ans = target_q.get("correct_index", -1)
                    if curr_ans not in choice_opts:
                        curr_ans = -1
                    target_q["correct_index"] = st.selectbox(
                        "Correct Answer Index",
                        options=choice_opts,
                        index=choice_opts.index(curr_ans),
                        format_func=lambda x: "Missing (-1)" if x == -1 else f"{_option_label(x)}: {opts[x][:25]}",
                        key=f"edit_ans_{test_id}_{q_select_idx}",
                    )
                with ans_col2:
                    target_q["explanation"] = st.text_input(
                        "Explanation",
                        value=target_q.get("explanation", ""),
                        key=f"edit_exp_{test_id}_{q_select_idx}",
                    )

                if st.button("⚡ Save Question Hotfix to Cloud", key=f"save_q_{test_id}_{q_select_idx}"):
                    try:
                        questions[q_select_idx] = target_q
                        update_test_in_firestore(
                            SERVICE_ACCOUNT_PATH,
                            test_id,
                            {"questions": questions},
                        )
                        st.success(f"Q{q_select_idx + 1} hotfix saved directly to Firestore!")
                        st.rerun()
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"Question update failed: {exc}")


def _render_app_config_tab() -> None:
    st.subheader("🗓️ Sunday Mega Mock & App Remote Controller (Firestore: `app_config/home_config`)")
    has_sa = SERVICE_ACCOUNT_PATH.exists()
    try:
        if hasattr(st, "secrets") and "firebase_service_account" in st.secrets:
            has_sa = True
    except Exception:
        pass

    if not has_sa:
        st.info("Upload `serviceAccountKey.json` in the sidebar to configure the live mobile/web app.")
        return

    try:
        current_cfg = fetch_app_config(SERVICE_ACCOUNT_PATH)
        published_tests = fetch_published_tests(SERVICE_ACCOUNT_PATH)
    except Exception as exc:  # noqa: BLE001
        st.error(f"Failed to fetch config or tests: {exc}")
        return

    st.info("💡 **25-Year EdTech Pro Tip:** यहाँ से आप बिना मोबाइल ऐप दोबारा अपडेट या रीबिल्ड किए, सीधे हज़ारों छात्रों की ऐप स्क्रीन पर संडे मेगा मॉक और नोटिस बैनर रियल-टाइम कंट्रोल कर सकते हैं।")

    st.markdown("#### 🔴 1. Sunday Mega Mock Event Controller")
    c1, c2 = st.columns(2)
    with c1:
        mega_active = st.checkbox("Show Sunday Mega Mock Card on Home Screen", value=bool(current_cfg.get("mega_mock_active", True)))
        mega_title = st.text_input("Mega Mock Title", value=str(current_cfg.get("mega_mock_title", "All-UP Sunday Mega Mock 01")))
        mega_exam = st.text_input("Target Exam Badge", value=str(current_cfg.get("mega_mock_exam", "UPSSSC PET 2026 / UP Police")))
    with c2:
        mega_date = st.text_input("Target Live Date / Timing", value=str(current_cfg.get("mega_mock_target_date", "Sunday, 9:00 AM - 11:59 PM")))
        mega_participants = st.number_input("Displayed Participants Count", min_value=100, max_value=500000, value=int(current_cfg.get("mega_mock_participants", 18400)), step=100)
        mega_cutoff = st.number_input("Expected Safe Cutoff Score", min_value=10.0, max_value=200.0, value=float(current_cfg.get("mega_mock_cutoff", 74.5)), step=0.5)

    test_options = [""] + [t["id"] for t in published_tests]
    curr_test_id = current_cfg.get("mega_mock_test_id", "")
    if curr_test_id not in test_options:
        test_options.append(curr_test_id)

    selected_test_id = st.selectbox(
        "Attached Mock Test (When student taps 'Start Mock Test')",
        options=test_options,
        index=test_options.index(curr_test_id),
        format_func=lambda tid: "Default (First Available Test)" if not tid else next(
            (f"{t.get('exam_name')} — {t.get('title')} (ID: {tid[:8]})" for t in published_tests if t["id"] == tid),
            f"Test ID: {tid}",
        ),
    )

    st.markdown("---")
    st.markdown("#### 📢 2. App Announcement / Notice Ticker")
    n1, n2 = st.columns([1, 3])
    with n1:
        notice_active = st.checkbox("Show Announcement Banner", value=bool(current_cfg.get("notice_active", True)))
        notice_tag = st.selectbox("Notice Tag", ["NEW", "LIVE", "ALERT", "UPDATE"], index=["NEW", "LIVE", "ALERT", "UPDATE"].index(current_cfg.get("notice_tag", "NEW")) if current_cfg.get("notice_tag") in ["NEW", "LIVE", "ALERT", "UPDATE"] else 0)
    with n2:
        notice_text = st.text_area(
            "Notice Headline",
            value=str(current_cfg.get("notice_text", "🔥 All-UP Sunday Mega Mock Test Registration is Live! Practice with State Rank Predictor.")),
            height=85,
        )

    st.markdown("---")
    if st.button("💾 Save Remote Config & Sync to App", type="primary", use_container_width=True):
        new_cfg = {
            "mega_mock_active": mega_active,
            "mega_mock_title": mega_title,
            "mega_mock_exam": mega_exam,
            "mega_mock_target_date": mega_date,
            "mega_mock_participants": mega_participants,
            "mega_mock_cutoff": mega_cutoff,
            "mega_mock_test_id": selected_test_id,
            "notice_active": notice_active,
            "notice_tag": notice_tag,
            "notice_text": notice_text,
        }
        try:
            save_app_config(SERVICE_ACCOUNT_PATH, new_cfg)
            st.success("✅ Remote Config saved to Firestore `app_config/home_config`! All student devices will load these settings instantly on launch/refresh.")
            st.rerun()
        except Exception as exc:  # noqa: BLE001
            st.error(f"Failed to save app config: {exc}")


def main() -> None:
    _init_session_state()
    api_key, selected_model, cloudinary_cfg = _render_sidebar()

    st.title("TestMitra AI — Admin Mission Control")
    tab1, tab2, tab3, tab4 = st.tabs([
        "📄 PDF-to-Test Converter",
        "📚 Manage Published Tests",
        "🗓️ Mega Mock & App Config",
        "🚩 Student Error Reports",
    ])
    with tab1:
        _render_converter_tab(api_key, selected_model, cloudinary_cfg)
    with tab2:
        _render_manage_tests_tab()
    with tab3:
        _render_app_config_tab()
    with tab4:
        _render_error_reports_tab()


if __name__ == "__main__":
    main()
