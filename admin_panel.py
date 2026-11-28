"""TestMitra AI — Admin Panel (1-Click PDF-to-Test Converter & Error Report Manager)."""

from __future__ import annotations

from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from dotenv import load_dotenv, set_key
import streamlit as st

from firebase_cloudinary import (
    fetch_error_reports,
    publish_test_to_firestore,
    reset_firebase_app,
    resolve_error_report,
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


def main() -> None:
    _init_session_state()
    api_key, selected_model, cloudinary_cfg = _render_sidebar()

    st.title("TestMitra AI — PDF-to-Test Admin Panel")
    tab1, tab2 = st.tabs(["📄 PDF-to-Test Converter", "🚩 Student Error Reports"])
    with tab1:
        _render_converter_tab(api_key, selected_model, cloudinary_cfg)
    with tab2:
        _render_error_reports_tab()


if __name__ == "__main__":
    main()
