"""Helpers for Cloudinary diagram uploads and Firebase Firestore publishing/error reports."""

from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any

import firebase_admin
from firebase_admin import credentials, firestore
import requests


def upload_image_to_cloudinary(
    image_bytes: bytes,
    filename: str,
    cloud_name: str,
    upload_preset: str = "",
    api_key: str = "",
    api_secret: str = "",
) -> str:
    """Upload a diagram/graph image to Cloudinary and return its secure_url."""
    if not cloud_name.strip():
        raise ValueError("Cloudinary Cloud Name is required.")

    url = f"https://api.cloudinary.com/v1_1/{cloud_name.strip()}/image/upload"
    files = {"file": (filename, image_bytes)}

    if upload_preset.strip():
        data = {"upload_preset": upload_preset.strip()}
    elif api_key.strip() and api_secret.strip():
        timestamp = str(int(time.time()))
        to_sign = f"timestamp={timestamp}{api_secret.strip()}"
        signature = hashlib.sha1(to_sign.encode("utf-8")).hexdigest()
        data = {
            "api_key": api_key.strip(),
            "timestamp": timestamp,
            "signature": signature,
        }
    else:
        raise ValueError("Provide either an unsigned Upload Preset or Cloudinary API Key + Secret.")

    resp = requests.post(url, data=data, files=files, timeout=30)
    resp.raise_for_status()
    payload = resp.json()
    return str(payload["secure_url"])


def reset_firebase_app() -> None:
    """Clear cached firebase_admin app so newly uploaded credentials take effect immediately."""
    for app in list(firebase_admin._apps.values()):
        try:
            firebase_admin.delete_app(app)
        except Exception:
            pass


def get_firestore_client(service_account_path: Path | None = None) -> firestore.Client:
    """Initialize (once) and return a Firestore client from serviceAccountKey.json or Streamlit secrets."""
    if not firebase_admin._apps:
        cred = None
        # Check Streamlit secrets first if available
        try:
            import streamlit as st
            if hasattr(st, "secrets") and "firebase_service_account" in st.secrets:
                sa_dict = dict(st.secrets["firebase_service_account"])
                cred = credentials.Certificate(sa_dict)
        except Exception:
            pass

        if cred is None:
            if service_account_path and Path(service_account_path).exists():
                cred = credentials.Certificate(str(service_account_path))
            else:
                raise FileNotFoundError(f"Service account key not found at {service_account_path}")

        firebase_admin.initialize_app(cred)
    return firestore.client()


def publish_test_to_firestore(service_account_path: Path, test_doc: dict[str, Any]) -> str:
    """Publish a single-document mock test to Firestore collection `tests`."""
    db = get_firestore_client(service_account_path)
    _, doc_ref = db.collection("tests").add(test_doc)
    return str(doc_ref.id)


def fetch_error_reports(service_account_path: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch student error_reports from Firestore ordered by newest first."""
    db = get_firestore_client(service_account_path)
    docs = (
        db.collection("error_reports")
        .order_by("created_at", direction=firestore.Query.DESCENDING)
        .limit(limit)
        .stream()
    )
    results: list[dict[str, Any]] = []
    for doc in docs:
        data = doc.to_dict() or {}
        data["id"] = doc.id
        results.append(data)
    return results


def resolve_error_report(service_account_path: Path, report_id: str, admin_note: str = "") -> None:
    """Mark a student error report as resolved in Firestore."""
    db = get_firestore_client(service_account_path)
    db.collection("error_reports").document(report_id).update(
        {
            "status": "resolved",
            "admin_note": admin_note,
            "resolved_at": firestore.SERVER_TIMESTAMP,
        }
    )


def fetch_published_tests(service_account_path: Path, limit: int = 100) -> list[dict[str, Any]]:
    """Fetch all published tests from Firestore collection `tests`."""
    db = get_firestore_client(service_account_path)
    docs = db.collection("tests").limit(limit).stream()
    tests: list[dict[str, Any]] = []
    for doc in docs:
        data = doc.to_dict() or {}
        data["id"] = doc.id
        tests.append(data)
    # Sort in memory: newest first if created_at exists
    tests.sort(
        key=lambda t: str(t.get("created_at") or t.get("createdAt") or ""),
        reverse=True,
    )
    return tests


def update_test_in_firestore(
    service_account_path: Path, test_id: str, updated_fields: dict[str, Any]
) -> None:
    """Update fields of an existing published test in Firestore."""
    db = get_firestore_client(service_account_path)
    updated_fields["updated_at"] = firestore.SERVER_TIMESTAMP
    db.collection("tests").document(test_id).update(updated_fields)


def delete_test_from_firestore(service_account_path: Path, test_id: str) -> None:
    """Delete a test document from Firestore."""
    db = get_firestore_client(service_account_path)
    db.collection("tests").document(test_id).delete()


def fetch_app_config(service_account_path: Path) -> dict[str, Any]:
    """Fetch app-wide remote configuration (Sunday Mega Mock, Notice Ticker) from Firestore `app_config/home_config`."""
    db = get_firestore_client(service_account_path)
    doc_ref = db.collection("app_config").document("home_config")
    snap = doc_ref.get()
    if snap.exists:
        data = snap.to_dict() or {}
        data["id"] = snap.id
        return data

    # Default fallback configuration
    return {
        "mega_mock_active": True,
        "mega_mock_title": "All-UP Sunday Mega Mock 01",
        "mega_mock_exam": "UPSSSC PET / UP Police",
        "mega_mock_target_date": "Sunday, 9:00 AM",
        "mega_mock_participants": 18400,
        "mega_mock_cutoff": 74.5,
        "mega_mock_test_id": "",
        "notice_active": True,
        "notice_text": "🔥 All-UP Sunday Mega Mock Test Registration is Live! Practice with State Rank Predictor.",
        "notice_tag": "NEW",
    }


def save_app_config(service_account_path: Path, config_data: dict[str, Any]) -> None:
    """Save or update app-wide remote configuration in Firestore `app_config/home_config`."""
    db = get_firestore_client(service_account_path)
    config_data["updated_at"] = firestore.SERVER_TIMESTAMP
    db.collection("app_config").document("home_config").set(config_data, merge=True)

