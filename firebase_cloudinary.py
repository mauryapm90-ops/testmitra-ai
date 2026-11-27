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


def get_firestore_client(service_account_path: Path) -> firestore.Client:
    """Initialize (once) and return a Firestore client from serviceAccountKey.json."""
    if not service_account_path.exists():
        raise FileNotFoundError(f"Service account key not found at {service_account_path}")

    if not firebase_admin._apps:
        cred = credentials.Certificate(str(service_account_path))
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
