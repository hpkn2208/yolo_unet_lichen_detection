import csv
import io
import json
import uuid
import zipfile
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import numpy as np
import streamlit as st
from sqlalchemy import create_engine, text

import storage

CATEGORY_MAP = {
    "YOLO missed lesion": "YOLO_FN",
    "YOLO false alarm": "YOLO_FP",
    "Wrong lichen mask": "UNet_Bad_Mask",
    "Wrong class": "UNet_Wrong_Class",
}


@lru_cache(maxsize=1)
def _engine():
    return create_engine(st.secrets["postgres_url"])


def init_feedback_table() -> None:
    with _engine().begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS feedback (
                id TEXT PRIMARY KEY,
                image_id TEXT NOT NULL,
                original_filename TEXT,
                category TEXT NOT NULL,
                feedback_type TEXT NOT NULL,
                reason TEXT,
                correct_class TEXT,
                original_path TEXT NOT NULL,
                overlay_path TEXT,
                predictions TEXT,
                models_used TEXT,
                created_at TEXT NOT NULL
            )
        """))


def resolve_category(feedback_type, reason=None):
    if feedback_type == "Correct":
        return "Success_Data"
    if feedback_type == "Incorrect":
        return CATEGORY_MAP.get(reason, "General_Feedback")
    return "General_Feedback"


def save_feedback(image_array, overlay_array, image_id, feedback_type,
                  reason, correct_class, predictions, uploaded_filename, models_used):
    category = resolve_category(feedback_type, reason)
    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

    original_key = storage.upload_image(image_array, f"{category}/{image_id}_{ts}_original.png")
    overlay_key = None
    if overlay_array is not None:
        overlay_key = storage.upload_image(overlay_array, f"{category}/{image_id}_{ts}_overlay.png")

    with _engine().begin() as conn:
        conn.execute(
            text("""INSERT INTO feedback
                    (id, image_id, original_filename, category, feedback_type, reason,
                     correct_class, original_path, overlay_path, predictions, models_used, created_at)
                    VALUES (:id, :image_id, :original_filename, :category, :feedback_type, :reason,
                            :correct_class, :original_path, :overlay_path, :predictions, :models_used, :created_at)"""),
            {
                "id": str(uuid.uuid4()), "image_id": image_id, "original_filename": uploaded_filename,
                "category": category, "feedback_type": feedback_type, "reason": reason,
                "correct_class": correct_class, "original_path": original_key, "overlay_path": overlay_key,
                "predictions": json.dumps(predictions), "models_used": json.dumps(models_used or {}),
                "created_at": datetime.now(timezone.utc).isoformat(),
            },
        )

    return original_key


def render_feedback_widget(col, image_array, overlay_array, image_id,
                           predictions, uploaded_filename, models_used=None):
    with col:
        st.divider()
        st.markdown("📋 **Feedback**")

        fb_key   = f"fb_{image_id}"
        re_key   = f"re_{image_id}"
        cls_key  = f"cls_{image_id}"
        sub_key  = f"sub_{image_id}"
        done_key = f"done_{image_id}"

        for k, v in [(fb_key, "Correct"), (re_key, None),
                     (cls_key, None), (done_key, False)]:
            if k not in st.session_state:
                st.session_state[k] = v

        if st.session_state[done_key]:
            st.success("✓ Feedback recorded.")
            return

        feedback = st.radio("Is the prediction correct?", ["Correct", "Incorrect"],
                            key=fb_key, horizontal=True)

        reason = None
        correct_class = None
        can_submit = feedback == "Correct"

        if feedback == "Incorrect":
            reason = st.radio(
                "What was wrong?",
                ["YOLO missed lesion", "YOLO false alarm",
                 "Wrong lichen mask", "Wrong class"],
                key=re_key,
            )
            if reason == "Wrong class":
                correct_class = st.selectbox(
                    "Correct class?", ["Normal", "Lichen", "Other"], key=cls_key)
                can_submit = bool(correct_class)
            else:
                can_submit = bool(reason)

        if can_submit:
            if st.button("Submit Feedback", key=sub_key, use_container_width=True):
                try:
                    save_feedback(image_array, overlay_array, image_id,
                                  feedback, reason, correct_class,
                                  predictions, uploaded_filename, models_used or {})
                    st.session_state[done_key] = True
                    st.rerun()
                except Exception as e:
                    st.error(f"Error saving feedback: {e}")
        else:
            st.info("Fill in feedback options to enable submit.")


def count_feedback() -> int:
    with _engine().connect() as conn:
        return conn.execute(text("SELECT COUNT(*) FROM feedback")).scalar()


def delete_all_feedback() -> None:
    """Permanently removes every feedback row and its R2 images. Cannot be undone."""
    with _engine().begin() as conn:
        original_keys = conn.execute(text("SELECT original_path FROM feedback")).scalars().all()
        overlay_keys = conn.execute(text("SELECT overlay_path FROM feedback WHERE overlay_path IS NOT NULL")).scalars().all()
        conn.execute(text("DELETE FROM feedback"))

    for key in [*original_keys, *overlay_keys]:
        storage.delete_image(key)


def create_feedback_zip():
    """Bundles every feedback image AND its metadata (label, category, correct
    class, predictions, timestamp — the full `feedback` table) into one ZIP, so
    anyone with access to this app can self-serve the complete training-ready
    dataset without needing Supabase access or running a separate script."""
    keys = storage.list_keys()
    if not keys:
        return None

    zip_path = Path("feedback_data.zip")
    if zip_path.exists():
        zip_path.unlink()

    with _engine().connect() as conn:
        rows = conn.execute(text("SELECT * FROM feedback ORDER BY created_at")).mappings().all()

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for key in keys:
            arcname = key[len(storage.PREFIX) + 1:]  # strip "research-app/" prefix
            zf.writestr(arcname, storage.download_bytes(key))

        if rows:
            csv_buf = io.StringIO()
            writer = csv.DictWriter(csv_buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for row in rows:
                writer.writerow(dict(row))
            zf.writestr("metadata.csv", csv_buf.getvalue())

    return zip_path
