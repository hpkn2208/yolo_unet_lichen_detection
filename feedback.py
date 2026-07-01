import streamlit as st
import numpy as np
from PIL import Image
import json
import zipfile
from datetime import datetime
from pathlib import Path


def resolve_folder(feedback_type, reason=None):
    base = Path("feedback_data")
    if feedback_type == "Correct":
        folder = base / "Success_Data"
    elif feedback_type == "Incorrect":
        mapping = {
            "YOLO missed lesion":   base / "YOLO_FN",
            "YOLO false alarm":     base / "YOLO_FP",
            "Wrong lichen mask":    base / "UNet_Bad_Mask",
            "Wrong class":          base / "UNet_Wrong_Class",
        }
        folder = mapping.get(reason, base / "General_Feedback")
    else:
        folder = base / "General_Feedback"
    folder.mkdir(parents=True, exist_ok=True)
    return folder


def save_feedback(image_array, overlay_array, image_id, feedback_type,
                  reason, correct_class, predictions, uploaded_filename, models_used):
    folder = resolve_folder(feedback_type, reason)
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")

    img_path     = folder / f"{image_id}_{ts}_original.png"
    overlay_path = folder / f"{image_id}_{ts}_overlay.png"
    meta_path    = folder / f"{image_id}_{ts}_meta.json"

    Image.fromarray(image_array.astype(np.uint8)).save(img_path)
    if overlay_array is not None:
        Image.fromarray(overlay_array.astype(np.uint8)).save(overlay_path)

    meta = {
        "image_id":          image_id,
        "original_filename": uploaded_filename,
        "feedback":          feedback_type,
        "reason":            reason,
        "correct_class":     correct_class,
        "timestamp":         datetime.now().isoformat(),
        "predictions":       predictions,
        "models_used":       models_used,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    return str(img_path)


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


def create_feedback_zip():
    feedback_dir = Path("feedback_data")
    if not feedback_dir.exists() or not any(feedback_dir.rglob("*")):
        return None
    zip_path = Path("feedback_data.zip")
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in feedback_dir.rglob("*"):
            if p.is_file():
                zf.write(p, p.relative_to(feedback_dir.parent))
    return zip_path
