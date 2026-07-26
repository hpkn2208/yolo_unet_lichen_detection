-- Feedback schema. Also created automatically by feedback.py::init_feedback_table()
-- on first run — this file is a reference for running directly in the
-- Supabase SQL editor if you want the table to exist before first boot.

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
);
