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
    comment TEXT,
    evidence_paths TEXT,  -- JSON array of R2 keys, e.g. biopsy/lab report attachments
    created_at TEXT NOT NULL
);

-- Backfill for a table created before comment/evidence-attachment support existed
-- (feedback.py::init_feedback_table() also runs these on every app boot).
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS comment TEXT;
ALTER TABLE feedback ADD COLUMN IF NOT EXISTS evidence_paths TEXT;

-- NOTE: the live table also has a feedback_by column from a since-reverted
-- sign-in feature (testuser1/2/3 reviewer tracking). It was deliberately left
-- in place rather than dropped, to avoid destroying the historical values it
-- already has — new rows just don't populate it anymore. Not recreated here
-- since a brand-new deployment no longer needs it.
