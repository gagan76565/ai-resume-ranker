"""Initial v2 schema: screenings and candidates.

Revision ID: 0001
Revises:
Create Date: 2026-11-19
"""

import sqlalchemy as sa
from alembic import op

revision = "0001"
down_revision = None
branch_labels = None
depends_on = None


def upgrade():
    op.create_table(
        "screenings",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column("job_title", sa.String(120), nullable=False),
        sa.Column("job_description", sa.Text, nullable=False),
        sa.Column("candidate_count", sa.Integer, nullable=False, server_default="0"),
        sa.Column("must_have", sa.JSON, nullable=False),
        sa.Column("settings", sa.JSON, nullable=False),
        sa.Column("scoring_version", sa.String(20), nullable=False),
        sa.Column("legacy_id", sa.Integer, nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.UniqueConstraint("legacy_id", name="uq_screenings_legacy_id"),
    )
    op.create_index("ix_screenings_created_at", "screenings", ["created_at"])
    op.create_table(
        "candidates",
        sa.Column("id", sa.Integer, primary_key=True, autoincrement=True),
        sa.Column(
            "screening_id",
            sa.Integer,
            sa.ForeignKey("screenings.id", ondelete="CASCADE", name="fk_candidates_screening_id"),
            nullable=False,
        ),
        sa.Column("rank", sa.Integer, nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("final_score", sa.Float, nullable=False),
        sa.Column("tfidf_score", sa.Float, nullable=True),
        sa.Column("skill_coverage", sa.Float, nullable=True),
        sa.Column("section_signal", sa.Float, nullable=True),
        sa.Column("label", sa.String(40), nullable=False),
        sa.Column("explanation", sa.Text, nullable=False),
        sa.Column("breakdown", sa.JSON, nullable=False),
        sa.Column("shortlisted", sa.Boolean, nullable=False, server_default=sa.false()),
        sa.Column("notes", sa.Text, nullable=False, server_default=""),
        sa.Column("updated_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index("ix_candidates_screening_id", "candidates", ["screening_id"])


def downgrade():
    op.drop_index("ix_candidates_screening_id", table_name="candidates")
    op.drop_table("candidates")
    op.drop_index("ix_screenings_created_at", table_name="screenings")
    op.drop_table("screenings")
