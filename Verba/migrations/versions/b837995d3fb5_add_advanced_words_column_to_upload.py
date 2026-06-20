"""add advanced_words column to upload

Revision ID: b837995d3fb5
Revises: 
Create Date: 2025-07-11 03:46:52.656949

"""
from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision: str = 'b837995d3fb5'
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade():
    op.add_column('upload', sa.Column('pitch_variation_percent', sa.Integer(), nullable=True))
def downgrade():
    op.drop_column('upload', 'pitch_variation_percent')