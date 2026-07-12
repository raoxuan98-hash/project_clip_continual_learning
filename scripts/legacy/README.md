# Legacy Entry Points

This directory preserves superseded root entry points and compatibility
wrappers for historical commands. Current continual-learning work should use
`main_incremental.py`, `main_joint.py`, and the canonical tools directly under
`scripts/`.

Files here should not receive new features. If a historical experiment still
depends on one, keep the corresponding launcher and chat-history reference
together until the result package is frozen.
