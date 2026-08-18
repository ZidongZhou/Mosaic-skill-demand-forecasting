from __future__ import annotations

LOOKBACK = 12
HORIZON = 3
TRAIN_STARTS = [12, 15, 18]
VALIDATION_STARTS = [21]
TEST_STARTS = [24, 27, 30, 33]

def block_label(months: list[str], start: int, horizon: int = HORIZON) -> str:
    return f"{months[start]} to {months[start + horizon - 1]}"
