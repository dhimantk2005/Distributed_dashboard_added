"""
Parses raw stdout/stderr lines from train.py into structured log entries.
Also detects METRIC: JSON lines for structured metric updates.
"""

import re
import json
import time
from typing import Optional, Tuple

# ── Regex patterns ────────────────────────────────────────────────────

RANK_RE = re.compile(r'\[Rank\s+(\d+)\]')
EPOCH_RE = re.compile(r'\[Epoch\s+(\d+)/(\d+)\]')
TQDM_RE = re.compile(r'\d+%\|[█▌ ]*\|')  # tqdm progress bars


def classify_level(line: str) -> str:
    """Infer log level from line content."""
    lower = line.lower()

    if any(k in lower for k in [
        'error:', 'fatal:', 'exception', 'traceback (most recent',
        'failed to', 'could not', 'no such file', 'sigkill', 'sigterm',
    ]):
        return 'error'

    if any(k in lower for k in [
        'warning:', 'warn:', 'retry in', 'retrying', 'deprecated',
        'slow batch', 'heartbeat delay', 'interrupted',
    ]):
        return 'warning'

    if any(k in lower for k in [
        'checkpoint saved', 'training complete', 'connected!',
        'all nodes synchronized', 'global metrics aggregated',
        'training complete!',
    ]):
        return 'success'

    # Epoch summary lines from rank 0
    if EPOCH_RE.search(line) and ('loss:' in lower or 'acc:' in lower):
        return 'success'

    if any(k in lower for k in [
        'nccl:', 'distributedsampler:', 'memory allocated',
        'allreduce on', 'shuffled with seed',
    ]):
        return 'debug'

    return 'info'


def parse_rank(line: str) -> int:
    """Extract rank number from [Rank N] pattern, default 0."""
    m = RANK_RE.search(line)
    return int(m.group(1)) if m else 0


def is_noise_line(line: str) -> bool:
    """Return True for lines that should be silently discarded."""
    if not line.strip():
        return True
    # tqdm bar lines contain \r or %-block patterns
    if '\r' in line:
        return True
    if TQDM_RE.search(line):
        return True
    # Very long lines of spaces / progress chars
    if line.count('█') + line.count('▌') > 5:
        return True
    return False


def parse_line(raw: str) -> Tuple[Optional[dict], Optional[dict]]:
    """
    Parse one stdout line from train.py.

    Returns:
        (log_entry, metric_data)
        Either value may be None.
        log_entry  — structured log dict (without 'id', assigned later)
        metric_data — parsed METRIC: JSON dict
    """
    line = raw.rstrip('\n\r').strip()

    if is_noise_line(line):
        return None, None

    # ── METRIC: lines ────────────────────────────────────────────────
    if line.startswith('METRIC:'):
        raw_json = line[len('METRIC:'):].strip()
        try:
            data = json.loads(raw_json)
            return None, data
        except json.JSONDecodeError:
            pass
        # Malformed METRIC line — log as debug
        return {
            "timestamp": int(time.time() * 1000),
            "rank": 0,
            "level": "debug",
            "message": line,
        }, None

    # ── Regular log line ─────────────────────────────────────────────
    rank = parse_rank(line)
    level = classify_level(line)

    return {
        "timestamp": int(time.time() * 1000),
        "rank": rank,
        "level": level,
        "message": line,
    }, None
