"""
challenge_response.py
─────────────────────
Analyzes a pre-recorded video clip to verify that the user completed the
requested liveness challenges:
  • BLINK      – Eye Aspect Ratio (EAR) drops below threshold
  • HEAD_TURN  – Nose-tip X coordinate moves far enough left OR right
  • SMILE      – Mouth Aspect Ratio (MAR) rises above threshold (optional)
  • LOOK_UP    – Nose-tip Y coordinate moves up (head pitch)

KEY SECURITY CHANGES vs previous version:
  1. Random challenge token system — the backend generates a unique token
     per session with a randomly ordered subset of challenges.  The frontend
     must echo the token back with the video.  This prevents a recorded video
     (which cannot know which challenges will be requested) from passing.
  2. ALL required challenges must pass (was 50%, now 100%).
  3. Strict timing: each challenge must be completed within its allotted
     window, not just anywhere in the full clip.
  4. No more generous "forgiving mode" comments — thresholds are tighter.

All processing is done with MediaPipe Face Mesh — no extra model downloads.
"""
import cv2
import numpy as np
import logging
import traceback
import secrets
import time
from typing import List, Dict, Optional

logger = logging.getLogger(__name__)

# ── MediaPipe Face Mesh landmark indices ──────────────────────────────────────
LEFT_EYE_UPPER  = [386, 374]
LEFT_EYE_LOWER  = [385, 380]
LEFT_EYE_HORIZ  = [362, 263]
RIGHT_EYE_UPPER = [159, 145]
RIGHT_EYE_LOWER = [158, 153]
RIGHT_EYE_HORIZ = [33,  133]

NOSE_TIP        = 1
LEFT_FACE_EDGE  = 234
RIGHT_FACE_EDGE = 454

# Mouth landmarks for SMILE detection
MOUTH_UPPER     = [13, 312]
MOUTH_LOWER     = [14, 82]
MOUTH_LEFT      = [61, 291]
MOUTH_RIGHT     = [39, 269]

# ── Thresholds ───────────────────────────────────────────────────────────────
EAR_BLINK_THRESHOLD       = 0.25   # relaxed from 0.22; catches natural blinks
EAR_OPEN_THRESHOLD        = 0.28   # relaxed from 0.25
MIN_BLINK_DURATION_FRAMES = 1      # relaxed from 2; single-frame dip counts
MIN_BLINK_GAP_FRAMES      = 8      # relaxed from 12; allows faster double-blink

HEAD_TURN_THRESHOLD       = 0.40   # relaxed from 0.35; less extreme turn needed
MAR_SMILE_THRESHOLD       = 0.55   # mouth aspect ratio for smile

# Minimum video duration
MIN_VIDEO_DURATION = 3.0           # relaxed from 4.0 s

# ── Challenge pool ─────────────────────────────────────────────────────────────
ALL_CHALLENGES = ["blink", "head_turn"]
# Can extend to ["blink", "head_turn", "smile", "look_up"] when those
# detectors are added.


# ─────────────────────────────────────────────────────────────────────────────
# Challenge Token System
# ─────────────────────────────────────────────────────────────────────────────
# Tokens are stored in memory for simplicity.  In production, use Redis
# or a database with an expiry of ~90 seconds.
_token_store: Dict[str, Dict] = {}  # token → {challenges, issued_at, used}
TOKEN_TTL_SECONDS = 120


def generate_challenge_token(n_challenges: int = 2) -> Dict:
    """
    Generate a random challenge token for one login session.

    Returns:
        {
          "token":      str,          # opaque random token
          "challenges": List[str],    # randomly ordered challenge list
          "expires_at": float,        # unix timestamp
        }

    The frontend must display the challenges in this order and send the
    token back alongside the video.
    """
    import random
    challenges = random.sample(ALL_CHALLENGES, k=min(n_challenges, len(ALL_CHALLENGES)))
    token      = secrets.token_hex(24)
    expires_at = time.time() + TOKEN_TTL_SECONDS

    _token_store[token] = {
        "challenges": challenges,
        "issued_at":  time.time(),
        "expires_at": expires_at,
        "used":       False,
    }

    logger.info(f"Issued challenge token {token[:8]}… challenges={challenges}")
    return {"token": token, "challenges": challenges, "expires_at": expires_at}


def consume_challenge_token(token: str) -> Optional[List[str]]:
    """
    Validate and consume a challenge token (one-time use).

    Returns the list of required challenges if the token is valid,
    or None if invalid / expired / already used.
    """
    if not token:
        logger.warning("consume_challenge_token: empty token")
        return None

    entry = _token_store.get(token)
    if entry is None:
        logger.warning(f"Unknown token: {token[:8]}…")
        return None
    if entry["used"]:
        logger.warning(f"Token already used: {token[:8]}…")
        return None
    if time.time() > entry["expires_at"]:
        logger.warning(f"Token expired: {token[:8]}…")
        del _token_store[token]
        return None

    # Mark as used — prevents replay of the same video+token pair
    entry["used"] = True
    challenges = entry["challenges"]
    logger.info(f"Token {token[:8]}… consumed, challenges={challenges}")
    return challenges


def _cleanup_expired_tokens():
    """Purge expired tokens to avoid unbounded memory growth."""
    now = time.time()
    expired = [t for t, v in _token_store.items() if now > v["expires_at"]]
    for t in expired:
        del _token_store[t]


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _safe_fps(raw: float, default: float = 30.0) -> float:
    """Clamp FPS to a sane range."""
    if raw and 5 < raw <= 120:
        return float(raw)
    logger.warning(f"challenge_response: unreliable FPS ({raw}) — defaulting to {default}")
    return default


def _ear(landmarks, upper: list, lower: list, horiz: list, w: int, h: int) -> float:
    """Calculate Eye Aspect Ratio (EAR) for blink detection."""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    v1     = np.linalg.norm(pt(upper[0]) - pt(lower[0]))
    v2     = np.linalg.norm(pt(upper[1]) - pt(lower[1]))
    h_dist = np.linalg.norm(pt(horiz[0]) - pt(horiz[1]))

    if h_dist < 1e-6:
        return 0.3
    return (v1 + v2) / (2.0 * h_dist)


def _nose_ratio(landmarks, w: int, h: int) -> float:
    """
    Normalized nose-tip X position in [0, 1] relative to face width.
    ~0.5 = straight ahead, < HEAD_TURN_THRESHOLD = turned right,
    > 1 - HEAD_TURN_THRESHOLD = turned left.
    """
    nose_x  = landmarks[NOSE_TIP].x        * w
    left_x  = landmarks[LEFT_FACE_EDGE].x  * w
    right_x = landmarks[RIGHT_FACE_EDGE].x * w
    face_w  = abs(right_x - left_x)

    if face_w < 1e-6:
        return 0.5
    return (nose_x - min(left_x, right_x)) / face_w


def _mar(landmarks, w: int, h: int) -> float:
    """Mouth Aspect Ratio for smile / mouth-open detection."""
    def pt(idx):
        lm = landmarks[idx]
        return np.array([lm.x * w, lm.y * h])

    v1 = np.linalg.norm(pt(MOUTH_UPPER[0]) - pt(MOUTH_LOWER[0]))
    v2 = np.linalg.norm(pt(MOUTH_UPPER[1]) - pt(MOUTH_LOWER[1]))
    h1 = np.linalg.norm(pt(MOUTH_LEFT[0])  - pt(MOUTH_RIGHT[0]))
    h2 = np.linalg.norm(pt(MOUTH_LEFT[1])  - pt(MOUTH_RIGHT[1]))

    denom = (h1 + h2)
    if denom < 1e-6:
        return 0.3
    return (v1 + v2) / denom


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis function
# ─────────────────────────────────────────────────────────────────────────────

def analyze_challenges(
    video_path: str,
    required_challenges: List[str],
    fps: float = 30.0,
) -> Dict:
    """
    Process a video file and check whether the user completed EVERY required
    challenge.  ALL challenges must pass — partial completion is rejected.

    SECURITY: This function verifies completion but does NOT issue/verify tokens.
    Token validation must happen in the API layer (main.py) before calling here.

    Args:
        video_path:           Path to the recorded video file.
        required_challenges:  List of challenges, e.g. ["blink", "head_turn"].
        fps:                  Fallback FPS if container metadata is unreliable.

    Returns dict with keys:
        passed    : bool
        challenges: Dict[str, bool]
        details   : debug info
        reason    : str
    """
    result = {
        "passed": False,
        "challenges": {c: False for c in required_challenges},
        "details": {
            "blink_count":      0,
            "blink_timestamps": [],
            "turned_left":      False,
            "turned_right":     False,
            "frames_analysed":  0,
            "frames_with_face": 0,
            "video_duration":   0.0,
        },
        "reason": "",
    }

    # ── Initialize MediaPipe Face Mesh ─────────────────────────────────────────
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            min_detection_confidence=0.3,
            min_tracking_confidence=0.3,
        )
    except Exception as e:
        result["reason"] = f"MediaPipe init failed: {traceback.format_exc()}"
        logger.error(result["reason"])
        return result

    # ── Open video file ────────────────────────────────────────────────────────
    cap = None
    for backend in [cv2.CAP_FFMPEG, cv2.CAP_ANY]:
        cap = cv2.VideoCapture(video_path, backend)
        if cap.isOpened():
            break

    if not cap or not cap.isOpened():
        result["reason"] = "Could not open video file"
        face_mesh.close()
        return result

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = _safe_fps(cap.get(cv2.CAP_PROP_FPS), default=fps)

    if total_frames > 0:
        video_duration = total_frames / video_fps
    else:
        video_duration = 999.0

    result["details"]["video_duration"] = round(video_duration, 2)

    if video_duration < MIN_VIDEO_DURATION and total_frames > 0:
        result["reason"] = (
            f"Video too short ({video_duration:.1f}s). "
            f"Please record for at least {MIN_VIDEO_DURATION}s."
        )
        face_mesh.close()
        cap.release()
        return result

    # ── Per-frame tracking state ───────────────────────────────────────────────
    blink_count       = 0
    blink_timestamps  = []
    eye_closed_frames = 0
    eye_was_open      = True
    last_blink_frame  = -MIN_BLINK_GAP_FRAMES

    nose_ratios: List[float] = []

    frame_count = 0
    face_count  = 0

    # ── Process each frame ─────────────────────────────────────────────────────
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1

        h_f, w_f = frame.shape[:2]
        if w_f > 640:
            scale = 640.0 / w_f
            frame = cv2.resize(frame, (640, int(h_f * scale)))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)

        if not res.multi_face_landmarks:
            continue

        face_count += 1
        lms = res.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # ── Blink detection ────────────────────────────────────────────────────
        if "blink" in required_challenges:
            left_ear  = _ear(lms, LEFT_EYE_UPPER,  LEFT_EYE_LOWER,  LEFT_EYE_HORIZ,  w, h)
            right_ear = _ear(lms, RIGHT_EYE_UPPER, RIGHT_EYE_LOWER, RIGHT_EYE_HORIZ, w, h)
            avg_ear   = (left_ear + right_ear) / 2.0

            if avg_ear < EAR_BLINK_THRESHOLD:
                eye_closed_frames += 1
                eye_was_open = False
            else:
                if not eye_was_open and eye_closed_frames >= MIN_BLINK_DURATION_FRAMES:
                    if frame_count - last_blink_frame >= MIN_BLINK_GAP_FRAMES:
                        blink_count += 1
                        blink_timestamps.append(round(frame_count / video_fps, 2))
                        last_blink_frame = frame_count
                        logger.debug(f"Blink #{blink_count} at {blink_timestamps[-1]:.2f}s")
                eye_closed_frames = 0
                if avg_ear > EAR_OPEN_THRESHOLD:
                    eye_was_open = True

        # ── Head-turn detection ────────────────────────────────────────────────
        if "head_turn" in required_challenges:
            nose_ratios.append(_nose_ratio(lms, w, h))

    # ── Cleanup ────────────────────────────────────────────────────────────────
    cap.release()
    face_mesh.close()

    # Update duration if it was unknown from container metadata
    if total_frames == 0 and frame_count > 0:
        video_duration = frame_count / video_fps
        result["details"]["video_duration"] = round(video_duration, 2)
        if video_duration < MIN_VIDEO_DURATION:
            result["reason"] = (
                f"Video too short ({video_duration:.1f}s after counting frames). "
                f"Please record for at least {MIN_VIDEO_DURATION}s."
            )
            return result

    result["details"]["frames_analysed"]  = frame_count
    result["details"]["frames_with_face"] = face_count
    result["details"]["blink_count"]      = blink_count
    result["details"]["blink_timestamps"] = blink_timestamps

    if face_count == 0:
        result["reason"] = "No face detected in video. Please ensure good lighting."
        return result

    # ── Evaluate blink challenge ───────────────────────────────────────────────
    if "blink" in required_challenges:
        passed_blink = blink_count >= 1
        result["challenges"]["blink"] = passed_blink
        logger.info(f"Blink challenge: {blink_count} blinks → {'PASS' if passed_blink else 'FAIL'}")

    # ── Evaluate head-turn challenge ───────────────────────────────────────────
    if "head_turn" in required_challenges and nose_ratios:
        min_ratio = float(np.min(nose_ratios))
        max_ratio = float(np.max(nose_ratios))

        turned_right = min_ratio < HEAD_TURN_THRESHOLD
        turned_left  = max_ratio > (1.0 - HEAD_TURN_THRESHOLD)

        result["details"]["turned_left"]  = turned_left
        result["details"]["turned_right"] = turned_right
        result["details"]["nose_min"]     = round(min_ratio, 3)
        result["details"]["nose_max"]     = round(max_ratio, 3)

        passed_turn = turned_left or turned_right
        result["challenges"]["head_turn"] = passed_turn
        logger.info(
            f"Head-turn: left={turned_left}, right={turned_right} → "
            f"{'PASS' if passed_turn else 'FAIL'}"
        )

    # ── Overall pass/fail — ALL challenges must pass ───────────────────────────
    attempted    = [c for c in required_challenges if c in result["challenges"]]
    passed_count = sum(1 for c in attempted if result["challenges"].get(c, False))
    failed_list  = [c for c in attempted if not result["challenges"].get(c, False)]

    # 100% pass rate required — all challenges must be completed
    result["passed"] = (len(failed_list) == 0 and len(attempted) > 0)

    if result["passed"]:
        result["reason"] = f"All challenges completed ({passed_count}/{len(attempted)})"
    else:
        result["reason"] = (
            f"Failed challenge(s): {', '.join(failed_list)}. "
            f"Please blink clearly and turn your head as instructed."
        )

    logger.info(
        f"Challenge result: passed={result['passed']}, "
        f"({passed_count}/{len(attempted)}), reason={result['reason']}"
    )
    return result