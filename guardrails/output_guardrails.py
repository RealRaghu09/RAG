DEFAULT_CONFIDENCE_THRESHOLD = 0.7


def enforce_confidence_threshold(confidence: float, threshold: float = DEFAULT_CONFIDENCE_THRESHOLD) -> None:
    if confidence < threshold:
        raise ValueError(f"Low confidence: {confidence} < {threshold}")


def enforce_refusal_policy(answer: str) -> None:
    """Reject empty or whitespace-only model answers."""
    if not answer or not answer.strip():
        raise ValueError("No valid answer.")
