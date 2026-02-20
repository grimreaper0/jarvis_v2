"""Unit tests for ConfidenceGate v2."""
import pytest


def test_confidence_thresholds():
    """Verify thresholds match production constants."""
    from jarvis.core.confidence import THRESHOLD_EXECUTE, THRESHOLD_DELEGATE
    assert THRESHOLD_EXECUTE == 0.90
    assert THRESHOLD_DELEGATE == 0.60


def test_confidence_weights():
    """Verify 4-layer weights sum to 1.0."""
    from jarvis.core.confidence import WEIGHT_BASE, WEIGHT_VALIDATION, WEIGHT_HISTORICAL, WEIGHT_REFLEXIVE
    total = WEIGHT_BASE + WEIGHT_VALIDATION + WEIGHT_HISTORICAL + WEIGHT_REFLEXIVE
    assert abs(total - 1.0) < 0.001


def test_base_score_known_task():
    """Known task type should get base score boost."""
    from jarvis.core.confidence import ConfidenceGate
    gate = ConfidenceGate(bot_name="test_bot", automem=None)
    score = gate._calculate_base_score(
        task_type="content_post",
        task_description="Create an Instagram carousel about AI tools",
        context={"platform": "instagram"}
    )
    assert score >= 0.7  # Known type + description + context


def test_base_score_unknown_task():
    """Unknown task type should score lower."""
    from jarvis.core.confidence import ConfidenceGate
    gate = ConfidenceGate(bot_name="test_bot", automem=None)
    score = gate._calculate_base_score(
        task_type="unknown_type_xyz",
        task_description="x",  # Too short
        context={}
    )
    assert score < 0.4


def test_reflexive_score_agreement():
    """Scores in agreement should get reflexive bonus."""
    from jarvis.core.confidence import ConfidenceGate
    gate = ConfidenceGate(bot_name="test_bot", automem=None)
    # All scores close together
    score = gate._calculate_reflexive_score(0.8, 0.82, 0.79)
    avg = (0.8 + 0.82 + 0.79) / 3
    assert score > avg  # Should be boosted


def test_reflexive_score_divergence():
    """Widely divergent scores should get reflexive penalty."""
    from jarvis.core.confidence import ConfidenceGate
    gate = ConfidenceGate(bot_name="test_bot", automem=None)
    # Scores far apart
    score = gate._calculate_reflexive_score(0.1, 0.9, 0.5)
    avg = (0.1 + 0.9 + 0.5) / 3
    assert score < avg  # Should be penalized


def test_determine_decision_execute():
    from jarvis.core.confidence import ConfidenceGate
    gate = ConfidenceGate(bot_name="test_bot", automem=None)
    assert gate._determine_decision(0.95, []) == "execute"


def test_determine_decision_delegate():
    from jarvis.core.confidence import ConfidenceGate
    gate = ConfidenceGate(bot_name="test_bot", automem=None)
    assert gate._determine_decision(0.75, []) == "delegate"


def test_determine_decision_clarify():
    from jarvis.core.confidence import ConfidenceGate
    gate = ConfidenceGate(bot_name="test_bot", automem=None)
    assert gate._determine_decision(0.45, []) == "clarify"


def test_financial_guardrail_overrides_execute():
    """Financial guardrail should cap execute → delegate."""
    from jarvis.core.confidence import ConfidenceGate
    gate = ConfidenceGate(bot_name="test_bot", automem=None)
    decision = gate._determine_decision(0.95, ["max_trade_amount"])
    assert decision == "delegate"
