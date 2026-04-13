from __future__ import annotations

from collections import defaultdict

from .models import CategoryConfig, CategoryResult, EngineConfig, Evidence


METHOD_ORDER = ["lexical", "fuzzy", "linguistic", "semantic", "presidio"]


def score_category(
    category: CategoryConfig,
    evidences: list[Evidence],
    config: EngineConfig,
) -> CategoryResult:
    default_weights = config.scoring.get("method_weights", {})
    category_weights = {**default_weights, **category.weights}
    method_scores: dict[str, float] = defaultdict(float)

    for evidence in evidences:
        method_scores[evidence.method] = max(method_scores[evidence.method], evidence.score)

    weighted_total = 0.0
    active_weight_total = 0.0
    for method_name, score in method_scores.items():
        weight = float(category_weights.get(method_name, 0.0))
        weighted_total += score * weight
        active_weight_total += weight

    normalized_weighted_score = (
        weighted_total / active_weight_total if active_weight_total else 0.0
    )
    evidence_bonus = min(0.15, max(0, len(evidences) - 1) * 0.03)
    final_score = round(min(1.0, normalized_weighted_score + evidence_bonus), 4)

    review_threshold = float(
        category.thresholds.get("review", config.scoring.get("review_threshold", 0.45))
    )
    alert_threshold = float(
        category.thresholds.get("alert", config.scoring.get("alert_threshold", 0.75))
    )

    if final_score >= alert_threshold:
        decision = "alert"
    elif final_score >= review_threshold:
        decision = "review"
    else:
        decision = "clear"

    sorted_evidences = sorted(
        evidences,
        key=lambda item: (
            item.score,
            METHOD_ORDER.index(item.method) if item.method in METHOD_ORDER else 99,
        ),
        reverse=True,
    )
    top_evidences = sorted_evidences[:3]
    explanation_parts = [
        f"{evidence.method} -> '{evidence.trigger}' ({evidence.score:.2f})"
        for evidence in top_evidences
    ]
    explanation = "; ".join(explanation_parts) or "Aucune evidence significative."

    return CategoryResult(
        category_key=category.key,
        category_label=category.label,
        score=final_score,
        decision=decision,
        evidences=sorted_evidences,
        method_scores={key: round(value, 4) for key, value in method_scores.items()},
        explanation=explanation,
    )
