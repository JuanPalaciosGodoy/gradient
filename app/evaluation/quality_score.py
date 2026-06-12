from dataclasses import dataclass, field


@dataclass
class QualityScore:
    score: float        # 0.0–1.0; higher is better match to original
    method: str         # heuristic | prometheus | llm_judge
    details: dict = field(default_factory=dict)
