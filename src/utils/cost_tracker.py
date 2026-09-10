"""
API cost tracking for the POSAS-MLLM study.

Estimates costs from token counts and records actual usage.
Pricing is loaded from settings.yaml — update there when rates change.
"""

from dataclasses import dataclass, field
from typing import List


@dataclass
class CallCost:
    """Cost record for a single API call."""
    model_name: str
    input_tokens: int
    output_tokens: int
    cost_usd: float


@dataclass
class CostTracker:
    """
    Accumulates costs across all API calls in a pipeline run.
    """
    calls: List[CallCost] = field(default_factory=list)

    def record(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
        pricing_input_per_1m: float,
        pricing_output_per_1m: float,
    ) -> CallCost:
        cost = (
            (input_tokens / 1_000_000) * pricing_input_per_1m
            + (output_tokens / 1_000_000) * pricing_output_per_1m
        )
        entry = CallCost(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            cost_usd=round(cost, 6),
        )
        self.calls.append(entry)
        return entry

    def total_cost(self) -> float:
        return round(sum(c.cost_usd for c in self.calls), 4)

    def total_tokens(self) -> int:
        return sum(c.input_tokens + c.output_tokens for c in self.calls)

    def cost_by_model(self) -> dict:
        breakdown = {}
        for c in self.calls:
            breakdown[c.model_name] = breakdown.get(c.model_name, 0.0) + c.cost_usd
        return {k: round(v, 4) for k, v in breakdown.items()}

    def call_count_by_model(self) -> dict:
        counts = {}
        for c in self.calls:
            counts[c.model_name] = counts.get(c.model_name, 0) + 1
        return counts

    def summary(self) -> str:
        lines = [
            "=" * 60,
            "COST SUMMARY",
            "=" * 60,
            f"Total API calls:  {len(self.calls)}",
            f"Total tokens:     {self.total_tokens():,}",
            f"Total cost:       ${self.total_cost():.4f}",
            "",
            "Per-model breakdown:",
        ]
        for model, cost in sorted(self.cost_by_model().items()):
            count = self.call_count_by_model()[model]
            lines.append(f"  {model:<16} {count:>4} calls  ${cost:.4f}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def estimate_full_run(self, num_images: int, num_models: int) -> str:
        if not self.calls:
            return "No calls recorded yet — run pilot first."

        avg_cost = self.total_cost() / len(self.calls)
        estimated_total = avg_cost * num_images * num_models

        return (
            f"Pilot: {len(self.calls)} calls, ${self.total_cost():.4f}\n"
            f"Avg cost/call: ${avg_cost:.4f}\n"
            f"Estimated full run ({num_images} images x {num_models} models): "
            f"${estimated_total:.2f}"
        )
