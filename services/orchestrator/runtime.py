"""Deployment composition for the orchestrator during route decomposition."""

from services.orchestrator import app as orchestrator
from services.orchestrator.history_adapter import (
    evidence_history_review,
    neutral_interaction_classification,
    no_repeat_lobe_reuse,
)

# Transitional wiring only: the deployed route remains singular while historical
# matching is moved out of the monolithic route module. These assignments disappear
# when ``chat`` is extracted into a turn service that depends on the matcher directly.
orchestrator.InteractionClassification = neutral_interaction_classification
orchestrator.executive_repeat_review = evidence_history_review
orchestrator.immediate_repeat_lobe_reuse = no_repeat_lobe_reuse

app = orchestrator.app
