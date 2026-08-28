from services.common import EpistemicType, MutationOperation, MutationProposal
from services.memory.app import validate_proposal
from services.orchestrator.app import normalize_topic


def test_core_mutation_is_rejected():
    result = validate_proposal(
        MutationProposal(
            operation=MutationOperation.UPDATE_CORE,
            target="identity.birthplace",
            value="Southport",
            evidence=["evt_1"],
            epistemic_type=EpistemicType.FACT,
        )
    )
    assert result.status == "rejected"


def test_belief_revision_requires_evidence():
    result = validate_proposal(
        MutationProposal(
            operation=MutationOperation.SET_BELIEF,
            target="trust.mara",
            value=0.2,
            evidence=[],
            epistemic_type=EpistemicType.BELIEF,
        )
    )
    assert result.status == "rejected"


def test_belief_revision_is_versioned_with_evidence():
    result = validate_proposal(
        MutationProposal(
            operation=MutationOperation.SET_BELIEF,
            target="trust.mara",
            value=0.2,
            evidence=["evt_1"],
            epistemic_type=EpistemicType.BELIEF,
        )
    )
    assert result.status == "versioned"


def test_common_birthplace_phrasings_share_topic():
    assert normalize_topic("Where were you born?") == "self.birthplace"
    assert normalize_topic("What's your hometown again?") == "self.birthplace"
    assert normalize_topic("Where are you from?") == "self.birthplace"
