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


def test_runtime_policy_rejects_unknown_state_targets_and_unrecorded_evidence():
    result = validate_proposal(
        MutationProposal(
            operation=MutationOperation.SET_MUTABLE_STATE,
            target="invented_state",
            value=True,
            evidence=["evt_untrusted"],
        ),
        allowed_mutable_keys={"mood", "topic_defensiveness"},
        evidence_event_ids={"evt_recorded"},
    )

    assert result.status == "rejected"
    assert "evidence" in result.reason.lower()


def test_runtime_policy_rejects_model_promoted_facts():
    result = validate_proposal(
        MutationProposal(
            operation=MutationOperation.SET_BELIEF,
            target="cargo_missing",
            value=True,
            evidence=["evt_recorded"],
            epistemic_type=EpistemicType.FACT,
        ),
        evidence_event_ids={"evt_recorded"},
    )

    assert result.status == "rejected"
    assert "promote" in result.reason.lower()


def test_common_birthplace_phrasings_share_topic():
    assert normalize_topic("Where were you born?") == "self.birthplace"
    assert normalize_topic("What's your hometown again?") == "self.birthplace"
    assert normalize_topic("Where are you from?") == "self.birthplace"
