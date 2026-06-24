import pytest
from pydantic import ValidationError
from src.agents.domain.models import (
    AgentConfig,
    RAGFacts,
    TransportDetails,
    AccommodationDetails,
    PlannerFacts,
    OperatorFacts,
)


# ─────────────────────────────────────────
#   AgentConfig
# ─────────────────────────────────────────

def test_agent_config_defaults():
    cfg = AgentConfig()
    assert cfg.model_id == ""
    assert cfg.temperature == 0.1
    assert cfg.max_tokens == 1024
    assert cfg.base_url is None
    assert cfg.extra_params == {}


def test_agent_config_custom_values():
    cfg = AgentConfig(model_id="gpt-4o", temperature=0.5, max_tokens=2048, base_url="http://localhost:11434")
    assert cfg.model_id == "gpt-4o"
    assert cfg.temperature == 0.5
    assert cfg.base_url == "http://localhost:11434"


# ─────────────────────────────────────────
#   RAGFacts
# ─────────────────────────────────────────

def test_rag_facts_defaults():
    facts = RAGFacts()
    assert facts.max_hotel_rate_eur is None
    assert facts.max_per_diem_eur is None
    assert facts.is_train_mandatory is False
    assert facts.allowed_flight_class is None
    assert facts.preferred_airlines == []
    assert facts.destination_risk_tier is None
    assert facts.is_5_star_allowed is False
    assert facts.client_entertainment_allowance_eur is None


def test_rag_facts_valid_flight_class():
    for cls in ("Economy", "Premium Economy", "Business"):
        facts = RAGFacts(allowed_flight_class=cls)
        assert facts.allowed_flight_class == cls


def test_rag_facts_invalid_flight_class_raises():
    with pytest.raises(ValidationError):
        RAGFacts(allowed_flight_class="First Class")


def test_rag_facts_valid_risk_tiers():
    for tier in ("Tier 1", "Tier 2", "Tier 3 Restricted"):
        facts = RAGFacts(destination_risk_tier=tier)
        assert facts.destination_risk_tier == tier


def test_rag_facts_invalid_risk_tier_raises():
    with pytest.raises(ValidationError):
        RAGFacts(destination_risk_tier="Tier 4")


# ─────────────────────────────────────────
#   TransportDetails
# ─────────────────────────────────────────

def test_transport_details_defaults():
    t = TransportDetails()
    assert t.mode is None
    assert t.carrier_name is None
    assert t.price_eur is None
    assert t.is_low_cost_carrier is False


def test_transport_details_valid_mode():
    for mode in ("Flight", "Train", "Car"):
        t = TransportDetails(mode=mode)
        assert t.mode == mode


def test_transport_details_invalid_mode_raises():
    with pytest.raises(ValidationError):
        TransportDetails(mode="Bus")


# ─────────────────────────────────────────
#   AccommodationDetails
# ─────────────────────────────────────────

def test_accommodation_details_defaults():
    a = AccommodationDetails()
    assert a.hotel_name is None
    assert a.weekend_bleisure_nights == 0
    assert a.distance_to_client_km is None


def test_accommodation_details_valid_data():
    a = AccommodationDetails(hotel_name="NH Madrid", stars=4, price_per_night_eur=140.0, business_nights=2)
    assert a.hotel_name == "NH Madrid"
    assert a.stars == 4
    assert a.price_per_night_eur == 140.0
    assert a.business_nights == 2


# ─────────────────────────────────────────
#   PlannerFacts
# ─────────────────────────────────────────

def test_planner_facts_defaults():
    p = PlannerFacts()
    assert p.destination is None
    assert p.has_client_dinner is False
    assert p.airport_transfer_estimated_cost_eur == 0.0
    assert p.airport_transfer_mode is None


def test_planner_facts_valid_airport_transfer_modes():
    for mode in ("Public Transit", "Taxi/Uber", "Rental Car"):
        p = PlannerFacts(airport_transfer_mode=mode)
        assert p.airport_transfer_mode == mode


def test_planner_facts_invalid_airport_transfer_raises():
    with pytest.raises(ValidationError):
        PlannerFacts(airport_transfer_mode="Helicopter")


# ─────────────────────────────────────────
#   OperatorFacts
# ─────────────────────────────────────────

def test_operator_facts_auto_approved():
    op = OperatorFacts(
        is_fully_compliant=True,
        approval_routing="AUTO_APPROVED",
        operator_notes="All good.",
    )
    assert op.is_fully_compliant is True
    assert op.approval_routing == "AUTO_APPROVED"
    assert op.violations_detected == []


def test_operator_facts_pending_manager():
    op = OperatorFacts(
        approval_routing="PENDING_MANAGER",
        operator_notes="Minor hotel rate overage.",
        violations_detected=["Hotel rate 10% over limit"],
    )
    assert op.approval_routing == "PENDING_MANAGER"
    assert len(op.violations_detected) == 1


def test_operator_facts_invalid_approval_routing_raises():
    with pytest.raises(ValidationError):
        OperatorFacts(approval_routing="UNKNOWN_STATUS", operator_notes="test")


def test_operator_facts_all_valid_routing_values():
    valid_routings = [
        "AUTO_APPROVED",
        "REJECTED_MUST_REPLAN",
        "PENDING_MANAGER",
        "PENDING_VP_FINANCE",
        "PENDING_CEO_SECURITY",
        "INCOMPLETE_DATA",
    ]
    for routing in valid_routings:
        op = OperatorFacts(approval_routing=routing, operator_notes="ok")
        assert op.approval_routing == routing


def test_operator_facts_missing_operator_notes_raises():
    with pytest.raises(ValidationError):
        OperatorFacts(approval_routing="AUTO_APPROVED")
