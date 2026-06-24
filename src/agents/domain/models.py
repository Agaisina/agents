import dataclasses
from pydantic import BaseModel, Field, field_validator, ConfigDict
from typing import Any, Dict, Optional, List, Literal


# ------------------------ #
#       AGENT CONFIG       #
# ------------------------ #

@dataclasses.dataclass
class AgentConfig:
    """Configuration for a concrete agent instance. Lives in the domain layer
    so application and infrastructure layers share the same definition."""
    model_id: str = ""
    temperature: float = 0.1
    max_tokens: int = 1024
    base_url: Optional[str] = None
    extra_params: Dict[str, Any] = dataclasses.field(default_factory=dict)


# ------------------------ #
#          AGENTS          #
# ------------------------ #

class RAGFacts(BaseModel):
    """Structured output produced by the Corporate Researcher (Compliance Auditor).

    Contains policy limits and risk flags extracted from the internal T&E knowledge
    base via RAG. All downstream agents consume this object as their single source
    of truth for policy compliance.
    """

    max_hotel_rate_eur: Optional[float] = Field(None, description="Maximum allowed nightly hotel rate in EUR. Leave null if not found.")
    max_per_diem_eur: Optional[float] = Field(None, description="Maximum allowed daily meal/per diem budget in EUR. Leave null if not found.")
    
    is_train_mandatory: bool = Field(default=False, description="True if destination is under 3.5h by train.")
    allowed_flight_class: Optional[Literal["Economy", "Premium Economy", "Business"]] = Field(None, description="Allowed cabin class based on flight duration and user tier.")
    preferred_airlines: List[str] = Field(default_factory=list, description="List of preferred airline partners for the region.")
    
    destination_risk_tier: Optional[Literal["Tier 1", "Tier 2", "Tier 3 Restricted"]] = Field(None, description="Risk level of the destination.")
    
    is_5_star_allowed: bool = Field(default=False, description="True ONLY if user is VP/C-Level OR hosting a Tier-1 client.")
    client_entertainment_allowance_eur: Optional[float] = Field(None, description="Max allowance per person if hosting a client (usually 120 EUR).")


class TransportDetails(BaseModel):
    """Single-leg transport details within a travel itinerary."""

    mode: Optional[Literal["Flight", "Train", "Car"]] = Field(None)
    carrier_name: Optional[str] = Field(None, description="Airline or Train company name.")
    price_eur: Optional[float] = Field(None)
    duration_hours: Optional[float] = Field(None)
    cabin_class: Optional[str] = Field(None)
    is_low_cost_carrier: bool = Field(default=False, description="True if Ryanair, EasyJet, WizzAir, etc.")


class AccommodationDetails(BaseModel):
    """Hotel and accommodation details for the business trip."""

    hotel_name: Optional[str] = Field(None)
    stars: Optional[int] = Field(None)
    price_per_night_eur: Optional[float] = Field(None)

    client_office_address: Optional[str] = Field(None, description="Leave NULL unless the user explicitly typed a street address in the chat.")
    distance_to_client_km: Optional[float] = Field(None, description="Leave NULL unless calculated with the geo tool.")
    commute_time_minutes: Optional[int] = Field(None, description="Leave NULL unless calculated with the geo tool.")

    business_nights: Optional[int] = Field(None)
    weekend_bleisure_nights: int = Field(default=0)


class PlannerFacts(BaseModel):
    """Full itinerary produced by the Senior Corporate Logistics Planner.

    Captures the complete logistics blueprint — transport, accommodation,
    airport transfers, and client entertainment — ready to be audited by the
    Financial Operator.
    """

    destination: Optional[str] = Field(default=None, description="The main destination city (Mandatory).")
    days_in_advance_booked: Optional[int] = Field(None, description="How many days before the trip is this being booked?")

    outward_transport: Optional[TransportDetails] = Field(default_factory=TransportDetails)
    return_transport: Optional[TransportDetails] = Field(default_factory=TransportDetails)

    accommodation: Optional[AccommodationDetails] = Field(default_factory=AccommodationDetails)

    airport_transfer_mode: Optional[Literal["Public Transit", "Taxi/Uber", "Rental Car"]] = Field(None)
    airport_transfer_estimated_cost_eur: float = Field(default=0.0)

    has_client_dinner: bool = Field(default=False)
    client_company_name: Optional[str] = Field(None, description="Leave NULL. Do NOT invent company names.")
    itinerary_summary: Optional[str] = Field(None, description="Brief step-by-step summary.")


class OperatorFacts(BaseModel):
    """Financial audit report produced by the Financial Controller (Operator).

    Compares the Planner's proposed itinerary against the RAG-extracted policy
    limits and emits a compliance verdict, total cost breakdown, and the
    approval routing decision.
    """

    total_hotel_cost_eur: Optional[float] = Field(None, description="Business nights * price per night. (Weekend nights EXCLUDED).")
    total_transport_cost_eur: Optional[float] = Field(None)
    total_estimated_per_diem_eur: Optional[float] = Field(None, description="Business days * per diem limit. (Weekends EXCLUDED).")
    grand_total_tco_eur: Optional[float] = Field(None, description="Total Cost of Ownership billed to the company.")
    
    is_fully_compliant: bool = Field(default=False, description="True ONLY if NO policy rules are broken.")
    
    approval_routing: Literal["AUTO_APPROVED", "REJECTED_MUST_REPLAN", "PENDING_MANAGER", "PENDING_VP_FINANCE", "PENDING_CEO_SECURITY", "INCOMPLETE_DATA"] = Field(
        ..., 
        description="""
        AUTO_APPROVED: 100% compliant.
        REJECTED_MUST_REPLAN: Banned items (e.g., Airbnb, 5-star without privilege).
        PENDING_MANAGER: Minor exception (< 20% overage).
        PENDING_VP_FINANCE: Major exception (> 20% overage).
        PENDING_CEO_SECURITY: Tier 3 Risk or Tier-1 Client exception.
        INCOMPLETE_DATA: If PlannerFacts is missing required pricing or logistics.
        """
    )
    
    violations_detected: List[str] = Field(default_factory=list, description="List of specific policy rules violated. Empty if fully compliant.")
    operator_notes: str = Field(..., description="Detailed explanation of the final approval or rejection.")


# ------------------------ #
#   TOOLS (STRICT INPUTS)  #
# ------------------------ #

class DistanceInput(BaseModel):
    """Input schema for the ``get_travel_distance`` tool."""

    origin: str = Field(
        ..., 
        description="The starting location. MUST be a strict street address (e.g., 'Calle de Delicias 42, Madrid, Spain'). NEVER use building or hotel names."
    )
    destination: str = Field(
        ..., 
        description="The target location. MUST be a strict street address (e.g., 'Paseo de las Delicias 20, Madrid, Spain'). NEVER use building or hotel names."
    )

class TrainRouteInput(BaseModel):
    """Input schema for train availability and duration tools."""

    origin_city: str = Field(..., description="The exact starting city. Example: 'Madrid'")
    destination_city: str = Field(..., description="The exact destination city. Example: 'Barcelona'")

class TransitTimeInput(BaseModel):
    origin: str = Field(..., description="Starting location. e.g., 'Madrid Barajas Airport'")
    destination: str = Field(..., description="Target location. e.g., 'NH Eurobuilding, Madrid'")
    arrival_time_24h: str = Field(..., description="MUST be strictly in HH:MM 24-hour format (e.g., '14:30'). Do not use AM/PM.")

class OptimalTransportInput(BaseModel):
    origin_city: str = Field(..., description="The departure city. Example: 'Valencia'")
    destination_city: str = Field(..., description="The destination city. Example: 'Madrid'")

class HotelOption(BaseModel):
    name: str = Field(..., description="Name of the hotel")
    price_eur: float = Field(..., description="Negotiated nightly rate in EUR")
    address: str = Field(..., description="Full street address of the hotel")
    zone: str = Field(..., description="Neighborhood or zone description")

class HotelList(BaseModel):
    hotels: List[HotelOption] = Field(..., description="List of available hotels in the requested city")

class HotelDirectoryInput(BaseModel):
    city: str = Field(..., description="The exact name of the destination city ONLY. Example: 'Madrid' or 'London'.")

class OptimalHotelInput(BaseModel):
    city: str = Field(..., description="The destination city. Example: 'Madrid'")
    client_address: str = Field(..., description="The exact street address. MUST be formatted cleanly for a strict GPS API. ALWAYS append the country. Example: 'Paseo de las Delicias 20, Madrid, Spain'.")
    max_price_eur: float = Field(..., description="The maximum allowed hotel price per night from RAGFacts. If unknown, pass 9999.0")

class FlightSearchInput(BaseModel):
    origin: str = Field(..., description="Departure city or airport code. Example: 'Madrid'")
    destination: str = Field(..., description="Target destination city or airport code. Example: 'London'")

class CalculatorInput(BaseModel):
    expression: str = Field(..., description="A pure mathematical expression using only numbers and operators. Example: '(300 * 3) + 1200'")

class CurrencyInput(BaseModel):
    amount: float = Field(..., description="The numeric value to convert.")
    from_currency: str = Field(..., description="3-letter currency code of origin (e.g., 'USD', 'GBP').")
    to_currency: str = Field(..., description="3-letter currency code of destination (e.g., 'EUR').")