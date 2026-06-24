import json
from unittest.mock import MagicMock, patch

import pytest
import requests

from src.agents.domain.models import HotelList
from src.agents.infrastructure.tools.geo_tools import (
    create_hotel_directory_tool,
    get_transit_vs_taxi_time,
    get_travel_distance
)

#---------------------------------#
#       get_travel_distance       #
#---------------------------------#

def test_get_travel_distance_should_return_route_summary():
    with patch(
        "src.agents.infrastructure.tools.geo_tools._geocode_location",
        side_effect=[(40.4168, -3.7038), (41.3851, 2.1734)],
    ), patch("src.agents.infrastructure.tools.geo_tools.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {
            "code": "Ok",
            "routes": [{"distance": 12345.0, "duration": 1800.0}],
        }
        mock_get.return_value = mock_response

        result_str = get_travel_distance.func("Madrid", "Barcelona")
        result = json.loads(result_str)

    assert result["status"] == "success", "The status should be 'success'"
    assert result["distance_km"] == 12.35, "The distance should be exactly 12.35"
    assert result["duration_minutes"] == 30, "The duration should be 30 minutes"
    assert result["policy_note"] == "Use this distance_km and duration_minutes for your PlannerFacts output.", "The policy note does not match"

    assert isinstance(result["status"], str), "The status must be a string"
    assert isinstance(result["distance_km"], float), "The distance must be a float"
    assert isinstance(result["duration_minutes"], int), "The duration must be an integer"
    assert "policy_note" in result, "The result must contain the 'policy_note' key"


def test_get_travel_distance_should_return_geocoding_error_message():
    with patch(
        "src.agents.infrastructure.tools.geo_tools._geocode_location",
        side_effect=ValueError("The location 'Unknown Place' could not be found."),
    ):
        result = get_travel_distance.func("Unknown Place", "Barcelona")

    assert result == (
        "Geocoding Error: The location 'Unknown Place' could not be found. "
        "Please provide a more specific location name and try again."
    )


def test_get_travel_distance_should_return_service_error_when_no_route_found():
    with patch(
        "src.agents.infrastructure.tools.geo_tools._geocode_location",
        return_value=(40.4168, -3.7038),
    ), patch("src.agents.infrastructure.tools.geo_tools.requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.raise_for_status.return_value = None
        mock_response.json.return_value = {"code": "NoRoute", "routes": []}
        mock_get.return_value = mock_response

        result = get_travel_distance.func("Madrid", "Unknown Route")

    assert "Error: The routing service could not find a valid driving path" in result


def test_get_travel_distance_should_return_timeout_message():
    with patch(
        "src.agents.infrastructure.tools.geo_tools._geocode_location",
        return_value=(40.4168, -3.7038),
    ), patch(
        "src.agents.infrastructure.tools.geo_tools.requests.get",
        side_effect=requests.exceptions.Timeout,
    ):
        result = get_travel_distance.func("Madrid", "Barcelona")

    assert result == (
        "Error: The geocoding or routing API timed out. Please try invoking the tool again."
    )

#-----------------------------------------#
#         get_transit_vs_taxi_time        #
#-----------------------------------------#

def test_taxi_authorized_late_night_arrival():
    result_str = get_transit_vs_taxi_time.func("Heathrow Airport", "Novotel London", "22:30")
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert result["taxi_authorized"] is True
    assert "Arrival is between 9:00 PM and 6:00 AM" in result["policy_note"]


def test_taxi_authorized_early_morning_arrival():
    result_str = get_transit_vs_taxi_time.func("CDG Airport", "Paris Center", "04:15")
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert result["taxi_authorized"] is True
    assert "Arrival is between 9:00 PM and 6:00 AM" in result["policy_note"]


def test_taxi_prohibited_during_normal_hours_with_short_transit():
    result_str = get_transit_vs_taxi_time.func("JFK Airport", "Times Square", "14:00")
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert result["taxi_authorized"] is False
    assert result["transit_time_minutes"] == 45
    assert "PROHIBITED" in result["policy_note"]


def test_time_format_error_handling():
    result_str = get_transit_vs_taxi_time.func("Airport", "Hotel", "2 PM")
    result = json.loads(result_str)

    assert result["status"] == "error"
    assert "Format Error" in result["message"]
    assert "MUST be formatted as HH:MM" in result["message"]


def test_time_format_out_of_bounds_error_handling():
    result_str = get_transit_vs_taxi_time.func("Airport", "Hotel", "   ")
    result = json.loads(result_str)

    assert result["status"] == "error"
    assert "Format Error" in result["message"]


#-----------------------------------------#
#             hotel_directory             #
#-----------------------------------------#


def test_hotel_directory_success():
    mock_doc = MagicMock()
    mock_doc.page_content = "Corporate rates available for NH Eurobuilding and Riu Plaza."
    
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [mock_doc] # AHORA ES .invoke()

    mock_extractor = MagicMock()
    mock_structured_llm = MagicMock()
    mock_extractor.with_structured_output.return_value = mock_structured_llm 

    mock_extracted_data = MagicMock()
    mock_hotel = MagicMock()
    mock_hotel.model_dump.return_value = {"name": "NH Eurobuilding", "price_eur": 120.0}
    mock_extracted_data.hotels = [mock_hotel] 

    mock_structured_llm.invoke.return_value = mock_extracted_data

    hotel_tool = create_hotel_directory_tool(mock_retriever, mock_extractor)
    result_str = hotel_tool.func(" madrid ")
    result = json.loads(result_str)

    assert result["status"] == "success"
    assert len(result["hotels"]) == 1
    assert result["hotels"][0]["name"] == "NH Eurobuilding"

    mock_retriever.invoke.assert_called_once()
    llamada_arg = mock_retriever.invoke.call_args[0][0]
    assert "Destination: Madrid" in llamada_arg


def test_hotel_directory_no_docs_found():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = []
    
    mock_extractor = MagicMock()

    hotel_tool = create_hotel_directory_tool(mock_retriever, mock_extractor)
    result_str = hotel_tool.func("Barcelona")
    result = json.loads(result_str)

    assert result["status"] == "not_found"
    assert result["hotels"] == []
    assert "No corporate partner hotels found" in result["policy_note"]
    
    mock_extractor.with_structured_output.assert_not_called()


def test_hotel_directory_docs_found_but_extractor_returns_empty():
    mock_retriever = MagicMock()
    mock_retriever.invoke.return_value = [MagicMock(page_content="Some irrelevant text")]
    
    mock_extractor = MagicMock()
    mock_structured_llm = MagicMock()
    mock_extractor.with_structured_output.return_value = mock_structured_llm
    
    mock_extracted_data = MagicMock()
    mock_extracted_data.hotels = []
    mock_structured_llm.invoke.return_value = mock_extracted_data

    hotel_tool = create_hotel_directory_tool(mock_retriever, mock_extractor)
    result_str = hotel_tool.func("Paris")
    result = json.loads(result_str)

    assert result["status"] == "not_found"
    assert result["hotels"] == []
    assert "Vector DB retrieved context, but no specific hotels matched" in result["policy_note"]


def test_hotel_directory_handles_exceptions():
    mock_retriever = MagicMock()
    mock_retriever.invoke.side_effect = Exception("Conexión perdida a la VectorDB")
    
    mock_extractor = MagicMock()

    hotel_tool = create_hotel_directory_tool(mock_retriever, mock_extractor)
    result_str = hotel_tool.func("London")
    result = json.loads(result_str)

    assert result["status"] == "error"
    assert result["hotels"] == []
    assert "System error retrieving hotel directory" in result["message"]


#-----------------------------------------#
#         find_optimal_transport          #
#-----------------------------------------#

from src.agents.infrastructure.tools.geo_tools import find_optimal_transport


def test_find_optimal_transport_train_route_blocks_flights():
    """Known train route < 3.5h → train selected, flights prohibited."""
    result = json.loads(find_optimal_transport.func("Madrid", "Barcelona"))

    assert result["status"] == "success"
    assert result["selected_transport"]["mode"] == "Train"
    assert "Flights are strictly prohibited" in result["decision_rationale"]


def test_find_optimal_transport_train_route_is_case_insensitive():
    """Input city names with mixed case still match the train route table."""
    result = json.loads(find_optimal_transport.func("VALENCIA", "madrid"))

    assert result["status"] == "success"
    assert result["selected_transport"]["mode"] == "Train"


def test_find_optimal_transport_train_route_with_whitespace():
    """Leading/trailing whitespace in city names is stripped before lookup."""
    result = json.loads(find_optimal_transport.func("  Paris  ", "  Lyon  "))

    assert result["status"] == "success"
    assert result["selected_transport"]["mode"] == "Train"


def test_find_optimal_transport_no_train_route_falls_back_to_flight():
    """Unknown/long route → flight logic is evaluated."""
    result = json.loads(find_optimal_transport.func("Madrid", "London"))

    assert result["status"] == "success"
    assert result["selected_transport"]["mode"] == "Flight"


def test_find_optimal_transport_flight_preferred_within_tolerance():
    """When preferred airline is within 15% of cheapest, it should be selected."""
    # Preferred = 130 EUR, Low-cost = 85 EUR, threshold = 85 * 1.15 = 97.75
    # 130 > 97.75, so low-cost wins — validate the rationale reflects this
    result = json.loads(find_optimal_transport.func("Madrid", "London"))

    transport = result["selected_transport"]
    rationale = result["decision_rationale"]

    assert transport["mode"] == "Flight"
    # The mocked preferred flight (Iberia, 130€) exceeds 15% threshold over
    # the low-cost option (Ryanair, 85€), so Ryanair must be selected.
    assert transport["provider"] == "Ryanair"
    assert "15% tolerance" in rationale


def test_find_optimal_transport_train_duration_returned():
    """Train result includes the route duration in hours."""
    result = json.loads(find_optimal_transport.func("London", "Manchester"))

    assert result["selected_transport"]["duration_hours"] == 2.2


def test_find_optimal_transport_reverse_route_not_symmetric():
    """A route only defined in one direction is not found in reverse."""
    forward = json.loads(find_optimal_transport.func("Paris", "Lyon"))
    backward = json.loads(find_optimal_transport.func("Lyon", "Paris"))

    # Forward: train exists
    assert forward["selected_transport"]["mode"] == "Train"
    # Backward: no train entry → falls back to flight
    assert backward["selected_transport"]["mode"] == "Flight"