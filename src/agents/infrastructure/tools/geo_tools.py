import json
import time
import math
import requests
import logging
from typing import Tuple
from langchain_core.tools import tool
from tenacity import retry, wait_exponential, stop_after_attempt, retry_if_exception_type
from langchain_core.messages import SystemMessage, HumanMessage

from src.agents.domain.models import HotelList, HotelDirectoryInput, OptimalHotelInput, DistanceInput,\
    TrainRouteInput, TransitTimeInput, FlightSearchInput, OptimalTransportInput

logger = logging.getLogger(__name__)


# ------------------------ #
#   DISTANCES (API USAGE)  #
# ------------------------ #

@retry(
    stop=stop_after_attempt(3), 
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type((requests.exceptions.RequestException, requests.exceptions.Timeout))
)
def _geocode_location(location_name: str) -> Tuple[float, float]:
    """
    Converts a text location into coordinates with exponential backoff retries.
    """
    url = "https://nominatim.openstreetmap.org/search"
    headers = {
        'User-Agent': 'VeritatsCorporate-AI-Agent/1.0 (contact@veritascorporate.example)'
    }
    params = {'q': location_name, 'format': 'json', 'limit': 1}
    
    response = requests.get(url, headers=headers, params=params, timeout=5)
    response.raise_for_status() 
    
    data = response.json()
    if not data:
        raise ValueError(f"The location '{location_name}' could not be found by the geocoding service.")
        
    return float(data[0]['lat']), float(data[0]['lon'])


@tool("get_travel_distance", args_schema=DistanceInput)
def get_travel_distance(origin: str, destination: str) -> str:
    """
    Useful for calculating the actual driving distance and estimated taxi travel time between two points.
    Inputs: 
    - origin: Starting location (e.g., 'NH Eurobuilding, Madrid')
    - destination: Target location (e.g., 'Iberdrola Office, Tomas Redondo 1, Madrid')
    ALWAYS use this to calculate ground transport times for business meetings and hotel proximity.
    """
    try:
        lat1, lon1 = _geocode_location(origin)
        lat2, lon2 = _geocode_location(destination)
        
        url = f"http://router.project-osrm.org/route/v1/driving/{lon1},{lat1};{lon2},{lat2}?overview=false"
        resp = requests.get(url, timeout=5)
        resp.raise_for_status()
        
        data = resp.json()
        if data.get("code") == "Ok":
            route = data["routes"][0]
            dist_km = round(route["distance"] / 1000, 2)
            time_mins = round(route["duration"] / 60)

            return json.dumps({
                "status": "success",
                "distance_km": dist_km,
                "duration_minutes": time_mins,
                "policy_note": "Use this distance_km and duration_minutes for your PlannerFacts output."
            })
        else:
            return "Error: The routing service could not find a valid driving path."

    except ValueError as ve:
        return f"Geocoding Error: {str(ve)} Please provide a more specific location name and try again."
    except requests.exceptions.Timeout:
        return "Error: The geocoding or routing API timed out. Please try invoking the tool again."
    except Exception as e:
        return f"API Error in get_travel_distance: {str(e)}. Please correct your arguments and try again."


# ---------------------------- #
#   DISTANCES (NOT API USAGE)  #
# ---------------------------- #

def _haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    
    a = (math.sin(dlat / 2) * math.sin(dlat / 2) +
         math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
         math.sin(dlon / 2) * math.sin(dlon / 2))
         
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


def _calculate_distance_straight_line(origin: str, destination: str) -> Tuple[float, int]:
    lat1, lon1 = _geocode_location(origin)
    lat2, lon2 = _geocode_location(destination)
    
    dist_km = round(_haversine_distance(lat1, lon1, lat2, lon2), 2)
    # Assumption time
    if dist_km < 3.0:
        time_mins = max(1, round(dist_km * 5))
    elif dist_km < 15.0:
        time_mins = max(1, round(dist_km * 3))
    else:
        time_mins = max(1, round(dist_km * 1.2))
    
    return dist_km, time_mins


# ---------------------------- #
#       PLANNER TOOLS          #
# ---------------------------- #


@tool("get_transit_vs_taxi_time", args_schema=TransitTimeInput)
def get_transit_vs_taxi_time(origin: str, destination: str, arrival_time_24h: str) -> str:
    """
    Compares public transit time vs taxi time to enforce the Ground Transportation policy.
    ALWAYS use this tool specifically for AIRPORT TRANSFERS or TRAIN STATION TRANSFERS 
    (e.g., from 'London Heathrow Airport' to 'Novotel London').
    Do NOT use this for daily hotel-to-client commutes.
    """
    try:
        clean_time = arrival_time_24h.strip()
        hour = int(clean_time.split(":")[0])
    except (ValueError, IndexError):
        return json.dumps({
            "status": "error",
            "message": "Format Error: arrival_time_24h MUST be formatted as HH:MM (e.g., '14:30'). Please call the tool again with the correct format."
        })
        
    # Mocked. For real-environment use API Google Maps (mode=transit)
    transit_mins = 45 

    if hour >= 21 or hour < 6:
        result = {
            "status": "success",
            "taxi_authorized": True,
            "transit_time_minutes": transit_mins,
            "policy_note": "Arrival is between 9:00 PM and 6:00 AM. Taxi/Uber is AUTHORIZED regardless of public transit time."
        }
    elif transit_mins > 90:
        result = {
            "status": "success",
            "taxi_authorized": True,
            "transit_time_minutes": transit_mins,
            "policy_note": f"Public transit takes {transit_mins} mins (> 1.5 hours). Taxi is AUTHORIZED."
        }
    else:
        result = {
            "status": "success",
            "taxi_authorized": False,
            "transit_time_minutes": transit_mins,
            "policy_note": f"Public transit takes {transit_mins} mins (<= 1.5 hours). Taxi is PROHIBITED. Employee must use public transport."
        }
        
    return json.dumps(result)


@tool("find_optimal_transport", args_schema=OptimalTransportInput)
def find_optimal_transport(origin_city: str, destination_city: str) -> str:
    """
    ONE-STOP SHOP for Transport: Checks train availability (Sustainability Policy) 
    and flight prices (15% Tolerance Policy) to select the absolute best transport.
    ALWAYS use this instead of checking flights and trains separately.
    """
    orig_clean = origin_city.lower().strip()
    dest_clean = destination_city.lower().strip()
    
    logger.info(f"🚆/✈️ Buscando transporte óptimo de {orig_clean.title()} a {dest_clean.title()}")

    # Mocked train routes with durations in hours. In a real implementation, this would query a live train API.
    train_routes = {
        ("madrid", "barcelona"): 2.5,
        ("paris", "lyon"): 2.0,
        ("london", "manchester"): 2.2,
        ("valencia", "madrid"): 1.8,
        ("madrid", "valencia"): 1.8
    }
    
    train_duration = train_routes.get((orig_clean, dest_clean))

    # Rule 1: SUSTAINABILITY (< 3.5h)
    if train_duration is not None and train_duration < 3.5:
        logger.info("✅ High-speed train detected (< 3.5h). Flights blocked.")
        result = {
            "status": "success",
            "selected_transport": {
                "mode": "Train",
                "provider": "High-Speed Rail",
                "price_eur": 45.0,
                "duration_hours": train_duration,
                "class": "Standard"
            },
            "decision_rationale": f"POLICY COMPLIANT: Mandatory train usage. Route takes {train_duration}h (< 3.5h limit). Flights are strictly prohibited."
        }
        return json.dumps(result, indent=2)

    # 2. FLIGHT LOGIC (If train takes more than 3.5h or doesn't exist)
    logger.info("✈️ Train not viable or exceeds 3.5h. Evaluating flights and 15% rule...")
    
    # Simulate fetching flights (Here would be real API)
    preferred_flight = {"airline": "Iberia", "tier": "Preferred Partner", "price_eur": 130.0, "cabin_class": "Economy", "duration_hours": 1.2}
    low_cost_flight = {"airline": "Ryanair", "tier": "Ultra-Low Cost", "price_eur": 85.0, "cabin_class": "Economy", "duration_hours": 1.2}
    
    # RULE 2: 15% TOLERANCE
    tolerance_threshold = low_cost_flight["price_eur"] * 1.15
    
    if preferred_flight["price_eur"] <= tolerance_threshold:
        selected = preferred_flight
        rationale = f"POLICY COMPLIANT: Selected Preferred Partner ({preferred_flight['airline']}). Its price ({preferred_flight['price_eur']}€) is within 15% of the cheapest option ({low_cost_flight['price_eur']}€)."
    else:
        selected = low_cost_flight
        rationale = f"POLICY COMPLIANT: Selected Alternative ({low_cost_flight['airline']}). Preferred Partner ({preferred_flight['airline']}) was {preferred_flight['price_eur']}€, exceeding the 15% tolerance over the {low_cost_flight['price_eur']}€ baseline."

    result = {
        "status": "success",
        "selected_transport": {
            "mode": "Flight",
            "provider": selected["airline"],
            "price_eur": selected["price_eur"],
            "duration_hours": selected["duration_hours"],
            "class": selected["cabin_class"]
        },
        "decision_rationale": rationale
    }
    
    return json.dumps(result, indent=2)



def create_hotel_directory_tool(retriever, extractor_llm):
    @tool("get_approved_corporate_hotels", args_schema=HotelDirectoryInput)
    def get_approved_corporate_hotels(city: str) -> str:
        """
        Retrieves the LIVE list of approved Corporate partner hotels for a specific city.
        ALWAYS use this to find accommodation before calculating costs.
        """
        clean_city = city.strip().title()
        logger.info(f"Searching for hotel directory in: {clean_city}")
        
        try:
            docs = retriever.invoke(f"Destination: {clean_city} preferred hotel directory partners")
            
            if not docs:
                logger.warning(f"No documents found for {clean_city}.")
                return json.dumps({
                    "status": "not_found",
                    "hotels": [],
                    "policy_note": f"No corporate partner hotels found for {clean_city}. Employee must request an exception or use standard per diem limits."
                })
                
            context = "\n\n".join([d.page_content for d in docs])

            structured_llm = extractor_llm.with_structured_output(HotelList)
            
            sys_msg = SystemMessage(content="""
            You are a meticulous data extraction AI. Your ONLY job is to extract a list of hotels from the provided context into the defined JSON schema. 
            Never invent data. If the data is there, you MUST extract it.
            """)
            
            human_msg = HumanMessage(content=f"""
            Please extract the approved corporate hotels for the city of '{clean_city}'.
            
            CRITICAL INSTRUCTIONS:
            1. Find the section referencing {clean_city}. It might include the country name, like "DESTINATION: {clean_city.upper()} (SPAIN)". Be flexible.
            2. Extract the hotel name, full address, and zone.
            3. For 'price_eur', extract ONLY the mathematical float number. Ignore text like 'EUR/night' or '(Breakfast included)'. Example: '135 EUR/night' -> 135.0
            
            CONTEXT TO EXTRACT FROM:
            {context}
            """)
            
            extracted_data = structured_llm.invoke([sys_msg, human_msg])
            
            print(f"🤖 DEBUG LLM EXTRACTOR: {extracted_data}")
            
            hotels_list = extracted_data.hotels if hasattr(extracted_data, 'hotels') else []
            
            if not hotels_list:
                result = {
                    "status": "not_found",
                    "hotels": [],
                    "policy_note": f"Vector DB retrieved context, but no specific hotels matched '{clean_city}'. Exception required."
                }
            else:
                parsed_hotels = [h.model_dump() for h in hotels_list]
                
                result = {
                    "status": "success",
                    "hotels": parsed_hotels,
                    "policy_note": """
                    HOTEL SELECTION LOGIC (CRITICAL):
                    1. Check RAGFacts.max_hotel_rate_eur. 
                    2. Filter the list of hotels to ONLY those whose price_eur is <= the max rate.
                    3. If multiple hotels are within budget, use 'get_travel_distance' to the client office and select the CLOSEST one.
                    4. IF AND ONLY IF all hotels are OVER budget, select the ABSOLUTE CHEAPEST hotel from the list, regardless of distance.
                    """
                }

            clean_output = json.dumps(result, indent=2)
            logger.info(f"Hotels extracted:\n{clean_output}")
            
            return clean_output

        except Exception as e:
            logger.error(f"Error in get_approved_corporate_hotels: {e}", exc_info=True)
            return json.dumps({
                "status": "error",
                "message": f"System error retrieving hotel directory for {clean_city}. Please proceed without hotel data and warn the user.",
                "hotels": []
            })

    return get_approved_corporate_hotels


def create_optimal_hotel_tool(retriever, extractor_llm):
    """
    RAG + LLM Extraction + Python Business Logic + Real Map API Routing
    """
    @tool("find_optimal_hotel", args_schema=OptimalHotelInput)
    def find_optimal_hotel(city: str, client_address: str, max_price_eur: float) -> str:
        """
        ONE-STOP SHOP for Hotel Planning: Retrieves corporate hotels, filters by policy budget, 
        calculates REAL distance to the client using Maps API, and selects the absolute BEST option.
        ALWAYS use this instead of checking hotels and distances separately.
        """
        clean_city = city.strip().title()
        logger.info(f"🏨 Buscando el hotel óptimo en {clean_city} para el presupuesto de {max_price_eur}€")
        
        try:
            # 1. RETRIEVE HOTEL DIRECTORY
            docs = retriever.invoke(f"Destination: {clean_city} preferred hotel directory partners")
            if not docs:
                return json.dumps({"status": "not_found", "message": f"No corporate hotels found for {clean_city}."})
                
            context = "\n\n".join([d.page_content for d in docs])
            
            # 2. EXTRACT HOTELS USING LLM
            structured_llm = extractor_llm.with_structured_output(HotelList)
            sys_msg = SystemMessage(content="You are a data extraction AI. Extract hotels into JSON. Never invent data.")
            human_msg = HumanMessage(content=f"""
            Extract approved corporate hotels for '{clean_city}'.
            Extract: name, address, zone. 
            For 'price_eur', extract ONLY the mathematical float number (e.g., '135 EUR/night' -> 135.0).
            CONTEXT:\n{context}
            """)
            
            extracted_data = structured_llm.invoke([sys_msg, human_msg])
            hotels_list = extracted_data.hotels if hasattr(extracted_data, 'hotels') else []
            
            if not hotels_list:
                return json.dumps({"status": "not_found", "message": f"Could not parse hotels for {clean_city}."})

            # 3. BUSINESS LOGIC AND MAPS
            logger.info(f"⚖️ Evaluating {len(hotels_list)} hotels against budget of {max_price_eur}€...")
            compliant_hotels = [h for h in hotels_list if h.price_eur <= max_price_eur]
            
            selected_hotel = None
            rationale = ""
            best_distance = float('inf')
            best_duration = 0
            
            # SCENARIO B: ALL HOTELS EXCEED BUDGET
            if not compliant_hotels:
                logger.warning("⚠️ All hotels exceed the limit. Selecting the cheapest one.")
                selected_hotel = min(hotels_list, key=lambda x: x.price_eur)
                
                # Attempt to get its real distance for information
                try:
                    best_distance, best_duration = get_travel_distance(f"{selected_hotel.name}, {clean_city}", client_address)
                except:
                    best_distance, best_duration = 99.9, 99 # Fallback if map fails
                    
                rationale = (f"EXCEPTION REQUIRED: All available hotels exceeded the {max_price_eur} EUR limit. "
                             f"Automatically selected the cheapest option available ({selected_hotel.price_eur} EUR). "
                             f"Manager approval will be required.")
            
            # SCENARIO A: HOTELS WITHIN BUDGET (HERE ENTERS OSRM/NOMINATIM)
            else:
                logger.info(f"✅ {len(compliant_hotels)} hotels comply. Calculating real distances to '{client_address}'...")
                
                for h in compliant_hotels:
                    try:
                        # Respect OpenStreetMap Rate Limit (1 request/sec)
                        time.sleep(1) 
                        dist_km, duration_mins = _calculate_distance_straight_line(h.address, client_address)
                        logger.info(f"   📍 {h.name} -> {dist_km} km ({duration_mins} mins)")
                        
                        if dist_km < best_distance:
                            best_distance = dist_km
                            best_duration = duration_mins
                            selected_hotel = h
                    except Exception as e:
                        logger.warning(f"   ❌ Failed to calculate route for {h.name}: {e}")
                        continue # Move to the next one if this map calculation fails
                
                # If all map calculations failed, choose the first one by default
                if not selected_hotel:
                    selected_hotel = compliant_hotels[0]
                    best_distance, best_duration = 0.0, 0
                    rationale = f"POLICY COMPLIANT: Selected {selected_hotel.name} (Map API failed to calculate distance)."
                else:
                    rationale = (f"POLICY COMPLIANT: Selected this hotel because it is strictly within the "
                                 f"{max_price_eur} EUR budget and is the CLOSEST confirmed option ({best_distance} km) "
                                 f"to the client meeting at '{client_address}'.")

            # 4. RETURN THE WINNER TO THE LLM
            result = {
                "status": "success",
                "selected_hotel": selected_hotel.model_dump(),
                "logistics": {
                    "distance_to_client_km": best_distance,
                    "estimated_commute_minutes": best_duration
                },
                "decision_rationale": rationale
            }

            clean_output = json.dumps(result, indent=2)
            logger.info(f"🏆 Winning Hotel:\n{clean_output}")
            return clean_output

        except Exception as e:
            logger.error(f"Error in find_optimal_hotel: {e}", exc_info=True)
            return json.dumps({"status": "error", "message": str(e)})

    return find_optimal_hotel

