"""
HMB Route Assignment Optimizer V2 for Uthangarai Chilling Center
================================================================

Given a new HMB's latitude, longitude, and expected milk volume, determines
which of the 7 existing milk collection routes at Uthangarai CC (Plant 1142)
should it be added to, minimizing cost (distance) and ensuring milk freshness.

Two distance modes:
  - haversine: Offline, GPS-based straight-line distance with road factor
  - osrm: Online, real road distances via OSRM public API

Algorithm: Minimum Insertion Cost + 2-Opt Improvement + Multi-Factor Scoring
"""

import csv
import json
import math
import os
import socket
import time
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Optional, List, Tuple


# --- Configuration -----------------------------------------------------------

# Configurable scoring weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "extra_distance": 0.40,      # Minimize additional KM added to route
    "total_route_km": 0.20,      # Prefer routes that stay short overall
    "centroid_proximity": 0.15,  # Logically coherent geographic assignment
    "uti_headroom": 0.25,        # Prefer underutilized routes with capacity
}

# Road factor multiplier for Haversine mode only.
ROAD_FACTOR = 1.3

# Milk spoilage constraint: max hours for total route duration (CC -> HMBs -> CC).
# ~4 hours for non-refrigerated trucks in Indian conditions.
MAX_ROUTE_HOURS = 4.0

# Average vehicle speed (km/h) for estimating route time
AVG_SPEED_KMH = 40.0

# Average time spent at each HMB for milk collection (hours)
AVG_STOP_TIME_HOURS = 10 / 60  # 10 minutes

# Uthangarai CC coordinates
CC_LAT = 12.308573
CC_LON = 78.535901

# OSRM public API base URL
OSRM_BASE_URL = "http://router.project-osrm.org"

# Data directory (relative to this script -- one level up, then into csv_files)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "..", "csv_files")


# --- Data Structures ---------------------------------------------------------

@dataclass
class HMB:
    """Represents a single milk collection point (Hire Milk Booth)."""
    sap_code: str
    name: str
    lat: float
    lon: float
    sequence: int
    distance_km: float  # Distance from previous stop (from master data)


@dataclass
class Route:
    """Represents a milk collection route with ordered HMB stops."""
    code: str              # e.g., "M10126"
    name: str              # e.g., "Bargoor"
    hmbs: list = field(default_factory=list)  # Ordered list of HMB objects
    total_km: float = 0.0
    capacity: int = 0
    current_milk_qty: float = 0.0
    uti_percent: float = 0.0
    transporter: str = ""
    vehicle_type: str = ""
    km_rate: float = 0.0
    fixed_rate: float = 0.0
    per_day_hire: float = 0.0
    cpl: float = 0.0


@dataclass
class InsertionResult:
    """Result of evaluating a new HMB insertion into a specific route."""
    route: Route
    position: int                # Insert position (0-based in stop list)
    extra_km: float              # Additional KM from insertion
    new_total_km: float          # Route total KM after insertion (pre-2opt)
    estimated_time_hours: float  # Estimated total route time (post-2opt)
    score: float = 0.0           # Weighted composite score (higher = better)
    prev_stop_name: str = ""     # Name of the stop before insertion point
    next_stop_name: str = ""     # Name of the stop after insertion point
    is_feasible: bool = True     # Whether insertion meets all constraints
    infeasibility_reason: str = ""
    # 2-opt improvement fields
    post_2opt_km: float = 0.0        # Total KM after 2-opt improvement
    improvement_km: float = 0.0      # KM saved by 2-opt
    optimized_stop_order: list = field(default_factory=list)  # HMB names after 2-opt


# --- Haversine Distance ------------------------------------------------------

def haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Calculate the great-circle distance between two GPS points.

    Args:
        lat1, lon1: Latitude and longitude of point 1 (degrees)
        lat2, lon2: Latitude and longitude of point 2 (degrees)

    Returns:
        Distance in kilometers (straight-line).
    """
    R = 6371.0  # Earth's radius in km

    lat1_r, lon1_r = math.radians(lat1), math.radians(lon1)
    lat2_r, lon2_r = math.radians(lat2), math.radians(lon2)

    dlat = lat2_r - lat1_r
    dlon = lon2_r - lon1_r

    a = math.sin(dlat / 2) ** 2 + \
        math.cos(lat1_r) * math.cos(lat2_r) * math.sin(dlon / 2) ** 2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c


def haversine_road_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Approximate road distance between two GPS points using Haversine.
    Applies the road factor multiplier to straight-line distance.
    """
    return haversine(lat1, lon1, lat2, lon2) * ROAD_FACTOR


# --- OSRM Distance -----------------------------------------------------------

# Maximum retries for OSRM API calls
OSRM_MAX_RETRIES = 3
OSRM_RETRY_BACKOFF = 2  # seconds, doubles each retry


def _osrm_request(url: str, timeout: int = 30) -> dict:
    """
    Make an OSRM API request with automatic retry on timeout.

    Args:
        url: The full OSRM API URL.
        timeout: Timeout in seconds per attempt.

    Returns:
        Parsed JSON response dict.

    Raises:
        RuntimeError: If all retries are exhausted.
    """
    last_error = None
    for attempt in range(1, OSRM_MAX_RETRIES + 1):
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=timeout) as response:
                return json.loads(response.read().decode())
        except (urllib.error.URLError, urllib.error.HTTPError,
                socket.timeout, ConnectionResetError) as e:
            last_error = e
            if attempt < OSRM_MAX_RETRIES:
                wait = OSRM_RETRY_BACKOFF * (2 ** (attempt - 1))
                time.sleep(wait)
    raise RuntimeError(f"OSRM API failed after {OSRM_MAX_RETRIES} retries: {last_error}")


def osrm_route_query(coords: List[Tuple[float, float]]) -> dict:
    """
    Query the OSRM route API for a full route.

    Args:
        coords: List of (lat, lon) tuples in visit order.

    Returns:
        Dict with 'distance' (km) and 'duration' (hours).

    Raises:
        RuntimeError: If OSRM API call fails after retries.
    """
    if len(coords) < 2:
        return {"distance": 0.0, "duration": 0.0}

    # OSRM expects lon,lat order, semicolon-separated
    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = f"{OSRM_BASE_URL}/route/v1/driving/{coord_str}?overview=false"

    data = _osrm_request(url, timeout=30)
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM error: {data.get('code')}")
    route = data["routes"][0]
    return {
        "distance": route["distance"] / 1000.0,  # meters to km
        "duration": route["duration"] / 3600.0,   # seconds to hours
    }


def osrm_table_query(coords: List[Tuple[float, float]]) -> List[List[float]]:
    """
    Query the OSRM table API for a distance matrix between all coordinate pairs.

    Args:
        coords: List of (lat, lon) tuples.

    Returns:
        2D list of distances in km: distances[i][j] = distance from i to j.

    Raises:
        RuntimeError: If OSRM API call fails after retries.
    """
    if len(coords) < 2:
        return [[0.0]]

    coord_str = ";".join(f"{lon},{lat}" for lat, lon in coords)
    url = (
        f"{OSRM_BASE_URL}/table/v1/driving/{coord_str}"
        f"?annotations=distance"
    )

    data = _osrm_request(url, timeout=45)
    if data.get("code") != "Ok":
        raise RuntimeError(f"OSRM table error: {data.get('code')}")
    # Convert meters to km
    return [
        [d / 1000.0 for d in row]
        for row in data["distances"]
    ]


# --- Distance Helpers ---------------------------------------------------------

def get_pair_distance(lat1: float, lon1: float, lat2: float, lon2: float,
                      mode: str = "haversine") -> float:
    """Get distance between two points using the specified mode."""
    if mode == "osrm":
        result = osrm_route_query([(lat1, lon1), (lat2, lon2)])
        return result["distance"]
    else:
        return haversine_road_distance(lat1, lon1, lat2, lon2)


def calculate_route_distance_from_coords(
    coords: List[Tuple[float, float]],
    mode: str = "haversine",
    distance_matrix: Optional[dict] = None,
) -> float:
    """
    Calculate total route distance for ordered coordinates.

    For haversine: sums consecutive pair distances.
    For osrm: uses the batch route API (single call).
    For 2-opt with matrix: uses pre-computed distance matrix.
    """
    if distance_matrix is not None:
        # Use the pre-computed matrix for fast 2-opt
        total = 0.0
        for i in range(len(coords) - 1):
            key = (coords[i], coords[i + 1])
            total += distance_matrix.get(key, 0.0)
        return total

    if mode == "osrm":
        result = osrm_route_query(coords)
        return result["distance"]
    else:
        total = 0.0
        for i in range(len(coords) - 1):
            total += haversine_road_distance(
                coords[i][0], coords[i][1],
                coords[i + 1][0], coords[i + 1][1],
            )
        return total


# --- Data Loading -------------------------------------------------------------

def _parse_lat_lon(coord_str: str) -> Optional[tuple]:
    """Parse a 'lat, lon' string into (float, float). Returns None on failure."""
    if not coord_str or not coord_str.strip():
        return None
    try:
        parts = coord_str.strip().split(",")
        if len(parts) != 2:
            return None
        lat = float(parts[0].strip())
        lon = float(parts[1].strip())
        if -90 <= lat <= 90 and -180 <= lon <= 180:
            return (lat, lon)
        return None
    except (ValueError, IndexError):
        return None


def load_route_summary() -> dict:
    """
    Load route summary data from HMB Details_Summary.csv.

    Returns:
        Dict mapping route_code -> {capacity, milk_qty, total_km, ...}
    """
    summary_file = os.path.join(CSV_DIR, "HMB Details_Summary.csv")
    routes_info = {}

    with open(summary_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if not row or not row[0].strip() or row[0].strip() == "Grand Total":
                continue

            route_code = row[2].strip() if len(row) > 2 else ""
            if not route_code.startswith("M"):
                continue

            routes_info[route_code] = {
                "plant": row[0].strip(),
                "plant_name": row[1].strip(),
                "route_name": row[3].strip() if len(row) > 3 else "",
                "transporter_code": row[4].strip() if len(row) > 4 else "",
                "transporter_name": row[5].strip() if len(row) > 5 else "",
                "truck_no": row[6].strip() if len(row) > 6 else "",
                "vehicle_type": row[7].strip() if len(row) > 7 else "",
                "capacity": int(float(row[8])) if len(row) > 8 and row[8].strip() else 0,
                "milk_qty": float(row[9]) if len(row) > 9 and row[9].strip() else 0.0,
                "per_day_km": float(row[10]) if len(row) > 10 and row[10].strip() else 0.0,
                "km_rate": float(row[11]) if len(row) > 11 and row[11].strip() else 0.0,
                "fixed_rate": float(row[12]) if len(row) > 12 and row[12].strip() else 0.0,
                "per_day_hire": float(row[13]) if len(row) > 13 and row[13].strip() else 0.0,
                "cpl": float(row[14]) if len(row) > 14 and row[14].strip() else 0.0,
                "uti_percent": float(row[15]) if len(row) > 15 and row[15].strip() else 0.0,
            }

    return routes_info


def load_route_data() -> list:
    """
    Load ordered route sequences with HMB coordinates from Master Data CSV.
    Filters for Uthangarai CC (Plant 1142) only.

    Returns:
        List of Route objects with ordered HMB sequences.
    """
    master_file = os.path.join(CSV_DIR, "HMB Details_Master Data.csv")
    summary = load_route_summary()

    # Parse master data: group HMBs by route code
    route_hmbs = {}  # route_code -> list of (sequence, HMB)

    with open(master_file, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)  # Skip header

        for row in reader:
            if not row or len(row) < 11:
                continue

            plant = row[0].strip()
            route_name = row[2].strip()
            center_code = row[3].strip()
            center_name = row[4].strip()
            sequence_str = row[5].strip()
            route_code = row[6].strip()
            distance_str = row[7].strip()
            coord_str = row[10].strip()

            # Uthangarai only
            if not plant or plant != "1142":
                continue
            if not route_code.startswith("M"):
                continue
            if center_name.lower() in ("cc", "total", ""):
                continue

            coords = _parse_lat_lon(coord_str)
            if not coords:
                continue

            try:
                sequence = int(float(sequence_str)) if sequence_str else 0
                distance = float(distance_str) if distance_str else 0.0
            except ValueError:
                continue

            hmb = HMB(
                sap_code=center_code,
                name=center_name,
                lat=coords[0],
                lon=coords[1],
                sequence=sequence,
                distance_km=distance,
            )

            if route_code not in route_hmbs:
                route_hmbs[route_code] = {"name": route_name, "hmbs": []}
            route_hmbs[route_code]["hmbs"].append((sequence, hmb))

    # Build Route objects
    routes = []
    for route_code, data in route_hmbs.items():
        # Sort HMBs by sequence
        sorted_hmbs = [hmb for _, hmb in sorted(data["hmbs"], key=lambda x: x[0])]

        # Get summary info
        info = summary.get(route_code, {})

        route = Route(
            code=route_code,
            name=data["name"] or info.get("route_name", ""),
            hmbs=sorted_hmbs,
            total_km=info.get("per_day_km", 0.0),
            capacity=info.get("capacity", 0),
            current_milk_qty=info.get("milk_qty", 0.0),
            uti_percent=info.get("uti_percent", 0.0),
            transporter=info.get("transporter_name", ""),
            vehicle_type=info.get("vehicle_type", ""),
            km_rate=info.get("km_rate", 0.0),
            fixed_rate=info.get("fixed_rate", 0.0),
            per_day_hire=info.get("per_day_hire", 0.0),
            cpl=info.get("cpl", 0.0),
        )
        routes.append(route)

    # Sort routes by code for consistent ordering
    routes.sort(key=lambda r: r.code)
    return routes


# --- Route Calculations -------------------------------------------------------

def build_route_coords(route: Route) -> List[Tuple[float, float]]:
    """Build ordered coordinate list: CC -> HMBs -> CC."""
    coords = [(CC_LAT, CC_LON)]
    for hmb in route.hmbs:
        coords.append((hmb.lat, hmb.lon))
    coords.append((CC_LAT, CC_LON))
    return coords


def calculate_route_distance(route: Route, mode: str = "haversine") -> float:
    """
    Calculate total route distance using GPS coordinates.
    CC -> HMB1 -> HMB2 -> ... -> HMBn -> CC
    """
    if not route.hmbs:
        return 0.0
    coords = build_route_coords(route)
    return calculate_route_distance_from_coords(coords, mode)


def estimate_route_time(total_km: float, num_stops: int) -> float:
    """
    Estimate total route time in hours.

    Args:
        total_km: Total route distance in km
        num_stops: Number of HMB stops on the route

    Returns:
        Estimated time in hours (driving + stop time)
    """
    driving_time = total_km / AVG_SPEED_KMH
    stop_time = num_stops * AVG_STOP_TIME_HOURS
    return driving_time + stop_time


def calculate_route_centroid(route: Route) -> tuple:
    """Calculate the geographic centroid (average lat/lon) of a route's HMBs."""
    if not route.hmbs:
        return (CC_LAT, CC_LON)

    avg_lat = sum(h.lat for h in route.hmbs) / len(route.hmbs)
    avg_lon = sum(h.lon for h in route.hmbs) / len(route.hmbs)
    return (avg_lat, avg_lon)


# --- 2-Opt Route Improvement --------------------------------------------------

def _build_distance_matrix_haversine(
    coords: List[Tuple[float, float]],
) -> dict:
    """Build a pairwise distance lookup dict for haversine mode."""
    matrix = {}
    for i, c1 in enumerate(coords):
        for j, c2 in enumerate(coords):
            if i != j:
                matrix[(c1, c2)] = haversine_road_distance(c1[0], c1[1], c2[0], c2[1])
    return matrix


def _build_distance_matrix_osrm(
    coords: List[Tuple[float, float]],
) -> dict:
    """
    Build a pairwise distance lookup dict using OSRM table API.
    Single API call for all pairs.
    """
    unique_coords = list(dict.fromkeys(coords))  # preserve order, deduplicate
    table = osrm_table_query(unique_coords)
    coord_to_idx = {c: i for i, c in enumerate(unique_coords)}

    matrix = {}
    for c1 in unique_coords:
        for c2 in unique_coords:
            if c1 != c2:
                i = coord_to_idx[c1]
                j = coord_to_idx[c2]
                matrix[(c1, c2)] = table[i][j]
    return matrix


def _total_distance_from_matrix(
    coords: List[Tuple[float, float]],
    matrix: dict,
) -> float:
    """Calculate total route distance using pre-computed matrix."""
    total = 0.0
    for i in range(len(coords) - 1):
        key = (coords[i], coords[i + 1])
        total += matrix.get(key, 0.0)
    return total


def optimize_2opt(
    coords: List[Tuple[float, float]],
    mode: str = "haversine",
) -> Tuple[List[Tuple[float, float]], float]:
    """
    Apply 2-opt improvement to a route.

    The first and last coordinates (CC depot) are fixed; only the intermediate
    HMB stops are candidates for reordering.

    For OSRM mode: builds a distance matrix via a single OSRM table API call,
    then runs 2-opt locally using the cached distances (no more API calls).

    Args:
        coords: Ordered list of (lat, lon) starting and ending at CC.
        mode: Distance calculation mode.

    Returns:
        Tuple of (improved_coords, improved_distance).
    """
    if len(coords) <= 3:
        # 0 or 1 intermediate stops -- nothing to optimize
        dist = calculate_route_distance_from_coords(coords, mode)
        return coords, dist

    # Build distance matrix (1 API call for OSRM, local for haversine)
    if mode == "osrm":
        matrix = _build_distance_matrix_osrm(coords)
    else:
        matrix = _build_distance_matrix_haversine(coords)

    best_coords = list(coords)
    best_distance = _total_distance_from_matrix(best_coords, matrix)
    improved = True

    while improved:
        improved = False
        # Only swap intermediate stops (indices 1 to n-2, keeping 0 and n-1 as CC)
        for i in range(1, len(best_coords) - 2):
            for j in range(i + 1, len(best_coords) - 1):
                # Reverse the segment between i and j (inclusive)
                new_coords = (
                    best_coords[:i]
                    + best_coords[i:j + 1][::-1]
                    + best_coords[j + 1:]
                )
                new_distance = _total_distance_from_matrix(new_coords, matrix)
                if new_distance < best_distance - 0.01:
                    best_coords = new_coords
                    best_distance = new_distance
                    improved = True

    return best_coords, best_distance


# --- Capacity Pre-Filter ------------------------------------------------------

def filter_routes_by_capacity(routes: list,
                              expected_milk_qty: float) -> Tuple[list, list]:
    """
    Hard pre-filter: reject routes that cannot fit the new HMB's milk.

    This is the MOST IMPORTANT constraint -- checked before any distance
    calculations to avoid recommending an overloaded truck.

    Args:
        routes: All available Route objects.
        expected_milk_qty: Expected milk volume (liters) from the new HMB.

    Returns:
        Tuple of (eligible_routes, rejected_routes_with_reasons).
    """
    eligible = []
    rejected = []

    for route in routes:
        remaining = route.capacity - route.current_milk_qty
        if remaining < expected_milk_qty:
            rejected.append({
                "route": route,
                "reason": (
                    f"Capacity exceeded: {route.current_milk_qty:.0f} + "
                    f"{expected_milk_qty:.0f} = {route.current_milk_qty + expected_milk_qty:.0f} "
                    f"> {route.capacity} ltrs"
                ),
            })
        else:
            eligible.append(route)

    return eligible, rejected


# --- Insertion Cost Algorithm -------------------------------------------------

def _build_stop_list(route: Route) -> list:
    """Build stop list with CC bookends: [(lat, lon, name), ...]"""
    stops = [(CC_LAT, CC_LON, "Chilling Center (CC)")]
    for hmb in route.hmbs:
        stops.append((hmb.lat, hmb.lon, hmb.name))
    stops.append((CC_LAT, CC_LON, "Chilling Center (CC)"))
    return stops


def _resolve_stop_name(coord: Tuple[float, float], new_lat: float, new_lon: float,
                       route_hmbs: list) -> str:
    """Map a coordinate back to an HMB name."""
    if abs(coord[0] - new_lat) < 0.0001 and abs(coord[1] - new_lon) < 0.0001:
        return "NEW HMB"
    for hmb in route_hmbs:
        if abs(coord[0] - hmb.lat) < 0.0001 and abs(coord[1] - hmb.lon) < 0.0001:
            return hmb.name
    return "Unknown"


def calculate_insertion_cost(route: Route, new_lat: float, new_lon: float,
                             mode: str = "haversine") -> InsertionResult:
    """
    Find the optimal insertion position for a new HMB in a route.

    Step 1: Use Haversine to quickly scan all positions (even in OSRM mode).
    Step 2: Insert at the best position found.
    Step 3: Apply 2-opt (uses OSRM table API if in OSRM mode -- single call).
    Step 4: Get final route distance (OSRM batch if in OSRM mode -- single call).

    This minimizes OSRM API calls to just 2 per route (table + route).

    Args:
        route: The route to evaluate
        new_lat, new_lon: Coordinates of the new HMB
        mode: Distance calculation mode ("haversine" or "osrm")

    Returns:
        InsertionResult with the best position and cost details
    """
    best_result = None
    best_extra_km = float("inf")

    stops = _build_stop_list(route)

    # STEP 1: Scan all positions using Haversine (fast, no API calls)
    current_route_km_hav = calculate_route_distance(route, "haversine")

    for i in range(len(stops) - 1):
        prev_lat, prev_lon, prev_name = stops[i]
        next_lat, next_lon, next_name = stops[i + 1]

        # Always use haversine for the scan (it's cheap)
        dist_current = haversine_road_distance(prev_lat, prev_lon, next_lat, next_lon)
        dist_to_new = haversine_road_distance(prev_lat, prev_lon, new_lat, new_lon)
        dist_from_new = haversine_road_distance(new_lat, new_lon, next_lat, next_lon)

        extra_km = dist_to_new + dist_from_new - dist_current

        if extra_km < best_extra_km:
            best_extra_km = extra_km
            new_total_km = current_route_km_hav + extra_km
            num_stops_count = len(route.hmbs) + 1
            est_time = estimate_route_time(new_total_km, num_stops_count)

            best_result = InsertionResult(
                route=route,
                position=i,
                extra_km=round(extra_km, 2),
                new_total_km=round(new_total_km, 2),
                estimated_time_hours=round(est_time, 2),
                prev_stop_name=prev_name,
                next_stop_name=next_name,
            )

    if not best_result:
        return best_result

    # STEP 2: Build inserted route
    hmb_coords_names = [(hmb.lat, hmb.lon, hmb.name) for hmb in route.hmbs]
    insert_idx = best_result.position  # position in HMB-only list
    hmb_coords_names.insert(insert_idx, (new_lat, new_lon, "NEW HMB"))

    inserted_coords = [(CC_LAT, CC_LON)]
    for lat, lon, name in hmb_coords_names:
        inserted_coords.append((lat, lon))
    inserted_coords.append((CC_LAT, CC_LON))

    # STEP 3: 2-opt improvement (for OSRM: uses table API -- 1 call)
    optimized_coords, optimized_dist = optimize_2opt(inserted_coords, mode)

    # If in OSRM mode, the optimized_dist came from the table matrix.
    # Get the precise route distance via OSRM route API for the final order.
    if mode == "osrm":
        result = osrm_route_query(optimized_coords)
        optimized_dist = result["distance"]

    # Also compute pre-2opt distance in the chosen mode
    if mode == "osrm":
        pre_result = osrm_route_query(inserted_coords)
        pre_2opt_dist = pre_result["distance"]
    else:
        pre_2opt_dist = calculate_route_distance_from_coords(inserted_coords, "haversine")

    # Extract stop names from optimized order
    optimized_names = []
    for oc in optimized_coords[1:-1]:  # Skip CC at start and end
        optimized_names.append(_resolve_stop_name(oc, new_lat, new_lon, route.hmbs))

    num_stops_count = len(route.hmbs) + 1
    opt_time = estimate_route_time(optimized_dist, num_stops_count)

    best_result.new_total_km = round(pre_2opt_dist, 2)
    best_result.extra_km = round(pre_2opt_dist - calculate_route_distance(route, mode)
                                 if mode == "osrm" else best_result.extra_km, 2)
    best_result.post_2opt_km = round(optimized_dist, 2)
    best_result.improvement_km = round(pre_2opt_dist - optimized_dist, 2)
    best_result.optimized_stop_order = optimized_names
    best_result.estimated_time_hours = round(opt_time, 2)

    # Feasibility check: time constraint (post-2opt)
    if best_result.estimated_time_hours > MAX_ROUTE_HOURS:
        best_result.is_feasible = False
        best_result.infeasibility_reason = (
            f"Route time ({best_result.estimated_time_hours:.1f}h) "
            f"exceeds max {MAX_ROUTE_HOURS:.1f}h for milk freshness"
        )

    return best_result


# --- Scoring ------------------------------------------------------------------

def score_routes(
    results: list,
    new_lat: float,
    new_lon: float,
    mode: str = "haversine",
    weights: Optional[dict] = None,
) -> list:
    """
    Score and rank all routes by weighted multi-factor analysis.

    Factors:
        1. Extra distance (lower is better)
        2. Post-2opt total route KM (lower is better)
        3. Proximity to route centroid (lower is better)
        4. UTI headroom (more headroom is better)

    Args:
        results: List of InsertionResult objects
        new_lat, new_lon: New HMB coordinates
        mode: Distance calculation mode
        weights: Optional custom scoring weights

    Returns:
        Sorted list of InsertionResult objects (best first)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    if not results:
        return results

    # Collect raw values for normalization
    extra_kms = [r.extra_km for r in results]
    total_kms = [r.post_2opt_km for r in results]
    centroid_dists = []
    uti_headrooms = []

    for r in results:
        centroid = calculate_route_centroid(r.route)
        # Use haversine for centroid distance (fast, mode-independent)
        dist = haversine_road_distance(new_lat, new_lon, centroid[0], centroid[1])
        centroid_dists.append(dist)

        headroom = max(0, r.route.capacity - r.route.current_milk_qty)
        uti_headrooms.append(headroom)

    # Min-max normalize each factor (0 = worst, 1 = best)
    def normalize_lower_better(values):
        """Lower raw value = better = higher normalized score."""
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return [1.0] * len(values)
        return [1.0 - (v - min_v) / (max_v - min_v) for v in values]

    def normalize_higher_better(values):
        """Higher raw value = better = higher normalized score."""
        min_v, max_v = min(values), max(values)
        if max_v == min_v:
            return [1.0] * len(values)
        return [(v - min_v) / (max_v - min_v) for v in values]

    norm_extra = normalize_lower_better(extra_kms)
    norm_total = normalize_lower_better(total_kms)
    norm_centroid = normalize_lower_better(centroid_dists)
    norm_headroom = normalize_higher_better(uti_headrooms)

    # Calculate weighted score for each route
    for i, result in enumerate(results):
        score = (
            weights["extra_distance"] * norm_extra[i]
            + weights["total_route_km"] * norm_total[i]
            + weights["centroid_proximity"] * norm_centroid[i]
            + weights["uti_headroom"] * norm_headroom[i]
        )
        result.score = round(score, 4)

    # Sort by: feasible first, then by score descending
    results.sort(key=lambda r: (-r.is_feasible, -r.score))
    return results


# --- Main Recommendation Function --------------------------------------------

def recommend_route(
    lat: float,
    lon: float,
    expected_milk_qty: float,
    mode: str = "haversine",
    weights: Optional[dict] = None,
    print_output: bool = True,
) -> dict:
    """
    Recommend the best route for a new HMB based on GPS coordinates and milk volume.

    Uses minimum insertion cost + 2-opt improvement + multi-factor scoring.

    Args:
        lat: Latitude of the new HMB
        lon: Longitude of the new HMB
        expected_milk_qty: Expected daily milk volume (liters) from the new HMB
        mode: Distance calculation mode -- "haversine" or "osrm"
        weights: Optional custom scoring weights dict
        print_output: If True, prints formatted recommendation to stdout

    Returns:
        dict with keys:
            - 'recommended': InsertionResult for the best route (or None)
            - 'all_results': List of all InsertionResult objects, ranked
            - 'rejected_routes': Routes rejected by capacity pre-filter
            - 'input': input parameters
            - 'cc_location': CC coordinates
            - 'config': configuration values used
    """
    # Validate inputs
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coordinates: ({lat}, {lon})")

    if expected_milk_qty < 0:
        raise ValueError(f"Expected milk quantity must be >= 0, got {expected_milk_qty}")

    if mode not in ("haversine", "osrm"):
        raise ValueError(f"Invalid mode: '{mode}'. Must be 'haversine' or 'osrm'")

    if weights is not None:
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    # Load route data
    routes = load_route_data()
    if not routes:
        raise RuntimeError("No route data could be loaded. Check CSV files.")

    # HARD PRE-FILTER: Capacity check (most important constraint)
    eligible_routes, rejected_routes = filter_routes_by_capacity(routes, expected_milk_qty)

    if print_output and mode == "osrm":
        print("\n  [~] Querying OSRM API for real road distances...")
        print(f"  [~] Processing {len(eligible_routes)} eligible routes...\n")

    # Calculate insertion cost for each eligible route
    results = []
    for route in eligible_routes:
        if print_output and mode == "osrm":
            print(f"      Processing: {route.code} {route.name}...", flush=True)
        result = calculate_insertion_cost(route, lat, lon, mode)
        if result:
            results.append(result)

    # Score and rank
    ranked = score_routes(results, lat, lon, mode, weights)

    # Build return value
    recommended = ranked[0] if ranked else None
    output = {
        "recommended": recommended,
        "all_results": ranked,
        "rejected_routes": rejected_routes,
        "input": {
            "lat": lat,
            "lon": lon,
            "expected_milk_qty": expected_milk_qty,
            "mode": mode,
        },
        "cc_location": {"lat": CC_LAT, "lon": CC_LON},
        "config": {
            "road_factor": ROAD_FACTOR if mode == "haversine" else "N/A (OSRM)",
            "max_route_hours": MAX_ROUTE_HOURS,
            "avg_speed_kmh": AVG_SPEED_KMH,
            "avg_stop_time_min": AVG_STOP_TIME_HOURS * 60,
            "weights": weights or DEFAULT_WEIGHTS,
        },
    }

    # Print formatted output
    if print_output:
        _print_recommendation(output)

    return output


# Legacy alias for backward compatibility
def haversine_recommendation(lat, lon, weights=None, print_output=True):
    """Legacy wrapper -- calls recommend_route with haversine mode and 0 milk qty."""
    return recommend_route(lat, lon, expected_milk_qty=0, mode="haversine",
                           weights=weights, print_output=print_output)


def _print_recommendation(output: dict):
    """Print a formatted recommendation to stdout."""
    rec = output["recommended"]
    results = output["all_results"]
    rejected = output["rejected_routes"]
    inp = output["input"]
    config = output["config"]
    mode_label = "OSRM (road distance)" if inp["mode"] == "osrm" else "Haversine (est. road)"

    print()
    print("=" * 70)
    print("  HMB ROUTE ASSIGNMENT -- Uthangarai Chilling Center (1142)")
    print(f"  Distance Mode: {mode_label}")
    print("=" * 70)
    print()
    print(f"  [>] New HMB Location:   ({inp['lat']:.6f}, {inp['lon']:.6f})")
    print(f"  [*] Chilling Center:    ({CC_LAT}, {CC_LON})")
    print(f"  [~] Expected milk:      {inp['expected_milk_qty']:.0f} ltrs/day")

    dist_to_cc = haversine_road_distance(inp['lat'], inp['lon'], CC_LAT, CC_LON)
    print(f"  [~] Distance to CC:     {dist_to_cc:.1f} KM (est.)")
    print()

    # Capacity rejections
    if rejected:
        print(f"  --- Rejected by Capacity ({len(rejected)} routes) ----------------")
        for rej in rejected:
            r = rej["route"]
            print(f"  [X]  {r.code}  {r.name:<18s}  {rej['reason']}")
        print()

    if not rec:
        print("  [!!] No eligible routes found. All routes exceed capacity.")
        print("=" * 70)
        print()
        return

    if rec.is_feasible:
        print(f"  [OK] RECOMMENDED: Route {rec.route.code} -- {rec.route.name}")
    else:
        print(f"  [!]  BEST AVAILABLE: Route {rec.route.code} -- {rec.route.name}")
        print(f"       Warning: {rec.infeasibility_reason}")

    print(f"     Insert between: {rec.prev_stop_name} -> {rec.next_stop_name}")
    print(f"     Position: #{rec.position + 1} in sequence")
    print(f"     Extra distance: +{rec.extra_km:.1f} KM")
    new_milk = rec.route.current_milk_qty + inp['expected_milk_qty']
    cap = rec.route.capacity if rec.route.capacity > 0 else 1
    print(f"     Milk load after: {new_milk:.0f}/{rec.route.capacity} ltrs "
          f"(UTI: {new_milk / cap * 100:.1f}%)")
    print(f"     Route KM (post-2opt): {rec.post_2opt_km:.1f} KM")
    if rec.improvement_km > 0:
        print(f"     2-opt saved: {rec.improvement_km:.1f} KM")
    print(f"     Estimated time: {rec.estimated_time_hours:.1f} hours")
    print(f"     Vehicle: {rec.route.vehicle_type} ({rec.route.transporter})")
    print(f"     Score: {rec.score:.2f} / 1.00")
    print()

    # Optimized stop order
    if rec.optimized_stop_order:
        print("  --- Optimized Stop Order (after 2-opt) --------------------")
        print(f"     CC -> ", end="")
        print(" -> ".join(rec.optimized_stop_order), end="")
        print(" -> CC")
        print()

    # All eligible routes ranked
    print(f"  --- Eligible Routes Ranked ({len(results)}) -----------------------")
    for i, r in enumerate(results):
        feasible_marker = "[OK]" if r.is_feasible else "[X] "
        two_opt_note = f"  2opt:-{r.improvement_km:.1f}" if r.improvement_km > 0 else ""
        print(f"  {feasible_marker} #{i+1}  {r.route.code}  {r.route.name:<18s}"
              f"  +{r.extra_km:>6.1f} KM"
              f"  Post2opt: {r.post_2opt_km:.1f}"
              f"  Time: {r.estimated_time_hours:.1f}h"
              f"  Score: {r.score:.2f}"
              f"{two_opt_note}")

    print()
    print(f"  --- Configuration -----------------------------------------------")
    if inp["mode"] == "haversine":
        print(f"  Road factor: {config['road_factor']}x | ", end="")
    else:
        print(f"  Distance: OSRM (real roads) | ", end="")
    print(f"Max time: {config['max_route_hours']}h | "
          f"Avg speed: {config['avg_speed_kmh']} km/h | "
          f"Stop time: {config['avg_stop_time_min']:.0f} min")
    w = config["weights"]
    print(f"  Weights: distance={w['extra_distance']}, "
          f"total_km={w['total_route_km']}, "
          f"centroid={w['centroid_proximity']}, "
          f"uti={w['uti_headroom']}")
    print("=" * 70)
    print()


# --- CLI Entry Point ----------------------------------------------------------

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HMB Route Assignment Optimizer V2 for Uthangarai CC"
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude of new HMB")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of new HMB")
    parser.add_argument("--milk-qty", type=float, required=True,
                        help="Expected daily milk volume (liters) from the new HMB")
    parser.add_argument("--mode", choices=["haversine", "osrm"], default="haversine",
                        help="Distance calculation mode (default: haversine)")
    parser.add_argument("--road-factor", type=float, default=ROAD_FACTOR,
                        help=f"Road distance multiplier for haversine mode (default: {ROAD_FACTOR})")
    args = parser.parse_args()

    # Allow overriding road factor from CLI
    if args.mode == "haversine":
        ROAD_FACTOR = args.road_factor

    recommend_route(args.lat, args.lon, args.milk_qty, mode=args.mode)
