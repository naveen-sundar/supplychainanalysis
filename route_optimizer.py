"""
HMB Route Assignment Optimizer for Uthangarai Chilling Center
=============================================================

Given a new HMB's latitude and longitude, determines which of the 7 existing
milk collection routes at Uthangarai CC (Plant 1142) should it be added to,
minimizing cost (distance) and ensuring milk freshness (time).

Algorithm: Minimum Insertion Cost with Multi-Factor Scoring
"""

import csv
import math
import os
from dataclasses import dataclass, field
from typing import Optional


# ─── Configuration ───────────────────────────────────────────────────────────

# Configurable scoring weights (must sum to 1.0)
DEFAULT_WEIGHTS = {
    "extra_distance": 0.50,      # Minimize additional KM added to route
    "total_route_km": 0.20,      # Prefer routes that stay short overall
    "centroid_proximity": 0.15,  # Logically coherent geographic assignment
    "uti_headroom": 0.15,        # Prefer underutilized routes with capacity
}

# Road factor multiplier: Haversine gives straight-line distance.
# Multiply by this factor to approximate actual road distance.
ROAD_FACTOR = 1.3

# Milk spoilage constraint: max hours for total route duration (CC → HMBs → CC).
# Existing routes take 3.6–5.6 hours based on GPS data. Raw milk in non-refrigerated
# trucks can last ~4-6 hours depending on ambient temperature. Set to 6h to allow
# current operations while flagging routes that become too long after insertion.
MAX_ROUTE_HOURS = 6.0

# Average vehicle speed (km/h) for estimating route time
AVG_SPEED_KMH = 25.0

# Average time spent at each HMB for milk collection (hours)
AVG_STOP_TIME_HOURS = 10 / 60  # 10 minutes

# Uthangarai CC coordinates
CC_LAT = 12.308573
CC_LON = 78.535901

# Data directory (relative to this script)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_DIR = os.path.join(BASE_DIR, "csv_files")


# ─── Data Structures ────────────────────────────────────────────────────────

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
    position: int              # Insert position (0-based in HMB list)
    extra_km: float            # Additional KM from insertion
    new_total_km: float        # Route total KM after insertion
    estimated_time_hours: float  # Estimated total route time after insertion
    score: float = 0.0        # Weighted composite score (higher = better)
    prev_stop_name: str = ""   # Name of the stop before insertion point
    next_stop_name: str = ""   # Name of the stop after insertion point
    is_feasible: bool = True   # Whether insertion meets all constraints
    infeasibility_reason: str = ""


# ─── Haversine Distance ─────────────────────────────────────────────────────

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


def road_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """
    Approximate road distance between two GPS points.
    Applies the road factor multiplier to Haversine distance.
    """
    return haversine(lat1, lon1, lat2, lon2) * ROAD_FACTOR


# ─── Data Loading ────────────────────────────────────────────────────────────

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

    Returns:
        List of Route objects with ordered HMB sequences.
    """
    master_file = os.path.join(CSV_DIR, "HMB Details_Master Data.csv")
    summary = load_route_summary()

    # Parse master data: group HMBs by route code
    route_hmbs = {}  # route_code -> list of (sequence, HMB)
    current_route_name = ""

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

            # Skip rows that are CC returns, totals, or empty
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


# ─── Route Calculations ─────────────────────────────────────────────────────

def calculate_route_distance(route: Route) -> float:
    """
    Calculate total route distance using GPS coordinates.
    CC → HMB1 → HMB2 → ... → HMBn → CC
    """
    if not route.hmbs:
        return 0.0

    total = 0.0
    # CC to first HMB
    total += road_distance(CC_LAT, CC_LON, route.hmbs[0].lat, route.hmbs[0].lon)

    # Between consecutive HMBs
    for i in range(len(route.hmbs) - 1):
        total += road_distance(
            route.hmbs[i].lat, route.hmbs[i].lon,
            route.hmbs[i + 1].lat, route.hmbs[i + 1].lon,
        )

    # Last HMB back to CC
    total += road_distance(
        route.hmbs[-1].lat, route.hmbs[-1].lon,
        CC_LAT, CC_LON,
    )

    return total


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


# ─── Insertion Cost Algorithm ───────────────────────────────────────────────

def calculate_insertion_cost(route: Route, new_lat: float, new_lon: float) -> InsertionResult:
    """
    Find the optimal insertion position for a new HMB in a route.

    Tries every possible position and returns the one with minimum extra distance.
    The insertion cost at position i is:
        dist(prev, new) + dist(new, next) - dist(prev, next)

    Args:
        route: The route to evaluate
        new_lat, new_lon: Coordinates of the new HMB

    Returns:
        InsertionResult with the best position and cost details
    """
    best_result = None
    best_extra_km = float("inf")

    # Build the full sequence: CC → HMBs → CC
    stops = [(CC_LAT, CC_LON, "Chilling Center (CC)")]
    for hmb in route.hmbs:
        stops.append((hmb.lat, hmb.lon, hmb.name))
    stops.append((CC_LAT, CC_LON, "Chilling Center (CC)"))

    # Current route distance (calculated from GPS)
    current_route_km = calculate_route_distance(route)

    # Try inserting at each position (between stop i and stop i+1)
    for i in range(len(stops) - 1):
        prev_lat, prev_lon, prev_name = stops[i]
        next_lat, next_lon, next_name = stops[i + 1]

        # Current distance between these two stops
        dist_current = road_distance(prev_lat, prev_lon, next_lat, next_lon)

        # Distance with new HMB inserted
        dist_to_new = road_distance(prev_lat, prev_lon, new_lat, new_lon)
        dist_from_new = road_distance(new_lat, new_lon, next_lat, next_lon)

        # Extra distance from this insertion
        extra_km = dist_to_new + dist_from_new - dist_current

        if extra_km < best_extra_km:
            best_extra_km = extra_km
            new_total_km = current_route_km + extra_km
            num_stops = len(route.hmbs) + 1  # +1 for new HMB
            est_time = estimate_route_time(new_total_km, num_stops)

            best_result = InsertionResult(
                route=route,
                position=i,  # Insert after this index in stop list
                extra_km=round(extra_km, 2),
                new_total_km=round(new_total_km, 2),
                estimated_time_hours=round(est_time, 2),
                prev_stop_name=prev_name,
                next_stop_name=next_name,
            )

    # Check feasibility constraints
    if best_result:
        # Time constraint: milk must reach CC within MAX_ROUTE_HOURS
        if best_result.estimated_time_hours > MAX_ROUTE_HOURS:
            best_result.is_feasible = False
            best_result.infeasibility_reason = (
                f"Route time ({best_result.estimated_time_hours:.1f}h) "
                f"exceeds max {MAX_ROUTE_HOURS:.1f}h for milk freshness"
            )

        # Capacity constraint: check if vehicle can handle more milk
        remaining_capacity = route.capacity - route.current_milk_qty
        if remaining_capacity <= 0:
            best_result.is_feasible = False
            best_result.infeasibility_reason = (
                f"Vehicle at full capacity ({route.current_milk_qty:.0f}/{route.capacity} ltrs)"
            )

    return best_result


# ─── Scoring ────────────────────────────────────────────────────────────────

def score_routes(
    results: list,
    new_lat: float,
    new_lon: float,
    weights: Optional[dict] = None,
) -> list:
    """
    Score and rank all routes by weighted multi-factor analysis.

    Factors:
        1. Extra distance (lower is better)
        2. New total route KM (lower is better)
        3. Proximity to route centroid (lower is better)
        4. UTI headroom (more headroom is better)

    Args:
        results: List of InsertionResult objects
        new_lat, new_lon: New HMB coordinates
        weights: Optional custom scoring weights

    Returns:
        Sorted list of InsertionResult objects (best first)
    """
    if weights is None:
        weights = DEFAULT_WEIGHTS

    # Collect raw values for normalization
    extra_kms = [r.extra_km for r in results]
    total_kms = [r.new_total_km for r in results]
    centroid_dists = []
    uti_headrooms = []

    for r in results:
        centroid = calculate_route_centroid(r.route)
        dist = road_distance(new_lat, new_lon, centroid[0], centroid[1])
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


# ─── Main Recommendation Function ───────────────────────────────────────────

def haversine_recommendation(
    lat: float,
    lon: float,
    weights: Optional[dict] = None,
    print_output: bool = True,
) -> dict:
    """
    Recommend the best route for a new HMB based on its GPS coordinates.

    Uses minimum insertion cost algorithm with multi-factor scoring to
    determine the optimal route assignment at Uthangarai Chilling Center.

    Args:
        lat: Latitude of the new HMB
        lon: Longitude of the new HMB
        weights: Optional custom scoring weights dict. Keys:
                 'extra_distance', 'total_route_km', 'centroid_proximity', 'uti_headroom'
                 Values must sum to 1.0
        print_output: If True, prints formatted recommendation to stdout

    Returns:
        dict with keys:
            - 'recommended': InsertionResult for the best route
            - 'all_results': List of all InsertionResult objects, ranked
            - 'input': {'lat': float, 'lon': float}
            - 'cc_location': {'lat': float, 'lon': float}
            - 'config': dict of configuration values used
    """
    # Validate inputs
    if not (-90 <= lat <= 90 and -180 <= lon <= 180):
        raise ValueError(f"Invalid coordinates: ({lat}, {lon})")

    if weights is not None:
        total = sum(weights.values())
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")

    # Load route data
    routes = load_route_data()
    if not routes:
        raise RuntimeError("No route data could be loaded. Check CSV files.")

    # Calculate insertion cost for each route
    results = []
    for route in routes:
        result = calculate_insertion_cost(route, lat, lon)
        if result:
            results.append(result)

    # Score and rank
    ranked = score_routes(results, lat, lon, weights)

    # Build return value
    recommended = ranked[0] if ranked else None
    output = {
        "recommended": recommended,
        "all_results": ranked,
        "input": {"lat": lat, "lon": lon},
        "cc_location": {"lat": CC_LAT, "lon": CC_LON},
        "config": {
            "road_factor": ROAD_FACTOR,
            "max_route_hours": MAX_ROUTE_HOURS,
            "avg_speed_kmh": AVG_SPEED_KMH,
            "avg_stop_time_min": AVG_STOP_TIME_HOURS * 60,
            "weights": weights or DEFAULT_WEIGHTS,
        },
    }

    # Print formatted output
    if print_output and recommended:
        _print_recommendation(output)

    return output


def _print_recommendation(output: dict):
    """Print a formatted recommendation to stdout."""
    rec = output["recommended"]
    results = output["all_results"]
    inp = output["input"]
    config = output["config"]

    print()
    print("═" * 65)
    print("  HMB ROUTE ASSIGNMENT — Uthangarai Chilling Center (1142)")
    print("═" * 65)
    print()
    print(f"  📍 New HMB Location: ({inp['lat']:.6f}, {inp['lon']:.6f})")
    print(f"  🏭 Chilling Center:  ({CC_LAT}, {CC_LON})")
    print(f"  📏 Distance to CC:   {road_distance(inp['lat'], inp['lon'], CC_LAT, CC_LON):.1f} KM (est. road)")
    print()

    if rec.is_feasible:
        print(f"  ✅ RECOMMENDED: Route {rec.route.code} — {rec.route.name}")
    else:
        print(f"  ⚠️  BEST AVAILABLE: Route {rec.route.code} — {rec.route.name}")
        print(f"      ⚠️  Warning: {rec.infeasibility_reason}")

    print(f"     Insert between: {rec.prev_stop_name} → {rec.next_stop_name}")
    print(f"     Position: #{rec.position + 1} in sequence")
    print(f"     Extra distance: +{rec.extra_km:.1f} KM")
    print(f"     New route total: {rec.new_total_km:.1f} KM")
    print(f"     Estimated time: {rec.estimated_time_hours:.1f} hours")
    print(f"     Vehicle: {rec.route.vehicle_type} ({rec.route.transporter})")
    print(f"     Current milk: {rec.route.current_milk_qty:.0f}/{rec.route.capacity} ltrs "
          f"(UTI: {rec.route.uti_percent:.1f}%)")
    print(f"     Score: {rec.score:.2f} / 1.00")
    print()

    # All routes ranked
    print("  ─── All Routes Ranked ─────────────────────────────────────")
    for i, r in enumerate(results):
        feasible_marker = "✅" if r.is_feasible else "❌"
        print(f"  {feasible_marker} #{i+1}  {r.route.code}  {r.route.name:<18s}"
              f"  +{r.extra_km:>6.1f} KM"
              f"  Time: {r.estimated_time_hours:.1f}h"
              f"  Score: {r.score:.2f}")

    print()
    print(f"  ─── Configuration ─────────────────────────────────────────")
    print(f"  Road factor: {config['road_factor']}x | "
          f"Max time: {config['max_route_hours']}h | "
          f"Avg speed: {config['avg_speed_kmh']} km/h | "
          f"Stop time: {config['avg_stop_time_min']:.0f} min")
    w = config["weights"]
    print(f"  Weights: distance={w['extra_distance']}, "
          f"total_km={w['total_route_km']}, "
          f"centroid={w['centroid_proximity']}, "
          f"uti={w['uti_headroom']}")
    print("═" * 65)
    print()


# ─── CLI Entry Point ────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="HMB Route Assignment Optimizer for Uthangarai CC"
    )
    parser.add_argument("--lat", type=float, required=True, help="Latitude of new HMB")
    parser.add_argument("--lon", type=float, required=True, help="Longitude of new HMB")
    parser.add_argument("--road-factor", type=float, default=ROAD_FACTOR,
                        help=f"Road distance multiplier (default: {ROAD_FACTOR})")
    args = parser.parse_args()

    # Allow overriding road factor from CLI
    ROAD_FACTOR = args.road_factor

    haversine_recommendation(args.lat, args.lon)
