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
    print(f"  [>] New HMB Location: ({inp['lat']:.6f}, {inp['lon']:.6f})")
    print(f"  [*] Chilling Center:  ({CC_LAT}, {CC_LON})")
    print(f"  [~] Distance to CC:   {road_distance(inp['lat'], inp['lon'], CC_LAT, CC_LON):.1f} KM (est. road)")
    print()

    if rec.is_feasible:
        print(f"  [OK] RECOMMENDED: Route {rec.route.code} -- {rec.route.name}")
    else:
        print(f"  [!] BEST AVAILABLE: Route {rec.route.code} -- {rec.route.name}")
        print(f"       Warning: {rec.infeasibility_reason}")

    print(f"     Insert between: {rec.prev_stop_name} -> {rec.next_stop_name}")
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
        feasible_marker = "[OK]" if r.is_feasible else "[X] "
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
