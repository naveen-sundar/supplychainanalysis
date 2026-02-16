"""
Test suite for HMB Route Assignment Optimizer.

Tests the Haversine distance calculation, data loading, insertion cost
algorithm, and the overall recommendation pipeline.
"""

import math
import pytest
from route_optimizer import (
    haversine,
    road_distance,
    load_route_data,
    load_route_summary,
    calculate_route_distance,
    calculate_insertion_cost,
    estimate_route_time,
    calculate_route_centroid,
    score_routes,
    haversine_recommendation,
    CC_LAT,
    CC_LON,
    ROAD_FACTOR,
)


# ─── Haversine Tests ────────────────────────────────────────────────────────

class TestHaversine:
    """Tests for the Haversine distance formula."""

    def test_same_point_returns_zero(self):
        """Distance from a point to itself should be 0."""
        assert haversine(12.0, 78.0, 12.0, 78.0) == 0.0

    def test_known_distance_cc_to_chandrapuram(self):
        """
        CC (12.308573, 78.535901) to Chandrapuram (12.511537, 78.479387).
        Expected straight-line: ~23 KM. Master data says 32.3 KM road distance.
        Haversine should give ~23 KM (straight-line).
        """
        dist = haversine(CC_LAT, CC_LON, 12.511537, 78.479387)
        assert 20 < dist < 30, f"Expected ~23 KM, got {dist:.1f} KM"

    def test_road_distance_applies_factor(self):
        """Road distance should be Haversine × ROAD_FACTOR."""
        h_dist = haversine(12.0, 78.0, 12.1, 78.1)
        r_dist = road_distance(12.0, 78.0, 12.1, 78.1)
        assert abs(r_dist - h_dist * ROAD_FACTOR) < 0.001

    def test_symmetry(self):
        """Distance from A→B should equal B→A."""
        d1 = haversine(12.3, 78.5, 12.5, 78.4)
        d2 = haversine(12.5, 78.4, 12.3, 78.5)
        assert abs(d1 - d2) < 0.001

    def test_short_distance(self):
        """Two points ~1 KM apart should give reasonable result."""
        # ~0.009 degrees latitude ≈ 1 KM
        dist = haversine(12.0, 78.0, 12.009, 78.0)
        assert 0.8 < dist < 1.2, f"Expected ~1 KM, got {dist:.2f} KM"


# ─── Data Loading Tests ─────────────────────────────────────────────────────

class TestDataLoading:
    """Tests for CSV data loading functions."""

    def test_load_route_summary_returns_7_routes(self):
        """Uthangarai has exactly 7 routes."""
        summary = load_route_summary()
        assert len(summary) == 7, f"Expected 7 routes, got {len(summary)}"

    def test_load_route_summary_keys(self):
        """All expected route codes should be present."""
        summary = load_route_summary()
        expected = {"M10126", "M10134", "M10135", "M11515", "M11830", "M13169", "M13483"}
        assert set(summary.keys()) == expected

    def test_load_route_data_returns_7_routes(self):
        """Should load 7 routes with HMB sequences."""
        routes = load_route_data()
        assert len(routes) == 7

    def test_routes_have_hmbs(self):
        """Every route should have at least 1 HMB."""
        routes = load_route_data()
        for route in routes:
            assert len(route.hmbs) > 0, f"Route {route.code} has no HMBs"

    def test_bargoor_has_10_hmbs(self):
        """Bargoor route (M10126) has 10 collection points."""
        routes = load_route_data()
        bargoor = next(r for r in routes if r.code == "M10126")
        assert len(bargoor.hmbs) == 10, f"Expected 10 HMBs, got {len(bargoor.hmbs)}"

    def test_hmb_coordinates_are_valid(self):
        """All HMB coordinates should be in valid range for Tamil Nadu / India."""
        routes = load_route_data()
        for route in routes:
            for hmb in route.hmbs:
                assert 10 < hmb.lat < 14, f"Bad lat {hmb.lat} for {hmb.name}"
                assert 76 < hmb.lon < 80, f"Bad lon {hmb.lon} for {hmb.name}"

    def test_hmbs_sorted_by_sequence(self):
        """HMBs in each route should be sorted by sequence number."""
        routes = load_route_data()
        for route in routes:
            sequences = [h.sequence for h in route.hmbs]
            assert sequences == sorted(sequences), \
                f"Route {route.code} HMBs not sorted: {sequences}"


# ─── Route Calculation Tests ────────────────────────────────────────────────

class TestRouteCalculations:
    """Tests for route distance and time calculations."""

    def test_route_distance_positive(self):
        """All routes should have positive total distance."""
        routes = load_route_data()
        for route in routes:
            dist = calculate_route_distance(route)
            assert dist > 0, f"Route {route.code} has non-positive distance: {dist}"

    def test_route_distances_reasonable(self):
        """Calculated route distances should be in realistic range (20-150 KM)."""
        routes = load_route_data()
        for route in routes:
            dist = calculate_route_distance(route)
            assert 20 < dist < 150, \
                f"Route {route.code} distance {dist:.1f} KM seems unreasonable"

    def test_route_time_estimate(self):
        """Route time should be reasonable (1-8 hours)."""
        time = estimate_route_time(80, 10)  # 80 KM, 10 stops
        assert 1 < time < 8, f"Unreasonable time estimate: {time:.1f}h"

    def test_centroid_in_valid_range(self):
        """Route centroids should be within the operational area."""
        routes = load_route_data()
        for route in routes:
            lat, lon = calculate_route_centroid(route)
            assert 10 < lat < 14 and 76 < lon < 80, \
                f"Route {route.code} centroid ({lat}, {lon}) out of range"


# ─── Insertion Cost Tests ────────────────────────────────────────────────────

class TestInsertionCost:
    """Tests for the minimum insertion cost algorithm."""

    def test_existing_hmb_zero_extra_km(self):
        """Inserting at an existing HMB's location should add ~0 extra KM."""
        routes = load_route_data()
        harur = next(r for r in routes if r.code == "M10135")
        katteri = next(h for h in harur.hmbs if "Katteri" in h.name)

        result = calculate_insertion_cost(harur, katteri.lat, katteri.lon)
        assert result.extra_km < 1.0, \
            f"Inserting at existing HMB location should add ~0 KM, got {result.extra_km}"

    def test_insertion_extra_km_positive(self):
        """Extra KM should be non-negative for any insertion."""
        routes = load_route_data()
        # A point far from any route
        for route in routes:
            result = calculate_insertion_cost(route, 12.0, 78.0)
            assert result.extra_km >= 0, \
                f"Route {route.code}: negative extra KM ({result.extra_km})"

    def test_insertion_returns_valid_result(self):
        """InsertionResult should have all required fields."""
        routes = load_route_data()
        route = routes[0]
        result = calculate_insertion_cost(route, 12.35, 78.55)

        assert result.route == route
        assert result.position >= 0
        assert result.extra_km >= 0
        assert result.new_total_km > 0
        assert result.estimated_time_hours > 0
        assert result.prev_stop_name != ""
        assert result.next_stop_name != ""

    def test_new_total_km_greater_equal_current(self):
        """New total KM should be >= current route distance."""
        routes = load_route_data()
        for route in routes:
            current = calculate_route_distance(route)
            result = calculate_insertion_cost(route, 12.35, 78.55)
            assert result.new_total_km >= current - 0.1, \
                f"Route {route.code}: new total {result.new_total_km} < current {current}"


# ─── Recommendation Tests ───────────────────────────────────────────────────

class TestRecommendation:
    """Tests for the haversine_recommendation function."""

    def test_existing_hmb_katteri_recommends_harur(self):
        """Katteri (12.200672, 78.544356) is on Harur route → should recommend Harur."""
        result = haversine_recommendation(12.200672, 78.544356, print_output=False)
        assert result["recommended"].route.code == "M10135", \
            f"Expected M10135 (Harur), got {result['recommended'].route.code}"

    def test_existing_hmb_chandrapuram_recommends_bargoor(self):
        """Chandrapuram (12.511537, 78.479387) is on Bargoor → should recommend Bargoor."""
        result = haversine_recommendation(12.511537, 78.479387, print_output=False)
        assert result["recommended"].route.code == "M10126", \
            f"Expected M10126 (Bargoor), got {result['recommended'].route.code}"

    def test_existing_hmb_anna_nagar_recommends_manmalai(self):
        """Anna Nagar (12.348228, 78.742339) is on Manmalai → should recommend Manmalai."""
        result = haversine_recommendation(12.348228, 78.742339, print_output=False)
        assert result["recommended"].route.code == "M13483", \
            f"Expected M13483 (Manmalai), got {result['recommended'].route.code}"

    def test_returns_7_results(self):
        """Should evaluate all 7 routes."""
        result = haversine_recommendation(12.35, 78.55, print_output=False)
        assert len(result["all_results"]) == 7

    def test_results_sorted_by_score(self):
        """Results should be sorted by score descending."""
        result = haversine_recommendation(12.35, 78.55, print_output=False)
        scores = [r.score for r in result["all_results"]]
        assert scores == sorted(scores, reverse=True), "Results not sorted by score"

    def test_return_dict_has_expected_keys(self):
        """Return dict should have all expected keys."""
        result = haversine_recommendation(12.35, 78.55, print_output=False)
        assert "recommended" in result
        assert "all_results" in result
        assert "input" in result
        assert "cc_location" in result
        assert "config" in result

    def test_custom_weights(self):
        """Custom weights should be accepted and used."""
        custom_weights = {
            "extra_distance": 0.70,
            "total_route_km": 0.10,
            "centroid_proximity": 0.10,
            "uti_headroom": 0.10,
        }
        result = haversine_recommendation(12.35, 78.55, weights=custom_weights, print_output=False)
        assert result["config"]["weights"] == custom_weights

    def test_invalid_coordinates_raises(self):
        """Invalid coordinates should raise ValueError."""
        with pytest.raises(ValueError):
            haversine_recommendation(200, 78.5, print_output=False)

    def test_invalid_weights_raises(self):
        """Weights that don't sum to 1.0 should raise ValueError."""
        bad_weights = {
            "extra_distance": 0.50,
            "total_route_km": 0.50,
            "centroid_proximity": 0.50,
            "uti_headroom": 0.50,
        }
        with pytest.raises(ValueError):
            haversine_recommendation(12.35, 78.55, weights=bad_weights, print_output=False)

    def test_cc_nearby_prefers_short_route(self):
        """An HMB very close to CC should prefer routes with nearby HMBs."""
        # Point very close to CC
        result = haversine_recommendation(CC_LAT + 0.01, CC_LON + 0.01, print_output=False)
        rec = result["recommended"]
        # Should have low extra KM since it's near the CC (start/end of all routes)
        assert rec.extra_km < 5.0, f"Near-CC HMB adds too much: {rec.extra_km} KM"


# ─── Scoring Tests ──────────────────────────────────────────────────────────

class TestScoring:
    """Tests for the multi-factor scoring system."""

    def test_scores_between_0_and_1(self):
        """All scores should be in [0, 1] range."""
        result = haversine_recommendation(12.35, 78.55, print_output=False)
        for r in result["all_results"]:
            assert 0 <= r.score <= 1.0, f"Score {r.score} out of range for {r.route.code}"

    def test_feasible_routes_rank_above_infeasible(self):
        """Feasible routes should always rank above infeasible ones."""
        result = haversine_recommendation(12.35, 78.55, print_output=False)
        found_infeasible = False
        for r in result["all_results"]:
            if not r.is_feasible:
                found_infeasible = True
            elif found_infeasible:
                pytest.fail("Feasible route found after infeasible route in ranking")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
