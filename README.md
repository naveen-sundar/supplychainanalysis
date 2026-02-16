# HMB Route Assignment Optimizer V2

Assigns new HMBs (Hire Milk Booths) to the most cost-effective route at Uthangarai Chilling Center (Plant 1142) using GPS coordinates and expected milk volume.

## Algorithm

**Minimum Insertion Cost + 2-Opt Improvement + Multi-Factor Scoring**

1. **Capacity Pre-Filter**: Rejects routes that cannot fit the new HMB's expected milk volume.
2. **Insertion Cost**: Finds the optimal position to insert the new HMB in each eligible route.
3. **2-Opt Improvement**: Reorders stops to minimize total route distance after insertion.
4. **Scoring**: Ranks routes on a weighted combination of:
   - Extra distance added (40%)
   - Total route length (20%)
   - Proximity to route centroid (15%)
   - Vehicle capacity headroom (25%)
5. **Constraints**: Vehicle capacity (hard pre-filter) and milk freshness (max 4h route time).

## Distance Modes

### Haversine (Offline, Instant)
Uses GPS straight-line distance × 1.3 road factor. Fast and works offline.

### OSRM (Real Roads, ~5 min)
Queries OpenStreetMap routing API for actual road distances. More accurate but requires network access.

## Usage

### CLI

```bash
# Haversine mode (instant)
python3 script/route_optimizer.py --lat 12.35 --lon 78.55 --milk-qty 100

# OSRM mode (real road distances)
python3 script/route_optimizer.py --lat 12.35 --lon 78.55 --milk-qty 100 --mode osrm

# Custom road factor (Haversine only)
python3 script/route_optimizer.py --lat 12.35 --lon 78.55 --milk-qty 100 --road-factor 1.5
```

### Python API

```python
from script.route_optimizer import recommend_route

# Haversine mode
result = recommend_route(12.35, 78.55, expected_milk_qty=100, mode="haversine")

# OSRM mode
result = recommend_route(12.35, 78.55, expected_milk_qty=100, mode="osrm")

# Custom weights
result = recommend_route(12.35, 78.55, expected_milk_qty=100, weights={
    "extra_distance": 0.50,
    "total_route_km": 0.20,
    "centroid_proximity": 0.15,
    "uti_headroom": 0.15,
})
```

### Legacy API (V1 compatibility)

```python
from script.route_optimizer import haversine_recommendation

result = haversine_recommendation(12.35, 78.55)
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROAD_FACTOR` | 1.3 | Multiplier on Haversine for road distance (Haversine mode only) |
| `MAX_ROUTE_HOURS` | 4.0 | Max route duration for milk freshness |
| `AVG_SPEED_KMH` | 40.0 | Average vehicle speed |
| `AVG_STOP_TIME_HOURS` | 10 min | Collection time per HMB |

All configurable at the top of `script/route_optimizer.py`.

## Output

The optimizer provides:
- **Recommended route** with insertion position, distance added, and estimated time
- **Capacity analysis** showing milk load after insertion
- **2-opt savings** (how much distance was saved by reordering)
- **Optimized stop order** showing the full route after insertion and 2-opt
- **Ranked list** of all routes with scores and feasibility status
- **Rejected routes** (if any exceed capacity)

## Data

CSV files in `csv_files/`:
- `HMB Details_Master Data.csv` — HMB sequences, GPS coordinates per route
- `HMB Details_Summary.csv` — Route capacities, milk volumes, vehicle types

## Tests

```bash
python3 -m pytest script/test_route_optimizer.py -v
```

68 tests covering:
- Haversine formula and distance calculations
- Data loading and coordinate parsing
- Capacity pre-filter logic
- Insertion cost algorithm
- 2-Opt route improvement
- Multi-factor scoring and ranking
- End-to-end recommendations (Haversine mode)
- Input validation

## What's New in V2

- **Dual distance modes**: Haversine (offline) and OSRM (real roads)
- **2-Opt improvement**: Post-insertion route reordering (saves 0-16 KM per route)
- **Capacity pre-filter**: Hard rejection of overloaded routes before scoring
- **Expected milk volume**: Required input for accurate capacity checks
- **Updated constraints**: 4h max route time (down from 6h), 40 km/h speed (up from 25 km/h)
- **Enhanced output**: Shows 2-opt savings, optimized stop order, capacity details
