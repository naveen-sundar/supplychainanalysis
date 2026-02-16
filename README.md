# HMB Route Assignment Optimizer

Assigns new HMBs (Hire Milk Booths) to the most cost-effective route at Uthangarai Chilling Center (Plant 1142) using GPS coordinates.

## Algorithm

**Minimum Insertion Cost with Multi-Factor Scoring**

1. For each of the 7 existing routes, finds the optimal position to insert the new HMB using the **Haversine formula** (great-circle distance with a 1.3x road factor).
2. Scores routes on a weighted combination of:
   - Extra distance added (50%)
   - Total route length (20%)
   - Proximity to route centroid (15%)
   - Vehicle capacity headroom (15%)
3. Applies constraints: vehicle capacity and milk freshness (max 6h route time).

## Usage

```bash
python3 script/route_optimizer.py --lat 12.35 --lon 78.55
```

```python
from script.route_optimizer import haversine_recommendation

result = haversine_recommendation(12.35, 78.55)
```

### Custom Weights

```python
result = haversine_recommendation(12.35, 78.55, weights={
    "extra_distance": 0.70,
    "total_route_km": 0.10,
    "centroid_proximity": 0.10,
    "uti_headroom": 0.10,
})
```

## Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROAD_FACTOR` | 1.3 | Multiplier on Haversine for road distance |
| `MAX_ROUTE_HOURS` | 6.0 | Max route duration (milk freshness) |
| `AVG_SPEED_KMH` | 25.0 | Average vehicle speed |
| `AVG_STOP_TIME_HOURS` | 10 min | Collection time per HMB |

All configurable at the top of `script/route_optimizer.py`.

## Data

CSV files in `csv_files/`:
- `HMB Details_Master Data.csv` — HMB sequences, GPS coordinates per route
- `HMB Details_Summary.csv` — Route capacities, milk volumes, vehicle types

## Tests

```bash
python3 -m pytest script/test_route_optimizer.py -v
```

32 tests covering Haversine math, data loading, insertion cost, scoring, and end-to-end recommendations.
