from __future__ import annotations

import sys
from dataclasses import dataclass
from functools import lru_cache
from math import sqrt
from pathlib import Path

import pandas as pd

from src.integration.scats_network import (
    ScatsEdge,
    ScatsSite,
    build_realistic_scats_edges,
    haversine_km,
    load_scats_sites,
    roads_from_location,
)
from src.services.prediction import predict_traffic_flow


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOCATION_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed_location" / "cleaned_scats_long.csv"
SEARCH_DIR = PROJECT_ROOT / "Search"
if str(SEARCH_DIR) not in sys.path:
    sys.path.insert(0, str(SEARCH_DIR))

from algorithms.astar import astar
from graph import Graph


INTERSECTION_DELAY_MINUTES = 0.5
DEFAULT_SPEED_LIMIT_KMH = 60.0


@dataclass(frozen=True)
class RouteStep:
    from_site: int
    to_site: int
    distance_km: float
    predicted_flow_hour: float
    speed_kmh: float
    travel_time_minutes: float
    from_location: str | None = None


@dataclass(frozen=True)
class RouteResult:
    path: list[int]
    total_minutes: float
    steps: list[RouteStep]
    nodes_created: int = 0


def speed_from_flow(flow_per_hour: float, speed_limit_kmh: float = DEFAULT_SPEED_LIMIT_KMH) -> float:
    flow = max(0.0, float(flow_per_hour))
    capped_flow_at_speed_limit = 351.0
    if flow <= capped_flow_at_speed_limit:
        return speed_limit_kmh

    # flow = -1.4648375 * speed^2 + 93.75 * speed. Use the under-capacity
    # branch, which is the assignment assumption for each network segment.
    a = 1.4648375
    b = -93.75
    c = flow
    discriminant = max(0.0, b * b - 4 * a * c)
    speed = (-b + sqrt(discriminant)) / (2 * a)
    return max(5.0, min(speed_limit_kmh, speed))


@lru_cache(maxsize=1)
def load_site_locations(
    processed_path: Path = DEFAULT_LOCATION_PROCESSED_PATH,
) -> dict[int, tuple[str, ...]]:
    if not processed_path.exists():
        return {}

    df = pd.read_csv(processed_path, usecols=["scats_site", "location"])
    df["location"] = df["location"].fillna("").astype(str)
    df = df.drop_duplicates(["scats_site", "location"])
    return {
        int(site): tuple(group["location"].sort_values().tolist())
        for site, group in df.groupby("scats_site")
    }


def choose_edge_location(
    from_site: ScatsSite,
    edge: ScatsEdge,
    locations_by_site: dict[int, tuple[str, ...]],
) -> str | None:
    locations = locations_by_site.get(from_site.scats_site, ())
    if not locations:
        return None

    # The data has location labels, not lane-to-edge mappings, so match by road name first.
    for location in locations:
        if edge.road in roads_from_location(location):
            return location

    if from_site.location in locations:
        return from_site.location
    return locations[0]


def estimate_edge(
    from_site: ScatsSite,
    to_site: ScatsSite,
    timestamp: pd.Timestamp,
    model_id: str,
    from_location: str | None = None,
) -> RouteStep:
    prediction = predict_traffic_flow(
        from_site.scats_site,
        timestamp,
        model_id=model_id,
        location=from_location,
    )
    distance_km = haversine_km(from_site, to_site)
    speed_kmh = speed_from_flow(prediction.predicted_flow_hour)
    travel_time_minutes = (distance_km / speed_kmh) * 60.0 + INTERSECTION_DELAY_MINUTES
    return RouteStep(
        from_site=from_site.scats_site,
        to_site=to_site.scats_site,
        distance_km=distance_km,
        predicted_flow_hour=prediction.predicted_flow_hour,
        speed_kmh=speed_kmh,
        travel_time_minutes=travel_time_minutes,
        from_location=prediction.location,
    )


def find_top_k_routes(
    origin: int,
    destination: int,
    timestamp: pd.Timestamp,
    model_id: str,
    k: int = 5,
    neighbours_per_site: int = 4,
    origin_location: str | None = None,
    destination_location: str | None = None,
) -> list[RouteResult]:
    sites = load_scats_sites()
    if origin not in sites:
        raise ValueError(f"Origin SCATS site {origin} has no usable location.")
    if destination not in sites:
        raise ValueError(f"Destination SCATS site {destination} has no usable location.")

    edge_map = build_realistic_scats_edges(sites)
    edges: dict[int, list[int]] = {site_id: [] for site_id in sites}
    edge_lookup: dict[tuple[int, int], ScatsEdge] = {}
    for (from_id, to_id), edge in edge_map.items():
        edges[from_id].append(to_id)
        edges[to_id].append(from_id)
        edge_lookup[(from_id, to_id)] = edge
        edge_lookup[(to_id, from_id)] = edge
    edges = {site_id: sorted(neighbours) for site_id, neighbours in edges.items()}
    locations_by_site = load_site_locations()
    edge_cache: dict[tuple[int, int], RouteStep] = {}

    def get_step(a: int, b: int) -> RouteStep:
        key = (a, b)
        if key not in edge_cache:
            # The user-selected origin location applies only to the first outgoing edge.
            from_location = (
                origin_location
                if a == origin and origin_location
                else choose_edge_location(sites[a], edge_lookup[key], locations_by_site)
            )
            edge_cache[key] = estimate_edge(sites[a], sites[b], timestamp, model_id, from_location)
        return edge_cache[key]

    def build_search_graph(
        origin_id: int,
        banned_edges: set[tuple[int, int]] | None = None,
        banned_nodes: set[int] | None = None,
    ) -> Graph:
        banned_edges = banned_edges or set()
        banned_nodes = banned_nodes or set()
        graph = Graph()
        for site_id, site in sites.items():
            if site_id in banned_nodes:
                continue
            graph.add_node(site_id, site.longitude, site.latitude)

        for from_id, neighbours in edges.items():
            if from_id in banned_nodes:
                continue
            for to_id in neighbours:
                if to_id in banned_nodes or (from_id, to_id) in banned_edges:
                    continue
                step = get_step(from_id, to_id)
                graph.add_edge(from_id, to_id, step.travel_time_minutes)

        graph.origin = graph.nodes[origin_id]
        graph.destinations = [graph.nodes[destination]]
        return graph

    def astar_path(
        origin_id: int,
        banned_edges: set[tuple[int, int]] | None = None,
        banned_nodes: set[int] | None = None,
    ) -> tuple[list[int], int] | None:
        graph = build_search_graph(origin_id, banned_edges, banned_nodes)
        goal, nodes_created, path = astar(graph)
        if goal is None or not path:
            return None
        return path, nodes_created

    def result_from_path(path: list[int], nodes_created: int = 0) -> RouteResult:
        steps = [get_step(path[index], path[index + 1]) for index in range(len(path) - 1)]
        total_minutes = sum(step.travel_time_minutes for step in steps)
        return RouteResult(
            path=path,
            total_minutes=total_minutes,
            steps=steps,
            nodes_created=nodes_created,
        )

    first_path = astar_path(origin)
    if first_path is None:
        return []

    path, nodes_created = first_path
    results = [result_from_path(path, nodes_created)]
    seen_paths = {tuple(path)}
    candidates: dict[tuple[int, ...], RouteResult] = {}

    while len(results) < k:
        previous_route = results[-1]
        for spur_index in range(len(previous_route.path) - 1):
            root_path = previous_route.path[: spur_index + 1]
            spur_node = root_path[-1]
            banned_edges: set[tuple[int, int]] = set()
            banned_nodes = set(root_path[:-1])

            for route in results:
                if route.path[: spur_index + 1] == root_path and len(route.path) > spur_index + 1:
                    banned_edges.add((route.path[spur_index], route.path[spur_index + 1]))

            spur_result = astar_path(spur_node, banned_edges=banned_edges, banned_nodes=banned_nodes)
            if spur_result is None:
                continue

            spur_path, spur_nodes_created = spur_result
            combined_path = root_path[:-1] + spur_path
            path_key = tuple(combined_path)
            if path_key in seen_paths:
                continue
            candidates[path_key] = result_from_path(combined_path, spur_nodes_created)

        if not candidates:
            break

        best_key, best = min(candidates.items(), key=lambda item: item[1].total_minutes)
        del candidates[best_key]
        if best_key in seen_paths:
            continue
        results.append(best)
        seen_paths.add(best_key)

    return results
