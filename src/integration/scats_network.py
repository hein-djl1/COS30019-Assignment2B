from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass
from math import asin, cos, radians, sin, sqrt
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_scats_long.csv"
DEFAULT_EXCLUDED_SITE_IDS = {2825, 2827, 3812, 2846, 2200, 4321}
DEFAULT_WEST_CUTOFF_SITE_ID = 4272
DEFAULT_INCLUDED_SITE_IDS = {4273}


@dataclass(frozen=True)
class ScatsSite:
    scats_site: int
    location: str
    latitude: float
    longitude: float

    @property
    def label(self) -> str:
        return f"{self.scats_site} - {self.location}"


@dataclass(frozen=True)
class ScatsEdge:
    from_site: int
    to_site: int
    kind: str
    road: str
    distance_km: float


def load_scats_sites(
    processed_path: Path = DEFAULT_PROCESSED_PATH,
    *,
    excluded_site_ids: set[int] | None = None,
    west_cutoff_site_id: int | None = DEFAULT_WEST_CUTOFF_SITE_ID,
) -> dict[int, ScatsSite]:
    df = pd.read_csv(processed_path, usecols=["scats_site", "location", "latitude", "longitude"])
    df = df.drop_duplicates("scats_site").copy()
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df = df.dropna(subset=["latitude", "longitude"])
    df = df[(df["latitude"] != 0) & (df["longitude"] != 0)]
    excluded_site_ids = DEFAULT_EXCLUDED_SITE_IDS if excluded_site_ids is None else excluded_site_ids

    if west_cutoff_site_id is not None:
        cutoff_row = df[df["scats_site"] == west_cutoff_site_id]
        if not cutoff_row.empty:
            cutoff_longitude = float(cutoff_row.iloc[0]["longitude"])
            df = df[
                (df["longitude"] >= cutoff_longitude)
                | df["scats_site"].isin(DEFAULT_INCLUDED_SITE_IDS)
            ]

    if excluded_site_ids:
        df = df[~df["scats_site"].isin(excluded_site_ids)]

    return {
        int(row.scats_site): ScatsSite(
            scats_site=int(row.scats_site),
            location=str(row.location),
            latitude=float(row.latitude),
            longitude=float(row.longitude),
        )
        for row in df.itertuples(index=False)
    }


def haversine_km(a: ScatsSite, b: ScatsSite) -> float:
    earth_radius_km = 6371.0
    lat1 = radians(a.latitude)
    lon1 = radians(a.longitude)
    lat2 = radians(b.latitude)
    lon2 = radians(b.longitude)
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    h = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * earth_radius_km * asin(sqrt(h))


def normalise_road_name(value: str) -> str:
    road = value.upper().replace("_", " ").replace(".", "")
    replacements = {
        " ROAD": " RD",
        " STREET": " ST",
        " HIGHWAY": " HWY",
        " FREEWAY": " FWY",
        " GROVE": " GV",
        " PARADE": " PDE",
        " AVENUE": " AVE",
    }
    for old, new in replacements.items():
        road = road.replace(old, new)
    return re.sub(r"\s+", " ", road).strip()


def roads_from_location(location: str) -> list[str]:
    text = location.upper().replace("_", " ")
    parts = re.split(
        r"\s+(?:N|S|E|W|NE|NW|SE|SW)\s+(?:OF)"
        r"|\s+(?:N|S|E|W|NE|NW|SE|SW)\s+BD\s+(?:N|S|E|W|NE|NW|SE|SW)?\s*(?:OF)?",
        text,
        maxsplit=1,
    )
    roads = [normalise_road_name(part) for part in parts if part.strip()]
    return roads[:2]


def _add_edge(
    edge_map: dict[tuple[int, int], ScatsEdge],
    sites: dict[int, ScatsSite],
    from_site: int,
    to_site: int,
    *,
    kind: str,
    road: str,
) -> None:
    if from_site == to_site:
        return
    key = tuple(sorted((from_site, to_site)))
    distance = haversine_km(sites[key[0]], sites[key[1]])
    existing = edge_map.get(key)
    if existing is not None and existing.kind != "connector":
        return
    edge_map[key] = ScatsEdge(
        from_site=key[0],
        to_site=key[1],
        kind=kind,
        road=road,
        distance_km=distance,
    )


def _connected_components(
    sites: dict[int, ScatsSite],
    edge_map: dict[tuple[int, int], ScatsEdge],
) -> list[set[int]]:
    adjacency: dict[int, set[int]] = {site_id: set() for site_id in sites}
    for a, b in edge_map:
        adjacency[a].add(b)
        adjacency[b].add(a)

    seen: set[int] = set()
    components: list[set[int]] = []
    for site_id in sites:
        if site_id in seen:
            continue
        stack = [site_id]
        seen.add(site_id)
        component: set[int] = set()
        while stack:
            current = stack.pop()
            component.add(current)
            for neighbour in adjacency[current]:
                if neighbour not in seen:
                    seen.add(neighbour)
                    stack.append(neighbour)
        components.append(component)
    return components


def build_realistic_scats_edges(
    sites: dict[int, ScatsSite],
    *,
    max_same_road_gap_km: float = 3.5,
    max_cross_road_gap_km: float = 1.25,
    max_local_link_km: float = 0.9,
    connect_components: bool = True,
) -> dict[int, ScatsEdge]:
    road_sites: dict[str, list[int]] = defaultdict(list)
    site_roads = {site_id: roads_from_location(site.location) for site_id, site in sites.items()}
    for site_id, roads in site_roads.items():
        for road in roads:
            road_sites[road].append(site_id)

    edge_map: dict[tuple[int, int], ScatsEdge] = {}
    for road, site_ids in road_sites.items():
        if len(site_ids) < 2:
            continue
        lat_range = max(sites[site_id].latitude for site_id in site_ids) - min(
            sites[site_id].latitude for site_id in site_ids
        )
        lon_range = max(sites[site_id].longitude for site_id in site_ids) - min(
            sites[site_id].longitude for site_id in site_ids
        )
        sort_key = (
            (lambda site_id: sites[site_id].latitude)
            if lat_range >= lon_range
            else (lambda site_id: sites[site_id].longitude)
        )
        ordered = sorted(site_ids, key=sort_key)
        for from_site, to_site in zip(ordered, ordered[1:]):
            distance = haversine_km(sites[from_site], sites[to_site])
            if distance <= max_same_road_gap_km:
                _add_edge(edge_map, sites, from_site, to_site, kind="road", road=road)

    site_ids = list(sites)
    for index, from_site in enumerate(site_ids):
        from_roads = set(site_roads[from_site])
        for to_site in site_ids[index + 1 :]:
            shared_roads = from_roads & set(site_roads[to_site])
            if not shared_roads:
                continue
            distance = haversine_km(sites[from_site], sites[to_site])
            if distance <= max_cross_road_gap_km:
                _add_edge(
                    edge_map,
                    sites,
                    from_site,
                    to_site,
                    kind="nearby_intersection",
                    road=sorted(shared_roads)[0],
                )

    for index, from_site in enumerate(site_ids):
        for to_site in site_ids[index + 1 :]:
            if tuple(sorted((from_site, to_site))) in edge_map:
                continue
            distance = haversine_km(sites[from_site], sites[to_site])
            if distance <= max_local_link_km:
                _add_edge(
                    edge_map,
                    sites,
                    from_site,
                    to_site,
                    kind="local_link",
                    road="nearby Boroondara street link",
                )

    if connect_components:
        while True:
            components = _connected_components(sites, edge_map)
            if len(components) <= 1:
                break

            best_pair: tuple[float, int, int] | None = None
            for left_index, left_component in enumerate(components):
                for right_component in components[left_index + 1 :]:
                    for left_site in left_component:
                        for right_site in right_component:
                            distance = haversine_km(sites[left_site], sites[right_site])
                            if best_pair is None or distance < best_pair[0]:
                                best_pair = (distance, left_site, right_site)

            if best_pair is None:
                break
            _, from_site, to_site = best_pair
            _add_edge(
                edge_map,
                sites,
                from_site,
                to_site,
                kind="connector",
                road="closest available SCATS connector",
            )

    return edge_map


def build_realistic_scats_adjacency(
    sites: dict[int, ScatsSite],
    **kwargs: object,
) -> dict[int, list[int]]:
    edge_map = build_realistic_scats_edges(sites, **kwargs)
    edges: dict[int, set[int]] = {site_id: set() for site_id in sites}
    for from_site, to_site in edge_map:
        edges[from_site].add(to_site)
        edges[to_site].add(from_site)
    return {site_id: sorted(neighbours) for site_id, neighbours in edges.items()}


def build_nearest_neighbour_edges(
    sites: dict[int, ScatsSite],
    neighbours_per_site: int = 4,
) -> dict[int, list[int]]:
    return build_realistic_scats_adjacency(sites)
