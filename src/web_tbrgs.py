from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pandas as pd

from src.integration.routing import find_top_k_routes
from src.integration.scats_network import build_realistic_scats_edges, load_scats_sites
from src.services.prediction import registry


HOST = "127.0.0.1"
PORT = 8000
PROJECT_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = PROJECT_ROOT / "config" / "tbrgs_config.json"
LOCATION_PROCESSED_PATH = PROJECT_ROOT / "data" / "processed_location" / "cleaned_scats_long.csv"
PROCESSED_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_scats_long.csv"


def load_config() -> dict[str, object]:
    defaults: dict[str, object] = {
        "host": HOST,
        "port": PORT,
        "default_model": "xgboost",
        "default_date": "2006-10-23",
        "default_time": "08:00",
        "default_top_k": 5,
    }
    if CONFIG_PATH.exists():
        defaults.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    return defaults


def load_scats_locations() -> list[dict[str, object]]:
    sites_by_id = load_scats_sites()
    path = LOCATION_PROCESSED_PATH if LOCATION_PROCESSED_PATH.exists() else PROCESSED_PATH
    df = pd.read_csv(path, usecols=["scats_site", "location", "latitude", "longitude"])
    df["scats_site"] = pd.to_numeric(df["scats_site"], errors="coerce")
    df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
    df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
    df["location"] = df["location"].fillna("").astype(str)
    df = df.dropna(subset=["scats_site"])
    df["scats_site"] = df["scats_site"].astype(int)
    df = df[df["scats_site"].isin(sites_by_id)]
    df = df.drop_duplicates(["scats_site", "location"]).sort_values(["scats_site", "location"])

    locations: list[dict[str, object]] = []
    for row in df.itertuples(index=False):
        site = sites_by_id[int(row.scats_site)]
        locations.append(
            {
                "scats_site": int(row.scats_site),
                "location": str(row.location),
                "latitude": site.latitude,
                "longitude": site.longitude,
            }
        )
    return locations


HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>TBRGS</title>
  <style>
    :root {
      color-scheme: light;
      font-family: Arial, Helvetica, sans-serif;
      background: #f5f7f8;
      color: #172026;
    }
    body {
      margin: 0;
      min-height: 100vh;
    }
    header {
      background: #16313a;
      color: #ffffff;
      padding: 18px 24px;
    }
    h1 {
      margin: 0;
      font-size: 24px;
      font-weight: 700;
      letter-spacing: 0;
    }
    main {
      max-width: 1180px;
      margin: 0 auto;
      padding: 22px;
    }
    .controls {
      display: grid;
      grid-template-columns: repeat(8, minmax(0, 1fr));
      gap: 14px;
      align-items: end;
      background: #ffffff;
      border: 1px solid #d9e0e4;
      border-radius: 8px;
      padding: 16px;
    }
    label {
      display: grid;
      gap: 6px;
      font-size: 13px;
      color: #49575f;
    }
    select,
    input,
    button {
      box-sizing: border-box;
      min-height: 38px;
      border: 1px solid #b8c4ca;
      border-radius: 6px;
      font: inherit;
      padding: 8px 10px;
      background: #ffffff;
      color: #172026;
      width: 100%;
    }
    button {
      border-color: #0d5b66;
      background: #0d5b66;
      color: #ffffff;
      cursor: pointer;
      font-weight: 700;
    }
    button:disabled {
      opacity: 0.65;
      cursor: wait;
    }
    .wide {
      grid-column: span 2;
    }
    .results {
      margin-top: 18px;
    }
    .workspace {
      display: grid;
      grid-template-columns: minmax(360px, 1fr) minmax(420px, 1.25fr);
      gap: 18px;
      margin-top: 18px;
      align-items: start;
    }
    .graph-panel {
      background: #ffffff;
      border: 1px solid #d9e0e4;
      border-radius: 8px;
      overflow: hidden;
    }
    .graph-header {
      display: flex;
      justify-content: space-between;
      gap: 12px;
      padding: 10px 12px;
      background: #edf3f5;
      color: #26363d;
      font-size: 14px;
      font-weight: 700;
    }
    #graph {
      display: block;
      width: 100%;
      height: 560px;
      background: #f9fbfb;
    }
    .graph-edge {
      stroke: #bfccd1;
      stroke-width: 1.1;
    }
    .graph-edge.connector {
      stroke: #d7b07b;
      stroke-dasharray: 5 5;
    }
    .graph-edge.nearby_intersection {
      stroke: #98b5bf;
    }
    .graph-edge.local_link {
      stroke: #a8b2b6;
      stroke-dasharray: 2 4;
    }
    .graph-edge.path {
      stroke: #d66f27;
      stroke-width: 4;
      stroke-linecap: round;
    }
    .graph-node {
      fill: #ffffff;
      stroke: #5b6c73;
      stroke-width: 1.4;
    }
    .graph-node.path {
      fill: #fff3dc;
      stroke: #d66f27;
      stroke-width: 2.5;
    }
    .graph-node.origin {
      fill: #dff2e3;
      stroke: #1c7f45;
    }
    .graph-node.destination {
      fill: #e2edff;
      stroke: #2563ad;
    }
    .graph-label {
      fill: #172026;
      font-size: 11px;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 3px;
      stroke-linejoin: round;
    }
    .graph-cost {
      fill: #8a3f13;
      font-size: 12px;
      font-weight: 700;
      paint-order: stroke;
      stroke: #ffffff;
      stroke-width: 4px;
      stroke-linejoin: round;
    }
    table {
      width: 100%;
      border-collapse: collapse;
      background: #ffffff;
      border: 1px solid #d9e0e4;
      border-radius: 8px;
      overflow: hidden;
    }
    th,
    td {
      border-bottom: 1px solid #e6ecef;
      padding: 10px;
      text-align: left;
      vertical-align: top;
      font-size: 14px;
    }
    th {
      background: #edf3f5;
      color: #26363d;
      font-weight: 700;
    }
    tr:last-child td {
      border-bottom: 0;
    }
    .status {
      margin-top: 12px;
      color: #49575f;
      min-height: 20px;
    }
    .error {
      color: #9d2424;
      font-weight: 700;
    }
    @media (max-width: 820px) {
      .controls {
        grid-template-columns: 1fr;
      }
      .wide {
        grid-column: span 1;
      }
      main {
        padding: 14px;
      }
      .workspace {
        grid-template-columns: 1fr;
      }
      #graph {
        height: 420px;
      }
    }
  </style>
</head>
<body>
  <header><h1>Traffic-Based Route Guidance System</h1></header>
  <main>
    <section class="controls">
      <label class="wide">Model<select id="model"></select></label>
      <label>Origin site<select id="originSite"></select></label>
      <label class="wide">Origin location<select id="originLocation"></select></label>
      <label>Destination site<select id="destinationSite"></select></label>
      <label class="wide">Destination location<select id="destinationLocation"></select></label>
      <label>Date<input id="date" type="date" value="2006-10-23"></label>
      <label>Time<input id="time" type="time" step="900" value="08:00"></label>
      <label>Top routes<input id="k" type="number" min="1" max="5" value="5"></label>
      <button id="find">Find Routes</button>
    </section>
    <div id="status" class="status">Loading SCATS sites...</div>
    <section class="workspace">
      <div class="graph-panel">
        <div class="graph-header">
          <span>SCATS Graph</span>
          <span id="graphSummary">Shortest path appears after search</span>
        </div>
        <svg id="graph" viewBox="0 0 900 560" role="img" aria-label="SCATS graph with shortest path"></svg>
      </div>
      <section class="results">
        <table>
          <thead>
            <tr>
              <th>#</th>
              <th>Route</th>
              <th>Total time</th>
            </tr>
          </thead>
          <tbody id="routes"></tbody>
        </table>
      </section>
    </section>
  </main>
  <script>
    const modelSelect = document.querySelector("#model");
    const originSiteSelect = document.querySelector("#originSite");
    const originLocationSelect = document.querySelector("#originLocation");
    const destinationSiteSelect = document.querySelector("#destinationSite");
    const destinationLocationSelect = document.querySelector("#destinationLocation");
    const status = document.querySelector("#status");
    const rows = document.querySelector("#routes");
    const findButton = document.querySelector("#find");
    const graphSvg = document.querySelector("#graph");
    const graphSummary = document.querySelector("#graphSummary");
    let graphData = { nodes: [], edges: [] };
    let locationsBySite = new Map();
    let hasRouteResults = false;

    function option(value, text) {
      const el = document.createElement("option");
      el.value = value;
      el.textContent = text;
      return el;
    }

    function selectedLocation(siteSelect, locationSelect) {
      return {
        site: siteSelect.value,
        location: locationSelect.value
      };
    }

    function populateLocations(siteSelect, locationSelect, preferredLocation = null) {
      locationSelect.replaceChildren();
      const locations = locationsBySite.get(siteSelect.value) || [];
      locations.forEach(location => {
        locationSelect.appendChild(option(location.location, location.location));
      });
      if (preferredLocation && locations.some(location => location.location === preferredLocation)) {
        locationSelect.value = preferredLocation;
      }
    }

    function markRouteResultsStale() {
      if (!hasRouteResults) return;
      hasRouteResults = false;
      rows.replaceChildren();
      status.className = "status";
      // Route rows are only valid for the exact controls used in the last search.
      status.textContent = "Inputs changed. Click Find Routes to recalculate.";
      renderGraph();
    }

    async function loadDefaults() {
      const [models, locations, graph] = await Promise.all([
        fetch("/api/models").then(r => r.json()),
        fetch("/api/locations").then(r => r.json()),
        fetch("/api/graph").then(r => r.json())
      ]);
      graphData = graph;
      const config = await fetch("/api/config").then(r => r.json());
      models.forEach(model => modelSelect.appendChild(option(model.id, model.label)));
      modelSelect.value = config.default_model;

      locationsBySite = new Map();
      locations.forEach(location => {
        const siteId = String(location.scats_site);
        if (!locationsBySite.has(siteId)) locationsBySite.set(siteId, []);
        locationsBySite.get(siteId).push(location);
      });

      [...locationsBySite.keys()].sort((a, b) => Number(a) - Number(b)).forEach(siteId => {
        originSiteSelect.appendChild(option(siteId, siteId));
        destinationSiteSelect.appendChild(option(siteId, siteId));
      });
      if (destinationSiteSelect.options.length > 1) destinationSiteSelect.selectedIndex = 1;
      populateLocations(originSiteSelect, originLocationSelect);
      populateLocations(destinationSiteSelect, destinationLocationSelect);
      document.querySelector("#date").value = config.default_date;
      document.querySelector("#time").value = config.default_time;
      document.querySelector("#k").value = config.default_top_k;
      status.textContent = `${locations.length} SCATS locations loaded`;
      renderGraph();
    }

    function projectGraph() {
      const width = 900;
      const height = 560;
      const padding = 34;
      const lats = graphData.nodes.map(node => node.latitude);
      const lons = graphData.nodes.map(node => node.longitude);
      const minLat = Math.min(...lats);
      const maxLat = Math.max(...lats);
      const minLon = Math.min(...lons);
      const maxLon = Math.max(...lons);
      const lonRange = maxLon - minLon || 1;
      const latRange = maxLat - minLat || 1;
      const positions = new Map();
      graphData.nodes.forEach(node => {
        const x = padding + ((node.longitude - minLon) / lonRange) * (width - padding * 2);
        const y = padding + ((maxLat - node.latitude) / latRange) * (height - padding * 2);
        positions.set(String(node.scats_site), { x, y, node });
      });
      return positions;
    }

    function renderGraph(shortestRoute = null) {
      graphSvg.replaceChildren();
      if (!graphData.nodes.length) return;

      const positions = projectGraph();
      const shortestPath = shortestRoute?.path || [];
      const pathNodes = new Set(shortestPath.map(String));
      const originSite = selectedLocation(originSiteSelect, originLocationSelect).site;
      const destinationSite = selectedLocation(destinationSiteSelect, destinationLocationSelect).site;
      const pathEdges = new Set();
      for (let i = 0; i < shortestPath.length - 1; i++) {
        pathEdges.add(`${shortestPath[i]}-${shortestPath[i + 1]}`);
        pathEdges.add(`${shortestPath[i + 1]}-${shortestPath[i]}`);
      }

      const edgeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const pathLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const costLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      const nodeLayer = document.createElementNS("http://www.w3.org/2000/svg", "g");
      graphSvg.append(edgeLayer, pathLayer, costLayer, nodeLayer);

      graphData.edges.forEach(edge => {
        const from = positions.get(String(edge.from));
        const to = positions.get(String(edge.to));
        if (!from || !to) return;
        const line = document.createElementNS("http://www.w3.org/2000/svg", "line");
        line.setAttribute("x1", from.x);
        line.setAttribute("y1", from.y);
        line.setAttribute("x2", to.x);
        line.setAttribute("y2", to.y);
        const isPath = pathEdges.has(`${edge.from}-${edge.to}`);
        line.setAttribute("class", isPath ? "graph-edge path" : `graph-edge ${edge.kind || ""}`);
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `${edge.from} -> ${edge.to}: ${edge.road || "road link"}`;
        line.appendChild(title);
        (isPath ? pathLayer : edgeLayer).appendChild(line);

      });

      (shortestRoute?.steps || []).forEach((step, index) => {
        const from = positions.get(String(step.from_site));
        const to = positions.get(String(step.to_site));
        if (!from || !to) return;

        const label = document.createElementNS("http://www.w3.org/2000/svg", "text");
        const midX = (from.x + to.x) / 2;
        const midY = (from.y + to.y) / 2;
        const dx = to.x - from.x;
        const dy = to.y - from.y;
        const length = Math.hypot(dx, dy) || 1;
        const offset = index % 2 === 0 ? 12 : -12;
        label.setAttribute("x", midX + (-dy / length) * offset);
        label.setAttribute("y", midY + (dx / length) * offset);
        label.setAttribute("class", "graph-cost");
        label.setAttribute("text-anchor", "middle");
        label.textContent = `${step.travel_time_minutes.toFixed(1)} min`;
        costLayer.appendChild(label);
      });

      graphData.nodes.forEach(node => {
        const pos = positions.get(String(node.scats_site));
        const group = document.createElementNS("http://www.w3.org/2000/svg", "g");
        const circle = document.createElementNS("http://www.w3.org/2000/svg", "circle");
        circle.setAttribute("cx", pos.x);
        circle.setAttribute("cy", pos.y);
        circle.setAttribute("r", pathNodes.has(String(node.scats_site)) ? 7 : 5);
        const classes = ["graph-node"];
        if (pathNodes.has(String(node.scats_site))) classes.push("path");
        if (String(node.scats_site) === originSite) classes.push("origin");
        if (String(node.scats_site) === destinationSite) classes.push("destination");
        circle.setAttribute("class", classes.join(" "));
        const title = document.createElementNS("http://www.w3.org/2000/svg", "title");
        title.textContent = `${node.scats_site} - ${node.location}`;
        circle.appendChild(title);
        group.appendChild(circle);
        if (pathNodes.has(String(node.scats_site))) {
          const text = document.createElementNS("http://www.w3.org/2000/svg", "text");
          text.setAttribute("x", pos.x + 9);
          text.setAttribute("y", pos.y - 9);
          text.setAttribute("class", "graph-label");
          text.textContent = node.scats_site;
          group.appendChild(text);
        }
        nodeLayer.appendChild(group);
      });

      graphSummary.textContent = shortestPath.length
        ? `A* shortest path: ${shortestRoute.total_minutes.toFixed(1)} min`
        : `${graphData.nodes.length} nodes, ${graphData.edges.length} edges`;
    }

    async function findRoutes() {
      findButton.disabled = true;
      status.textContent = "Calculating routes...";
      rows.replaceChildren();
      const origin = selectedLocation(originSiteSelect, originLocationSelect);
      const destination = selectedLocation(destinationSiteSelect, destinationLocationSelect);
      const params = new URLSearchParams({
        model: modelSelect.value,
        origin: origin.site,
        origin_location: origin.location,
        destination: destination.site,
        destination_location: destination.location,
        date: document.querySelector("#date").value,
        time: document.querySelector("#time").value,
        k: document.querySelector("#k").value
      });
      try {
        const response = await fetch(`/api/routes?${params}`);
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.error || "Route search failed");
        const shortestRoute = payload.routes[0] || null;
        renderGraph(shortestRoute);
        payload.routes.forEach(route => {
          const tr = document.createElement("tr");
          tr.innerHTML = `
            <td>${route.rank}</td>
            <td>${route.path.join(" -> ")}</td>
            <td>${route.total_minutes.toFixed(1)} min</td>
          `;
          rows.appendChild(tr);
        });
        hasRouteResults = payload.routes.length > 0;
        status.className = "status";
        status.textContent = `${payload.routes.length} route(s) found. Shortest path highlighted on graph.`;
      } catch (error) {
        hasRouteResults = false;
        status.className = "status error";
        status.textContent = error.message;
      } finally {
        findButton.disabled = false;
      }
    }

    findButton.addEventListener("click", findRoutes);
    modelSelect.addEventListener("change", markRouteResultsStale);
    originSiteSelect.addEventListener("change", () => {
      populateLocations(originSiteSelect, originLocationSelect);
      markRouteResultsStale();
    });
    originLocationSelect.addEventListener("change", markRouteResultsStale);
    destinationSiteSelect.addEventListener("change", () => {
      populateLocations(destinationSiteSelect, destinationLocationSelect);
      markRouteResultsStale();
    });
    destinationLocationSelect.addEventListener("change", markRouteResultsStale);
    ["#date", "#time", "#k"].forEach(selector => {
      const input = document.querySelector(selector);
      input.addEventListener("input", markRouteResultsStale);
      input.addEventListener("change", markRouteResultsStale);
    });
    loadDefaults().catch(error => {
      status.className = "status error";
      status.textContent = error.message;
    });
  </script>
</body>
</html>
"""


class TbrgsHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:
        parsed = urlparse(self.path)
        if parsed.path == "/":
            self._send_text(HTML, "text/html; charset=utf-8")
        elif parsed.path == "/api/models":
            models = [{"id": model_id, "label": label} for model_id, label in registry.list_models()]
            self._send_json(models)
        elif parsed.path == "/api/sites":
            sites = [
                {
                    "scats_site": site.scats_site,
                    "location": site.location,
                    "latitude": site.latitude,
                    "longitude": site.longitude,
                }
                for site in sorted(load_scats_sites().values(), key=lambda item: item.scats_site)
            ]
            self._send_json(sites)
        elif parsed.path == "/api/locations":
            self._send_json(load_scats_locations())
        elif parsed.path == "/api/graph":
            sites_by_id = load_scats_sites()
            edge_map = build_realistic_scats_edges(sites_by_id)
            edges = [
                {
                    "from": edge.from_site,
                    "to": edge.to_site,
                    "kind": edge.kind,
                    "road": edge.road,
                    "distance_km": edge.distance_km,
                }
                for edge in sorted(edge_map.values(), key=lambda item: (item.from_site, item.to_site))
            ]
            self._send_json(
                {
                    "nodes": [
                        {
                            "scats_site": site.scats_site,
                            "location": site.location,
                            "latitude": site.latitude,
                            "longitude": site.longitude,
                        }
                        for site in sorted(sites_by_id.values(), key=lambda item: item.scats_site)
                    ],
                    "edges": edges,
                }
            )
        elif parsed.path == "/api/config":
            self._send_json(load_config())
        elif parsed.path == "/api/routes":
            self._handle_routes(parsed.query)
        else:
            self.send_error(404)

    def log_message(self, format: str, *args: object) -> None:
        return

    def _handle_routes(self, query: str) -> None:
        try:
            params = parse_qs(query)
            origin = int(params.get("origin", [""])[0])
            destination = int(params.get("destination", [""])[0])
            origin_location = params.get("origin_location", [""])[0] or None
            destination_location = params.get("destination_location", [""])[0] or None
            model_id = params.get("model", ["xgboost"])[0]
            k = int(params.get("k", ["5"])[0])
            timestamp = pd.Timestamp(f"{params.get('date', ['2006-10-23'])[0]} {params.get('time', ['08:00'])[0]}")
            routes = find_top_k_routes(
                origin,
                destination,
                timestamp,
                model_id,
                k=k,
                origin_location=origin_location,
                destination_location=destination_location,
            )
            self._send_json(
                {
                    "routes": [
                        {
                            "rank": index,
                            "path": route.path,
                            "total_minutes": route.total_minutes,
                            "steps": [
                                {
                                    "from_site": step.from_site,
                                    "to_site": step.to_site,
                                    "from_location": step.from_location,
                                    "distance_km": step.distance_km,
                                    "predicted_flow_hour": step.predicted_flow_hour,
                                    "speed_kmh": step.speed_kmh,
                                    "travel_time_minutes": step.travel_time_minutes,
                                }
                                for step in route.steps
                            ],
                        }
                        for index, route in enumerate(routes, start=1)
                    ]
                }
            )
        except Exception as exc:
            self._send_json({"error": str(exc)}, status=400)

    def _send_text(self, body: str, content_type: str, status: int = 200) -> None:
        data = body.encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _send_json(self, payload: object, status: int = 200) -> None:
        self._send_text(json.dumps(payload), "application/json; charset=utf-8", status=status)


def main(host: str | None = None, port: int | None = None) -> None:
    config = load_config()
    host = host or str(config.get("host", HOST))
    port = port or int(config.get("port", PORT))
    server = ThreadingHTTPServer((host, port), TbrgsHandler)
    print(f"TBRGS UI running at http://{host}:{port}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
