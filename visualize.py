"""
visualize.py — Graph visualizer for COS30019 route finding problems

Usage:
    # Visualize a single file
    python visualize.py test_problems/t01_default_test.txt

    # Visualize all files in a folder
    python visualize.py test_problems/

    # Visualize multiple specific files
    python visualize.py test_problems/t01_default_test.txt test_problems/t02_simple_path.txt
"""

import sys
import os
import math
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch
from parser import parse_file


# ── Colours ────────────────────────────────────────────────────────────────────
COL_ORIGIN      = "#2ecc71"   # green
COL_DEST        = "#e74c3c"   # red
COL_BOTH        = "#e67e22"   # orange  (origin == destination)
COL_NORMAL      = "#3498db"   # blue
COL_EDGE        = "#555555"
COL_LABEL       = "white"
COL_COST        = "#c0392b"
COL_BG          = "#f9f9f9"


def draw_graph(filepath, ax):
    """Parse one problem file and draw it on the given Axes."""
    graph = parse_file(filepath)
    title = os.path.basename(filepath)

    # ── positions dict  {node_id: (x, y)} ──────────────────────────────────
    pos = {nid: (n.x, n.y) for nid, n in graph.nodes.items()}

    # ── node colours ────────────────────────────────────────────────────────
    origin_id = graph.origin.id
    dest_ids  = {d.id for d in graph.destinations}

    def node_colour(nid):
        is_origin = (nid == origin_id)
        is_dest   = (nid in dest_ids)
        if is_origin and is_dest:
            return COL_BOTH
        if is_origin:
            return COL_ORIGIN
        if is_dest:
            return COL_DEST
        return COL_NORMAL

    colours = [node_colour(nid) for nid in graph.nodes]

    # ── draw edges using matplotlib (so we can use actual coordinates) ──────
    node_radius = 0.25   # visual radius for arrow offset

    for from_id, neighbours in graph.edges.items():
        fx, fy = pos[from_id]
        for to_id, cost in neighbours:
            tx, ty = pos[to_id]

            # skip self-loops arrow (draw separately)
            if from_id == to_id:
                ax.annotate("", xy=(fx + 0.3, fy + 0.3), xytext=(fx, fy),
                            arrowprops=dict(arrowstyle="->", color=COL_EDGE,
                                            lw=1.2,
                                            connectionstyle="arc3,rad=0.8"))
                # cost label for self-loop
                ax.text(fx + 0.45, fy + 0.45, str(int(cost) if cost == int(cost) else cost),
                        fontsize=7, color=COL_COST, ha="center", va="center",
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.7))
                continue

            # check if reverse edge exists → curve both
            reverse_exists = any(nb == from_id for nb, _ in graph.edges.get(to_id, []))
            rad = 0.25 if reverse_exists else 0.0

            ax.annotate("", xy=(tx, ty), xytext=(fx, fy),
                        arrowprops=dict(arrowstyle="-|>",
                                        color=COL_EDGE,
                                        lw=1.2,
                                        mutation_scale=12,
                                        connectionstyle=f"arc3,rad={rad}"))

            # cost label at midpoint (offset slightly if curved)
            mx = (fx + tx) / 2
            my = (fy + ty) / 2
            if rad != 0:
                # perpendicular offset
                dx, dy = tx - fx, ty - fy
                length = math.sqrt(dx*dx + dy*dy) or 1
                mx += -dy / length * 0.18
                my +=  dx / length * 0.18

            label = str(int(cost) if cost == int(cost) else cost)
            ax.text(mx, my, label, fontsize=7, color=COL_COST,
                    ha="center", va="center",
                    bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="none", alpha=0.8))

    # ── draw nodes ───────────────────────────────────────────────────────────
    for nid, node in graph.nodes.items():
        x, y = pos[nid]
        circle = plt.Circle((x, y), node_radius, color=node_colour(nid),
                             zorder=3, linewidth=1.5, edgecolor="white")
        ax.add_patch(circle)
        ax.text(x, y, str(nid), fontsize=8, fontweight="bold",
                ha="center", va="center", color=COL_LABEL, zorder=4)

    # ── legend ───────────────────────────────────────────────────────────────
    legend_items = [
        mpatches.Patch(color=COL_ORIGIN, label="Origin"),
        mpatches.Patch(color=COL_DEST,   label="Destination"),
    ]
    if any(nid == origin_id and nid in dest_ids for nid in graph.nodes):
        legend_items.append(mpatches.Patch(color=COL_BOTH, label="Origin & Dest"))
    ax.legend(handles=legend_items, fontsize=7, loc="upper left",
              framealpha=0.7, edgecolor="gray")

    # ── axes formatting ──────────────────────────────────────────────────────
    all_x = [n.x for n in graph.nodes.values()]
    all_y = [n.y for n in graph.nodes.values()]
    pad = 1.0
    ax.set_xlim(min(all_x) - pad, max(all_x) + pad)
    ax.set_ylim(min(all_y) - pad, max(all_y) + pad)
    ax.set_aspect("equal")
    ax.set_facecolor(COL_BG)
    ax.grid(True, linestyle="--", alpha=0.4)
    ax.set_title(title, fontsize=9, fontweight="bold", pad=6)


def collect_files(args):
    """Return a list of .txt file paths from the given arguments."""
    files = []
    for arg in args:
        if os.path.isdir(arg):
            for fname in sorted(os.listdir(arg)):
                if fname.endswith(".txt"):
                    files.append(os.path.join(arg, fname))
        elif os.path.isfile(arg):
            files.append(arg)
        else:
            print(f"Warning: '{arg}' not found, skipping.")
    return files


def main():
    if len(sys.argv) < 2:
        print("Usage:")
        print("  python visualize.py <file.txt>")
        print("  python visualize.py <folder/>")
        print("  python visualize.py file1.txt file2.txt ...")
        sys.exit(1)

    files = collect_files(sys.argv[1:])
    if not files:
        print("No .txt files found.")
        sys.exit(1)

    n     = len(files)
    ncols = min(3, n)
    nrows = math.ceil(n / ncols)

    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(6 * ncols, 5 * nrows),
                              facecolor="white")

    # flatten axes into a list
    if n == 1:
        axes = [axes]
    else:
        axes = list(axes.flat)

    for i, filepath in enumerate(files):
        try:
            draw_graph(filepath, axes[i])
        except Exception as e:
            axes[i].set_title(f"ERROR: {os.path.basename(filepath)}", color="red")
            axes[i].text(0.5, 0.5, str(e), transform=axes[i].transAxes,
                         ha="center", va="center", fontsize=8, color="red")

    # hide unused subplots
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.tight_layout(pad=2.0)
    plt.savefig("graph_visualization.png", dpi=150, bbox_inches="tight")
    print("Saved: graph_visualization.png")
    plt.show()


if __name__ == "__main__":
    main()
