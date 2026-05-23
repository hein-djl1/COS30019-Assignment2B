from __future__ import annotations

from pathlib import Path

import pandas as pd


def _scale(value: float, src_min: float, src_max: float, dst_min: float, dst_max: float) -> float:
    if src_max == src_min:
        return (dst_min + dst_max) / 2
    return dst_min + (value - src_min) * (dst_max - dst_min) / (src_max - src_min)


def _polyline(points: list[tuple[float, float]]) -> str:
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def save_loss_curve(history_df: pd.DataFrame, output_path: Path, title: str) -> None:
    epochs = history_df["epoch"].astype(float).to_numpy()
    train_loss = history_df["loss"].astype(float).to_numpy()
    val_loss = history_df["val_loss"].astype(float).to_numpy()

    width = 900
    height = 560
    left = 90
    right = 30
    top = 60
    bottom = 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    x_min = float(epochs.min())
    x_max = float(epochs.max())
    y_min = 0.0
    y_max = float(max(train_loss.max(), val_loss.max()))
    y_max = y_max * 1.08 if y_max else 1.0

    def to_point(epoch: float, loss: float) -> tuple[float, float]:
        x = _scale(epoch, x_min, x_max, left, left + plot_width)
        y = _scale(loss, y_min, y_max, top + plot_height, top)
        return x, y

    train_points = [to_point(float(epoch), float(loss)) for epoch, loss in zip(epochs, train_loss)]
    val_points = [to_point(float(epoch), float(loss)) for epoch, loss in zip(epochs, val_loss)]

    y_ticks = []
    for i in range(6):
        value = y_min + (y_max - y_min) * i / 5
        y = _scale(value, y_min, y_max, top + plot_height, top)
        y_ticks.append((value, y))

    x_ticks = []
    tick_count = min(6, len(epochs))
    for i in range(tick_count):
        value = x_min + (x_max - x_min) * i / max(1, tick_count - 1)
        x = _scale(value, x_min, x_max, left, left + plot_width)
        x_ticks.append((value, x))

    grid_lines = []
    for value, y in y_ticks:
        grid_lines.append(
            f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" '
            'stroke="#e5e7eb" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#475569">{value:.4g}</text>'
        )

    for value, x in x_ticks:
        grid_lines.append(
            f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" '
            'stroke="#f1f5f9" stroke-width="1" />'
        )
        grid_lines.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" '
            f'font-size="12" fill="#475569">{value:.0f}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <rect width="100%" height="100%" fill="#ffffff" />
  <text x="{width / 2}" y="32" text-anchor="middle" font-size="22" font-family="Arial, sans-serif" font-weight="700" fill="#0f172a">{title}</text>
  <g font-family="Arial, sans-serif">
    {''.join(grid_lines)}
    <line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#334155" stroke-width="1.5" />
    <line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#334155" stroke-width="1.5" />
    <text x="{left + plot_width / 2}" y="{height - 24}" text-anchor="middle" font-size="14" fill="#0f172a">Epoch</text>
    <text x="24" y="{top + plot_height / 2}" text-anchor="middle" font-size="14" fill="#0f172a" transform="rotate(-90 24 {top + plot_height / 2})">MSE loss on normalized target</text>
    <polyline points="{_polyline(train_points)}" fill="none" stroke="#2563eb" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
    <polyline points="{_polyline(val_points)}" fill="none" stroke="#dc2626" stroke-width="3" stroke-linejoin="round" stroke-linecap="round" />
    <line x1="{width - 220}" y1="70" x2="{width - 190}" y2="70" stroke="#2563eb" stroke-width="3" />
    <text x="{width - 180}" y="75" font-size="13" fill="#0f172a">Training loss</text>
    <line x1="{width - 220}" y1="94" x2="{width - 190}" y2="94" stroke="#dc2626" stroke-width="3" />
    <text x="{width - 180}" y="99" font-size="13" fill="#0f172a">Validation loss</text>
  </g>
</svg>
'''
    output_path.write_text(svg, encoding="utf-8")
