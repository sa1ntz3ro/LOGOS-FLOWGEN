#!/usr/bin/env python3
"""
Read and normalize pipe velocity profile tables from two typical formats:
1) Exported CFD table like:
   Velocity magnitude;X Center;Y Center;Z Center
   1.56713\t11\t0.0311476\t0.0295256
2) Generated profile like:
   U,YC,ZC
   7.99e-04,2.29e-03,0.0

The script auto-detects delimiters, recognizes velocity/coordinate columns,
infer the flow direction when possible, prints a summary, optionally plots
the cross-section, and can export the data to a normalized CSV.

Important note:
- For files containing "Velocity magnitude" only, the script cannot recover
  the vector direction from the values alone. In that case it infers the most
  likely flow axis from the coordinate that is nearly constant.
"""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import matplotlib.tri as mtri
import numpy as np
import pandas as pd


# -----------------------------
# Header normalization helpers
# -----------------------------

def normalize_header(name: str) -> str:
    """Normalize a column header for robust matching."""
    s = name.strip().lower()
    s = s.replace('center', 'c')
    s = s.replace('coordinate', 'c')
    s = s.replace('coord', 'c')
    s = s.replace('velocity magnitude', 'velmag')
    s = s.replace('velocity', 'vel')
    s = s.replace('magnitude', 'mag')
    s = re.sub(r'[^a-z0-9]+', '', s)
    return s


def map_columns(columns: List[str]) -> Dict[str, str]:
    """Map raw columns to canonical names.

    Canonical names:
    - U, V, W, VMAG
    - XC, YC, ZC
    """
    out: Dict[str, str] = {}
    for col in columns:
        n = normalize_header(col)

        # Velocity component columns
        if n in {'u', 'velu', 'ux', 'velocityx'}:
            out['U'] = col
            continue
        if n in {'v', 'velv', 'uy', 'velocityy'}:
            out['V'] = col
            continue
        if n in {'w', 'velw', 'uz', 'velocityz'}:
            out['W'] = col
            continue
        if n in {'velmag', 'vmag', 'velocitymagnitude', 'magnitude'}:
            out['VMAG'] = col
            continue

        # Coordinate columns
        if n in {'xc', 'xcenter', 'x', 'cx'}:
            out['XC'] = col
            continue
        if n in {'yc', 'ycenter', 'y', 'cy'}:
            out['YC'] = col
            continue
        if n in {'zc', 'zcenter', 'z', 'cz'}:
            out['ZC'] = col
            continue

    return out


# -----------------------------
# File parsing helpers
# -----------------------------

def sniff_delimiter(text: str) -> Optional[str]:
    """Try to detect the delimiter from a sample."""
    candidates = [',', ';', '\t']
    try:
        dialect = csv.Sniffer().sniff(text, delimiters=''.join(candidates))
        return dialect.delimiter
    except csv.Error:
        pass

    # Fallback: pick the most frequent candidate in the first line.
    first_line = text.splitlines()[0] if text.splitlines() else text
    counts = {d: first_line.count(d) for d in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else None



def read_profile_file(path: Path) -> pd.DataFrame:
    """Read a profile file while handling mixed delimiter edge cases.

    The original example has ';' in the header but tab-separated data rows,
    so we first parse the header manually and then parse data rows flexibly.
    """
    raw = path.read_text(encoding='utf-8', errors='replace')
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        raise ValueError(f'File is empty: {path}')

    header_line = lines[0]
    data_lines = lines[1:]

    # Header delimiter may differ from data delimiter.
    header_delim = sniff_delimiter(header_line) or ','
    headers = [h.strip() for h in header_line.split(header_delim)]

    # Detect delimiter on actual data lines.
    data_sample = '\n'.join(data_lines[:10])
    data_delim = sniff_delimiter(data_sample)

    records: List[List[str]] = []
    for line in data_lines:
        if data_delim:
            parts = [p.strip() for p in line.split(data_delim)]
        else:
            # Final fallback: split on any whitespace.
            parts = re.split(r'\s+', line.strip())

        # Some files may still contain semicolons in header-like rows.
        if len(parts) == 1 and ';' in line:
            parts = [p.strip() for p in line.split(';')]

        if len(parts) != len(headers):
            # Try a generic split on comma/semicolon/tab/whitespace.
            parts = [p.strip() for p in re.split(r'[;,\t]|\s+', line.strip()) if p.strip()]

        if len(parts) != len(headers):
            raise ValueError(
                f'Could not parse line with {len(parts)} fields, expected {len(headers)}: {line}'
            )
        records.append(parts)

    df = pd.DataFrame(records, columns=headers)

    # Convert all columns to numeric if possible.
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='raise')

    return df


# -----------------------------
# Inference and normalization
# -----------------------------

def infer_flow_direction(df: pd.DataFrame, colmap: Dict[str, str]) -> Tuple[str, str, Tuple[str, str]]:
    """Infer flow direction and cross-section coordinates.

    Returns:
        velocity_label: U/V/W/VMAG
        direction_axis: X/Y/Z
        cross_section: pair of canonical coordinate labels
    """
    if 'U' in colmap:
        return 'U', 'X', ('YC', 'ZC')
    if 'V' in colmap:
        return 'V', 'Y', ('XC', 'ZC')
    if 'W' in colmap:
        return 'W', 'Z', ('XC', 'YC')

    if 'VMAG' not in colmap:
        raise ValueError('No velocity column found. Expected U, V, W or Velocity magnitude.')

    # Infer axis from the coordinate that is nearly constant.
    candidates = []
    for coord in ['XC', 'YC', 'ZC']:
        if coord in colmap:
            series = df[colmap[coord]].to_numpy(dtype=float)
            spread = float(np.max(series) - np.min(series))
            candidates.append((coord, spread))

    if not candidates:
        raise ValueError('Velocity magnitude found, but no coordinate columns available.')

    axial_coord = min(candidates, key=lambda x: x[1])[0]
    axis = axial_coord[0]  # X, Y or Z
    cross = tuple(c for c in ['XC', 'YC', 'ZC'] if c != axial_coord)
    if len(cross) != 2:
        raise ValueError('Could not infer cross-section coordinates.')

    return 'VMAG', axis, (cross[0], cross[1])



def build_normalized_dataframe(df: pd.DataFrame, colmap: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    """Convert the raw dataframe to a normalized one.

    Output columns:
    - velocity
    - c1, c2   : cross-section coordinates
    - axis_pos : axial coordinate if available
    - radius   : distance from the inferred section center
    """
    vel_label, axis, (c1_label, c2_label) = infer_flow_direction(df, colmap)

    velocity = df[colmap[vel_label]].astype(float).to_numpy()
    c1 = df[colmap[c1_label]].astype(float).to_numpy()
    c2 = df[colmap[c2_label]].astype(float).to_numpy()

    # Axial coordinate is optional when component velocity is already given.
    axis_coord_label = f'{axis}C'
    axis_pos = df[colmap[axis_coord_label]].astype(float).to_numpy() if axis_coord_label in colmap else np.full_like(c1, np.nan)

    # Estimate section center from min/max bounds.
    c1_center = 0.5 * (float(np.min(c1)) + float(np.max(c1)))
    c2_center = 0.5 * (float(np.min(c2)) + float(np.max(c2)))
    radius = np.sqrt((c1 - c1_center) ** 2 + (c2 - c2_center) ** 2)

    out = pd.DataFrame({
        'velocity': velocity,
        'c1': c1,
        'c2': c2,
        'axis_pos': axis_pos,
        'radius': radius,
    })

    meta = {
        'velocity_label': vel_label,
        'flow_axis': axis,
        'cross1_label': c1_label,
        'cross2_label': c2_label,
        'axis_coord_label': axis_coord_label,
        'c1_center': f'{c1_center:.12e}',
        'c2_center': f'{c2_center:.12e}',
    }
    return out, meta


# -----------------------------
# Reporting and plotting
# -----------------------------

def print_summary(df_raw: pd.DataFrame, df_norm: pd.DataFrame, meta: Dict[str, str]) -> None:
    """Print a concise summary of the detected file content."""
    print('=== File summary ===')
    print(f'Rows: {len(df_raw)}')
    print(f'Detected flow axis: {meta["flow_axis"]}')
    print(f'Detected velocity column: {meta["velocity_label"]}')
    print(f'Cross-section coordinates: {meta["cross1_label"]}, {meta["cross2_label"]}')
    print(f'Estimated section center: ({meta["c1_center"]}, {meta["c2_center"]})')
    print(f'Velocity min/max: {df_norm["velocity"].min():.12e} / {df_norm["velocity"].max():.12e}')
    print(f'Radius min/max:   {df_norm["radius"].min():.12e} / {df_norm["radius"].max():.12e}')

    if not np.all(np.isnan(df_norm['axis_pos'].to_numpy())):
        axis_unique = np.unique(np.round(df_norm['axis_pos'].to_numpy(), decimals=12))
        if len(axis_unique) == 1:
            print(f'Axial coordinate is constant: {axis_unique[0]:.12e}')
        else:
            print(f'Axial coordinate has {len(axis_unique)} unique values.')



def plot_cross_section(
    df_norm: pd.DataFrame,
    meta: Dict[str, str],
    output: Optional[Path] = None,
    *,
    show: bool = False,
) -> None:
    """Create a cross-section plot colored by velocity.

    The plot is rendered as a continuous filled field using triangulation-based
    interpolation, and masked outside the inferred pipe radius.
    """
    fig, ax = plt.subplots(figsize=(7, 6))
    vmin_raw = float(df_norm['velocity'].min())
    vmax_raw = float(df_norm['velocity'].max())
    vmin = vmin_raw
    vmax = vmax_raw
    if math.isclose(vmin, vmax, rel_tol=0.0, abs_tol=0.0):
        # Avoid a zero-range color scale when data are constant.
        eps = 1.0e-12 if vmin == 0.0 else abs(vmin) * 1.0e-12
        vmin -= eps
        vmax += eps

    x = df_norm['c1'].to_numpy(dtype=float)
    y = df_norm['c2'].to_numpy(dtype=float)
    v = df_norm['velocity'].to_numpy(dtype=float)

    # Build triangulation and mask triangles outside the pipe cross-section.
    tri = mtri.Triangulation(x, y)
    c1_center = float(meta.get('c1_center', 'nan'))
    c2_center = float(meta.get('c2_center', 'nan'))
    r_max = float(df_norm['radius'].max())

    # Triangle centroids.
    tris = tri.triangles
    xc = x[tris].mean(axis=1)
    yc = y[tris].mean(axis=1)
    rc = np.sqrt((xc - c1_center) ** 2 + (yc - c2_center) ** 2)
    tri.set_mask(rc > (r_max * 1.0005))

    # Filled continuous field.
    levels = 200
    cf = ax.tricontourf(tri, v, levels=levels, vmin=vmin, vmax=vmax, cmap='viridis')

    # Optional: overlay original points very lightly for reference.
    ax.scatter(x, y, c='k', s=2, alpha=0.15, linewidths=0)

    ax.set_xlabel(meta['cross1_label'])
    ax.set_ylabel(meta['cross2_label'])
    ax.set_title(f'Cross-section profile, flow axis = {meta["flow_axis"]}')
    ax.set_aspect('equal', adjustable='box')
    cb = plt.colorbar(cf, ax=ax)
    cb.set_label('Velocity [m/s]')

    def _fmt_decimal(x: float) -> str:
        s = f"{x:.6f}"
        s = s.rstrip("0").rstrip(".")
        return "0" if s == "-0" else s

    # Put min/max next to the colorbar so it never overlaps the profile.
    cb.ax.text(
        0.5,
        1.02,
        f"min = {_fmt_decimal(vmin_raw)}\nmax = {_fmt_decimal(vmax_raw)}\n[m/s]",
        transform=cb.ax.transAxes,
        ha='center',
        va='bottom',
        fontsize=9,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='0.5'),
        clip_on=False,
    )
    plt.tight_layout()

    if output:
        fig.savefig(output, dpi=200)
        print(f'Plot saved to: {output}')
    if show and not output:
        plt.show()
    elif show and output:
        plt.show()
    elif not output and not show:
        # Default behavior if called directly without flags.
        plt.show()


# -----------------------------
# CLI
# -----------------------------

def _resolve_input_path(p: Path) -> Path:
    """Resolve an input path. If relative, try CWD first, then the script directory."""
    if p.is_absolute():
        return p

    cwd_candidate = (Path.cwd() / p)
    if cwd_candidate.exists():
        return cwd_candidate

    script_dir = Path(__file__).resolve().parent
    script_candidate = (script_dir / p)
    return script_candidate


def main() -> None:
    parser = argparse.ArgumentParser(description='Read and normalize pipe velocity profile CSV/TXT files.')
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Ask for missing parameters in the console (wizard mode).',
    )
    parser.add_argument(
        'input_file',
        type=Path,
        nargs='?',
        default=None,
        help='Path to the input .csv/.txt file. Can be a full path or just a filename if it is in the script folder. If omitted, you will be prompted.',
    )
    parser.add_argument(
        '--export-normalized',
        type=Path,
        default=None,
                        help='Optional path for exporting normalized CSV with columns: velocity,c1,c2,axis_pos,radius')
    parser.add_argument('--plot', action='store_true',
                        help='Show a cross-section scatter plot colored by velocity.')
    parser.add_argument(
        '--save-plot',
        nargs='?',
        const='__AUTO__',
        type=str,
        default=None,
                        help='Optional path for saving the plot as PNG. If omitted, the plot is saved next to the input file by default. If provided without a value, also saves next to the input file.')
    args = parser.parse_args()

    if args.input_file is None:
        # Prompt for the input path if omitted.
        raw = input('Enter input file path (.csv/.txt), full path or filename: ').strip().strip('"')
        if not raw:
            raise SystemExit('No input file provided.')
        args.input_file = Path(raw)

    input_path = _resolve_input_path(args.input_file.expanduser())
    try:
        input_path = input_path.resolve()
    except OSError:
        # Keep as-is if the path cannot be resolved (e.g., invalid drive), let checks below handle it.
        pass

    if not input_path.exists():
        raise SystemExit(f'Input file not found: {input_path}')
    if not input_path.is_file():
        raise SystemExit(f'Input path is not a file: {input_path}')
    if input_path.suffix.lower() not in {'.csv', '.txt'}:
        print(f'Warning: unexpected file extension "{input_path.suffix}". Proceeding anyway.')

    df_raw = read_profile_file(input_path)
    colmap = map_columns(list(df_raw.columns))
    df_norm, meta = build_normalized_dataframe(df_raw, colmap)

    print_summary(df_raw, df_norm, meta)

    if args.export_normalized:
        out_path = args.export_normalized.expanduser()
        df_norm.to_csv(out_path, index=False)
        print(f'Normalized CSV saved to: {out_path}')

    # Always save a PNG next to the input file by default.
    if args.save_plot is None or args.save_plot == '__AUTO__':
        save_plot = input_path.with_name(f'{input_path.stem}_plot.png')
    else:
        save_plot = Path(args.save_plot).expanduser()

    plot_cross_section(df_norm, meta, output=save_plot, show=bool(args.plot))


if __name__ == '__main__':
    main()
