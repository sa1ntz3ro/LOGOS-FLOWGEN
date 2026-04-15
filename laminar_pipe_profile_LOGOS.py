#!/usr/bin/env python3
"""
Generate a laminar velocity profile table for a circular pipe inlet.

The script builds a wall-clustered polar point cloud, computes the fully developed
laminar (Poiseuille) profile from the target mass flow rate, and writes a CSV table
compatible with a coordinate-based inlet profile workflow.

Output columns depend on the selected flow direction:
    X  / -X -> U, YC, ZC
    Y  / -Y -> V, XC, ZC
    Z  / -Z -> W, XC, YC

Notes
-----
1. The input set {target y+, number of cells across diameter, growth factor} is
   generally over-constrained. This script preserves:
      - number of radial cells,
      - clustering pattern shape,
      - pipe radius,
   and then reports the achieved first-cell y+ after scaling the radial layers.
2. The generated points represent ring-sector cell centers in the pipe cross-section.
   The discrete profile is normalized so that the integrated mass flow exactly matches
   the target value.
3. For laminar flow, y+ is used only to estimate a wall-resolved first-cell size.
   It is not a turbulence-model requirement here.
"""

from __future__ import annotations

import argparse
import csv
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Sequence, Tuple


@dataclass
class RadialLayer:
    """Single annular layer represented by one radial center and one thickness."""

    r_inner: float
    r_outer: float
    r_center: float
    dr: float


@dataclass
class ProfilePoint:
    """One point in the exported profile table."""

    velocity_component: float
    coord1: float
    coord2: float
    area_weight: float
    radius: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a laminar inlet velocity profile table for a circular pipe."
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for all parameters in the console (wizard mode).",
    )

    # These may be provided via CLI or collected interactively.
    parser.add_argument("--mdot", type=float, default=None, help="Mass flow rate [kg/s].")
    parser.add_argument("--re", type=float, default=None, help="Reynolds number [-] (alternative to --mdot).")
    parser.add_argument("--diameter", type=float, default=None, help="Pipe diameter [m].")
    parser.add_argument("--rho", type=float, default=None, help="Density [kg/m^3].")
    parser.add_argument("--mu", type=float, default=None, help="Dynamic viscosity [Pa*s].")
    parser.add_argument(
        "--direction",
        type=str,
        default=None,
        choices=["X", "-X", "Y", "-Y", "Z", "-Z"],
        help="Main flow direction.",
    )
    parser.add_argument(
        "--cells-diameter",
        type=int,
        default=None,
        help="Approximate number of radial cells across the diameter.",
    )
    parser.add_argument(
        "--target-yplus",
        type=float,
        default=None,
        help="Preferred first-cell y+.",
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=None,
        help="Geometric growth factor from wall to center (>= 1).",
    )
    parser.add_argument(
        "--center1",
        type=float,
        default=0.0,
        help="First transverse coordinate of the pipe center [m].",
    )
    parser.add_argument(
        "--center2",
        type=float,
        default=0.0,
        help="Second transverse coordinate of the pipe center [m].",
    )
    parser.add_argument(
        "--min-points-per-ring",
        type=int,
        default=8,
        help="Minimum number of angular points for non-central rings.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("laminar_profile.csv"),
        help="Output CSV file path.",
    )

    return parser.parse_args()


def _prompt_str(prompt: str, default: str | None = None) -> str:
    if default is None:
        return input(f"{prompt}: ").strip()
    entered = input(f"{prompt} [{default}]: ").strip()
    return entered if entered else default


def _prompt_float(prompt: str, default: float | None = None, *, min_value: float | None = None) -> float:
    while True:
        raw = _prompt_str(prompt, None if default is None else f"{default}")
        try:
            value = float(raw)
        except ValueError:
            print("  Error: please enter a number (e.g. 0.0123).")
            continue
        if min_value is not None and value < min_value:
            print(f"  Error: value must be >= {min_value}.")
            continue
        return value


def _prompt_int(prompt: str, default: int | None = None, *, min_value: int | None = None) -> int:
    while True:
        raw = _prompt_str(prompt, None if default is None else f"{default}")
        try:
            value = int(raw)
        except ValueError:
            print("  Error: please enter an integer.")
            continue
        if min_value is not None and value < min_value:
            print(f"  Error: value must be >= {min_value}.")
            continue
        return value


def _prompt_choice(prompt: str, choices: Sequence[str], default: str | None = None) -> str:
    choices_str = "/".join(choices)
    while True:
        value = _prompt_str(f"{prompt} ({choices_str})", default).strip()
        if value in choices:
            return value
        print(f"  Error: choose one of: {choices_str}")


def _prompt_path(prompt: str, default: Path) -> Path:
    value = _prompt_str(prompt, str(default))
    return Path(value)


def collect_params(args: argparse.Namespace) -> argparse.Namespace:
    """
    Fill missing parameters either interactively or by validating CLI inputs.
    Wizard mode is enabled if:
      - --interactive is passed, or
      - the script is launched with no extra CLI args.
    """

    wizard = bool(getattr(args, "interactive", False)) or (len(sys.argv) <= 1)

    if wizard:
        print("=== Laminar pipe profile: parameter input ===")
        args.diameter = _prompt_float("Pipe diameter D [m]", args.diameter, min_value=0.0)
        args.rho = _prompt_float("Density rho [kg/m^3]", args.rho, min_value=0.0)
        args.mu = _prompt_float("Dynamic viscosity mu [Pa*s]", args.mu, min_value=0.0)

        flow_mode = _prompt_choice(
            "Specify flow by mdot or Re",
            ["mdot", "Re"],
            "mdot" if (args.mdot is not None or args.re is None) else "Re",
        )
        if flow_mode == "mdot":
            args.mdot = _prompt_float("Mass flow rate mdot [kg/s]", args.mdot, min_value=0.0)
            args.re = None
        else:
            args.re = _prompt_float("Reynolds number Re [-]", args.re, min_value=0.0)
            args.mdot = None

        args.direction = _prompt_choice("Flow direction", ["X", "-X", "Y", "-Y", "Z", "-Z"], args.direction or "X")
        args.cells_diameter = _prompt_int("Cells across diameter (approx.)", args.cells_diameter, min_value=1)
        args.target_yplus = _prompt_float("Target first-cell y+", args.target_yplus, min_value=0.0)
        args.growth_factor = _prompt_float("Growth factor (>=1)", args.growth_factor if args.growth_factor is not None else 1.1, min_value=1.0)
        args.center1 = _prompt_float("Pipe center coordinate (transverse 1) [m]", args.center1)
        args.center2 = _prompt_float("Pipe center coordinate (transverse 2) [m]", args.center2)
        args.min_points_per_ring = _prompt_int("Min points per ring", args.min_points_per_ring, min_value=1)
        args.output = _prompt_path("Output CSV path", args.output)
        print()
    else:
        missing = []
        for name in ("diameter", "rho", "mu", "direction", "cells_diameter", "target_yplus", "growth_factor"):
            if getattr(args, name) is None:
                missing.append(name)
        if missing:
            raise SystemExit(
                "Missing required parameters: "
                + ", ".join(missing)
                + ". Run without arguments for interactive mode, or add --interactive."
            )

    # Validate flow specification (exactly one of mdot or re).
    if args.mdot is not None and args.re is not None:
        raise SystemExit("Provide either --mdot or --re (not both).")
    if args.mdot is None and args.re is None:
        raise SystemExit("Flow is not specified: provide --mdot or --re (or use interactive mode).")

    # If Re is provided, compute the equivalent mass flow rate.
    if args.mdot is None and args.re is not None:
        if args.diameter <= 0.0 or args.rho <= 0.0 or args.mu <= 0.0:
            raise SystemExit("To compute mdot from Re, diameter, rho, and mu must be positive.")
        area = math.pi * args.diameter**2 / 4.0
        u_mean = args.re * args.mu / (args.rho * args.diameter)
        args.mdot = args.rho * u_mean * area

    return args


def compute_bulk_quantities(mdot: float, diameter: float, rho: float, mu: float) -> dict:
    """Compute mean velocity and wall quantities for a fully developed laminar pipe flow."""

    area = math.pi * diameter**2 / 4.0
    u_mean = mdot / (rho * area)
    nu = mu / rho
    re = rho * u_mean * diameter / mu

    # For fully developed laminar flow in a circular pipe:
    # tau_w = 8 * mu * U_mean / D
    tau_w = 8.0 * mu * u_mean / diameter
    u_tau = math.sqrt(tau_w / rho)

    return {
        "area": area,
        "u_mean": u_mean,
        "nu": nu,
        "re": re,
        "tau_w": tau_w,
        "u_tau": u_tau,
    }


def build_half_radius_layers(
    radius: float,
    cells_diameter: int,
    target_yplus: float,
    nu: float,
    u_tau: float,
    growth_factor: float,
) -> Tuple[List[RadialLayer], float, float, float]:
    """
    Build radial layers from center to wall.

    The first near-wall cell-center distance is estimated from target y+:
        y_first_center = y+ * nu / u_tau
    which corresponds to a first full cell thickness:
        dr_wall_target = 2 * y_first_center

    Because {target y+, number of cells, growth factor, radius} may conflict,
    the layer thicknesses are scaled to fill exactly half the pipe diameter.
    The achieved first-cell y+ is reported back.
    """

    half_cells = max(1, math.ceil(cells_diameter / 2))

    if growth_factor < 1.0:
        raise ValueError("growth_factor must be >= 1.0")

    if u_tau <= 0.0:
        raise ValueError("Computed friction velocity must be positive.")

    y_first_center_target = target_yplus * nu / u_tau
    dr_wall_target = 2.0 * y_first_center_target

    # Raw thicknesses are defined from the wall towards the center.
    raw_wall_to_center = [dr_wall_target * (growth_factor**i) for i in range(half_cells)]
    raw_sum = sum(raw_wall_to_center)

    if raw_sum <= 0.0:
        raise ValueError("Invalid raw radial thickness sum.")

    # Scale thicknesses so that the total exactly matches the pipe radius.
    scale = radius / raw_sum
    dr_wall_to_center = [dr * scale for dr in raw_wall_to_center]

    achieved_y_first_center = 0.5 * dr_wall_to_center[0]
    achieved_yplus = achieved_y_first_center * u_tau / nu

    # Convert the wall-to-center list into center-to-wall layers.
    dr_center_to_wall = list(reversed(dr_wall_to_center))

    layers: List[RadialLayer] = []
    r_inner = 0.0
    for dr in dr_center_to_wall:
        r_outer = r_inner + dr
        r_center = 0.5 * (r_inner + r_outer)
        layers.append(RadialLayer(r_inner=r_inner, r_outer=r_outer, r_center=r_center, dr=dr))
        r_inner = r_outer

    return layers, achieved_yplus, dr_wall_to_center[0], y_first_center_target


def estimate_points_per_ring(r_center: float, dr: float, min_points_per_ring: int) -> int:
    """Estimate the number of angular points for a given annular layer."""

    if r_center <= 1.0e-14:
        return 1

    circumference = 2.0 * math.pi * r_center

    # Use local radial thickness as the target tangential spacing scale.
    n_theta = max(min_points_per_ring, int(round(circumference / max(dr, 1.0e-14))))
    return max(1, n_theta)


def component_name_and_labels(direction: str) -> Tuple[str, Tuple[str, str], int]:
    """Map flow direction to the velocity component name, transverse labels, and sign."""

    sign = -1 if direction.startswith("-") else 1
    base = direction[-1]

    if base == "X":
        return "U", ("YC", "ZC"), sign
    if base == "Y":
        return "V", ("XC", "ZC"), sign
    if base == "Z":
        return "W", ("XC", "YC"), sign

    raise ValueError(f"Unsupported direction: {direction}")


def generate_profile_points(
    layers: Sequence[RadialLayer],
    radius: float,
    u_mean: float,
    mdot_target: float,
    rho: float,
    center1: float,
    center2: float,
    direction: str,
    min_points_per_ring: int,
) -> Tuple[List[ProfilePoint], float, float]:
    """
    Generate a discrete profile and normalize it to the exact target mass flow rate.

    Each ring is split into equal angular sectors. Sector area is used as the weight
    for exact discrete flow integration.
    """

    _, _, sign = component_name_and_labels(direction)
    points: List[ProfilePoint] = []

    for layer in layers:
        n_theta = estimate_points_per_ring(layer.r_center, layer.dr, min_points_per_ring)
        annulus_area = math.pi * (layer.r_outer**2 - layer.r_inner**2)
        sector_area = annulus_area / n_theta

        # Laminar fully developed profile (Poiseuille profile).
        u_axial = 2.0 * u_mean * (1.0 - (layer.r_center / radius) ** 2)
        u_axial *= sign

        if n_theta == 1:
            c1 = center1
            c2 = center2
            points.append(
                ProfilePoint(
                    velocity_component=u_axial,
                    coord1=c1,
                    coord2=c2,
                    area_weight=sector_area,
                    radius=layer.r_center,
                )
            )
            continue

        for j in range(n_theta):
            theta = 2.0 * math.pi * j / n_theta
            c1 = center1 + layer.r_center * math.cos(theta)
            c2 = center2 + layer.r_center * math.sin(theta)
            points.append(
                ProfilePoint(
                    velocity_component=u_axial,
                    coord1=c1,
                    coord2=c2,
                    area_weight=sector_area,
                    radius=layer.r_center,
                )
            )

    mdot_discrete = rho * sum(p.velocity_component * p.area_weight for p in points)

    if abs(mdot_discrete) < 1.0e-30:
        raise ValueError("Discrete mass flow is zero; cannot normalize the profile.")

    scale = mdot_target / mdot_discrete

    for p in points:
        p.velocity_component *= scale

    mdot_normalized = rho * sum(p.velocity_component * p.area_weight for p in points)
    return points, mdot_discrete, mdot_normalized


def write_csv(points: Sequence[ProfilePoint], direction: str, output_path: Path) -> None:
    """Write the profile table to CSV in the requested coordinate format."""

    component, (coord_a, coord_b), _ = component_name_and_labels(direction)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([component, coord_a, coord_b])
        for p in points:
            writer.writerow([
                f"{p.velocity_component:.12e}",
                f"{p.coord1:.12e}",
                f"{p.coord2:.12e}",
            ])


def main() -> None:
    args = collect_params(parse_args())

    radius = 0.5 * args.diameter

    bulk = compute_bulk_quantities(
        mdot=args.mdot,
        diameter=args.diameter,
        rho=args.rho,
        mu=args.mu,
    )

    layers, achieved_yplus, achieved_dr_wall, y_first_center_target = build_half_radius_layers(
        radius=radius,
        cells_diameter=args.cells_diameter,
        target_yplus=args.target_yplus,
        nu=bulk["nu"],
        u_tau=bulk["u_tau"],
        growth_factor=args.growth_factor,
    )

    points, mdot_discrete, mdot_normalized = generate_profile_points(
        layers=layers,
        radius=radius,
        u_mean=bulk["u_mean"],
        mdot_target=args.mdot,
        rho=args.rho,
        center1=args.center1,
        center2=args.center2,
        direction=args.direction,
        min_points_per_ring=args.min_points_per_ring,
    )

    write_csv(points=points, direction=args.direction, output_path=args.output)

    component, transverse_labels, sign = component_name_and_labels(args.direction)

    print("=== Laminar pipe profile summary ===")
    print(f"Output file                 : {args.output.resolve()}")
    print(f"Flow direction              : {args.direction} (component {component}, sign {sign:+d})")
    print(f"Transverse coordinates      : {transverse_labels[0]}, {transverse_labels[1]}")
    print(f"Pipe diameter [m]           : {args.diameter:.6f}")
    print(f"Pipe radius [m]             : {radius:.6f}")
    print(f"Mass flow target [kg/s]     : {args.mdot:.12e}")
    print(f"Density [kg/m^3]            : {args.rho:.6f}")
    print(f"Dynamic viscosity [Pa*s]    : {args.mu:.12e}")
    print(f"Kinematic viscosity [m^2/s] : {bulk['nu']:.12e}")
    print(f"Mean velocity [m/s]         : {bulk['u_mean']:.12e}")
    print(f"Reynolds number [-]         : {bulk['re']:.6f}")
    print(f"Wall shear stress [Pa]      : {bulk['tau_w']:.12e}")
    print(f"Friction velocity [m/s]     : {bulk['u_tau']:.12e}")
    print(f"Target first-center y [m]   : {y_first_center_target:.12e}")
    print(f"Achieved wall cell dr [m]   : {achieved_dr_wall:.12e}")
    print(f"Achieved first-cell y+ [-]  : {achieved_yplus:.6f}")
    print(f"Radial layers (half-radius) : {len(layers)}")
    print(f"Exported points             : {len(points)}")
    print(f"Discrete mdot before norm   : {mdot_discrete:.12e}")
    print(f"Discrete mdot after norm    : {mdot_normalized:.12e}")

    if bulk["re"] >= 2300.0:
        print("WARNING: Reynolds number is above the classical laminar threshold.")
        print("         The Poiseuille profile may not be physically valid for this case.")


if __name__ == "__main__":
    main()
