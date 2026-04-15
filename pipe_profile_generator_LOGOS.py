#!/usr/bin/env python3
"""
Generate a circular pipe inlet velocity profile table for LOGOS-like workflows.

Supported profile models
------------------------
1. laminar
   Fully developed laminar Poiseuille profile.

2. turbulent-powerlaw
   Engineering turbulent profile based on a power-law approximation.

3. turbulent-loglaw
   Turbulent profile based on wall units, friction factor, and roughness.

Output columns depend on the selected flow direction:
    X  / -X -> U, YC, ZC
    Y  / -Y -> V, XC, ZC
    Z  / -Z -> W, XC, YC

Notes
-----
1. The input set {target y+, number of cells across diameter, growth factor} is
   generally over-constrained. The script preserves:
      - number of radial cells,
      - clustering pattern shape,
      - pipe radius,
   and then reports the achieved first-cell y+ after scaling the radial layers.
2. The generated points represent ring-sector cell centers in the pipe cross-section.
   The discrete profile is normalized so that the integrated mass flow exactly matches
   the target value.
3. For turbulent profiles, this script checks Reynolds number and can reject or warn
   about transitional or laminar conditions unless explicitly allowed.
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
        description="Generate a circular pipe inlet velocity profile table."
    )

    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Ask for all parameters in the console (wizard mode).",
    )

    parser.add_argument(
        "--profile-model",
        type=str,
        default=None,
        choices=["laminar", "turbulent-powerlaw", "turbulent-loglaw"],
        help="Velocity profile model.",
    )

    # Flow may be specified by mass flow rate or Reynolds number.
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

    # Turbulence-model-related options.
    parser.add_argument(
        "--power-index",
        type=float,
        default=7.0,
        help="Exponent denominator n for the turbulent power-law profile (default: 7).",
    )
    parser.add_argument(
        "--roughness",
        type=float,
        default=0.0,
        help="Equivalent sand roughness height ks [m] for turbulent-loglaw.",
    )
    parser.add_argument(
        "--friction-model",
        type=str,
        default="auto",
        choices=["auto", "blasius", "haaland"],
        help="Friction factor model for turbulent-loglaw.",
    )
    parser.add_argument(
        "--allow-transitional",
        action="store_true",
        help="Allow turbulent profiles in the transitional range 2300 <= Re < 4000.",
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
        default=Path("pipe_profile.csv"),
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
        print("=== Pipe profile generator: parameter input ===")
        args.profile_model = _prompt_choice(
            "Profile model",
            ["laminar", "turbulent-powerlaw", "turbulent-loglaw"],
            args.profile_model or "laminar",
        )

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
        args.target_yplus = _prompt_float("Target first-cell y+", args.target_yplus if args.target_yplus is not None else 1.0, min_value=0.0)
        args.growth_factor = _prompt_float("Growth factor (>=1)", args.growth_factor if args.growth_factor is not None else 1.1, min_value=1.0)
        args.center1 = _prompt_float("Pipe center coordinate (transverse 1) [m]", args.center1)
        args.center2 = _prompt_float("Pipe center coordinate (transverse 2) [m]", args.center2)
        args.min_points_per_ring = _prompt_int("Min points per ring", args.min_points_per_ring, min_value=1)

        if args.profile_model == "turbulent-powerlaw":
            args.power_index = _prompt_float("Power-law exponent denominator n", args.power_index, min_value=1.0)

        if args.profile_model == "turbulent-loglaw":
            args.roughness = _prompt_float("Equivalent sand roughness ks [m]", args.roughness, min_value=0.0)
            args.friction_model = _prompt_choice(
                "Friction factor model", ["auto", "blasius", "haaland"], args.friction_model
            )
            args.allow_transitional = (
                _prompt_choice("Allow transitional Reynolds number", ["yes", "no"], "no") == "yes"
            )

        args.output = _prompt_path("Output CSV path", args.output)
        print()
    else:
        missing = []
        for name in (
            "profile_model",
            "diameter",
            "rho",
            "mu",
            "direction",
            "cells_diameter",
            "target_yplus",
            "growth_factor",
        ):
            if getattr(args, name) is None:
                missing.append(name)
        if missing:
            raise SystemExit(
                "Missing required parameters: "
                + ", ".join(missing)
                + ". Run without arguments for interactive mode, or add --interactive."
            )

    if args.mdot is not None and args.re is not None:
        raise SystemExit("Provide either --mdot or --re (not both).")
    if args.mdot is None and args.re is None:
        raise SystemExit("Flow is not specified: provide --mdot or --re (or use interactive mode).")

    if args.mdot is None and args.re is not None:
        if args.diameter <= 0.0 or args.rho <= 0.0 or args.mu <= 0.0:
            raise SystemExit("To compute mdot from Re, diameter, rho, and mu must be positive.")
        area = math.pi * args.diameter**2 / 4.0
        u_mean = args.re * args.mu / (args.rho * args.diameter)
        args.mdot = args.rho * u_mean * area

    if args.profile_model == "turbulent-loglaw" and args.roughness < 0.0:
        raise SystemExit("Roughness must be non-negative.")

    if args.profile_model == "turbulent-powerlaw" and args.power_index <= 1.0:
        raise SystemExit("Power-law exponent denominator n must be > 1.")

    return args


def classify_reynolds_number(re: float) -> str:
    """Classify the flow regime based on Reynolds number."""

    if re < 2300.0:
        return "laminar"
    if re < 4000.0:
        return "transitional"
    return "turbulent"


def enforce_model_applicability(profile_model: str, re: float, allow_transitional: bool) -> None:
    """Reject or warn about invalid combinations of profile model and Reynolds number."""

    regime = classify_reynolds_number(re)

    if profile_model == "laminar":
        if regime == "transitional":
            print("WARNING: Reynolds number is in the transitional range.")
            print("         A laminar Poiseuille profile may not be physically valid.")
        elif regime == "turbulent":
            print("WARNING: Reynolds number is in the turbulent range.")
            print("         A laminar Poiseuille profile is generally not physically valid.")
        return

    if regime == "laminar":
        raise SystemExit(
            "Requested a turbulent profile but Reynolds number is in the laminar range (Re < 2300)."
        )

    if regime == "transitional" and not allow_transitional:
        raise SystemExit(
            "Requested a turbulent profile but Reynolds number is transitional "
            "(2300 <= Re < 4000). Use --allow-transitional to override."
        )

    if regime == "transitional" and allow_transitional:
        print("WARNING: Reynolds number is transitional.")
        print("         The turbulent profile is an engineering approximation only.")


def compute_bulk_quantities(mdot: float, diameter: float, rho: float, mu: float) -> dict:
    """Compute mean velocity and basic flow quantities."""

    area = math.pi * diameter**2 / 4.0
    u_mean = mdot / (rho * area)
    nu = mu / rho
    re = rho * u_mean * diameter / mu

    # Laminar fully developed reference values.
    tau_w_laminar = 8.0 * mu * u_mean / diameter
    u_tau_laminar = math.sqrt(max(tau_w_laminar, 0.0) / rho)

    return {
        "area": area,
        "u_mean": u_mean,
        "nu": nu,
        "re": re,
        "tau_w_laminar": tau_w_laminar,
        "u_tau_laminar": u_tau_laminar,
    }


def friction_factor_blasius(re: float) -> float:
    """Blasius Darcy friction factor for smooth turbulent pipe flow."""

    return 0.3164 * re ** (-0.25)


def friction_factor_haaland(re: float, rel_roughness: float) -> float:
    """Haaland explicit approximation for the Darcy friction factor."""

    if re <= 0.0:
        raise ValueError("Reynolds number must be positive.")
    term = (rel_roughness / 3.7) ** 1.11 + 6.9 / re
    return (-1.8 * math.log10(term)) ** (-2.0)


def compute_turbulent_wall_quantities(
    re: float,
    u_mean: float,
    rho: float,
    roughness: float,
    diameter: float,
    friction_model: str,
) -> dict:
    """Compute turbulent wall shear stress and friction velocity."""

    rel_roughness = roughness / diameter if diameter > 0.0 else 0.0

    if friction_model == "blasius":
        if roughness > 0.0:
            print("WARNING: roughness is non-zero but Blasius assumes a smooth pipe.")
        f_darcy = friction_factor_blasius(re)
        model_used = "blasius"
    elif friction_model == "haaland":
        f_darcy = friction_factor_haaland(re, rel_roughness)
        model_used = "haaland"
    elif friction_model == "auto":
        if roughness <= 0.0:
            f_darcy = friction_factor_blasius(re)
            model_used = "blasius(auto)"
        else:
            f_darcy = friction_factor_haaland(re, rel_roughness)
            model_used = "haaland(auto)"
    else:
        raise ValueError(f"Unsupported friction model: {friction_model}")

    tau_w = f_darcy * rho * u_mean**2 / 8.0
    u_tau = math.sqrt(tau_w / rho)

    return {
        "f_darcy": f_darcy,
        "tau_w": tau_w,
        "u_tau": u_tau,
        "model_used": model_used,
        "rel_roughness": rel_roughness,
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

    raw_wall_to_center = [dr_wall_target * (growth_factor**i) for i in range(half_cells)]
    raw_sum = sum(raw_wall_to_center)

    if raw_sum <= 0.0:
        raise ValueError("Invalid raw radial thickness sum.")

    scale = radius / raw_sum
    dr_wall_to_center = [dr * scale for dr in raw_wall_to_center]

    achieved_y_first_center = 0.5 * dr_wall_to_center[0]
    achieved_yplus = achieved_y_first_center * u_tau / nu

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


def laminar_velocity_at_radius(r: float, radius: float, u_mean: float) -> float:
    """Poiseuille profile for fully developed laminar pipe flow."""

    return 2.0 * u_mean * (1.0 - (r / radius) ** 2)


def powerlaw_velocity_at_radius(r: float, radius: float, u_mean: float, power_index: float) -> float:
    """
    Turbulent engineering profile based on a power law.

    The local distance from the wall is:
        y = R - r

    The profile is:
        u = Umax * (y / R)^(1/n)

    Umax is chosen so that the area-averaged value equals u_mean.
    For a circular pipe:
        u_mean / Umax = 2 * [1/(a+1) - 1/(a+2)] = 2 / ((a+1)(a+2))
    where a = 1/n.
    Therefore:
        Umax = u_mean * (a+1)(a+2) / 2
    """

    if r >= radius:
        return 0.0

    a = 1.0 / power_index
    umax = u_mean * (a + 1.0) * (a + 2.0) / 2.0
    eta = max(0.0, 1.0 - r / radius)
    return umax * eta ** a


def roughness_shift_delta_b_plus(k_s_plus: float) -> float:
    """
    Very simple roughness shift model in wall units.

    This is an engineering approximation suitable for inlet-profile generation.
    """
    if k_s_plus <= 0.0:
        return 0.0
    if k_s_plus < 5.0:
        return 0.0
    if k_s_plus < 70.0:
        return max(0.0, (1.0 / 0.41) * math.log(1.0 + 0.3 * k_s_plus))
    return max(0.0, (1.0 / 0.41) * math.log(k_s_plus) - 3.5)


def loglaw_velocity_at_radius(
    r: float,
    radius: float,
    nu: float,
    u_tau: float,
    roughness: float,
) -> float:
    """
    Turbulent profile from wall units.

    The local wall-normal distance is:
        y = R - r

    Then:
        y+ = y * u_tau / nu
        u  = u_tau * u+(y+)

    A roughness shift is applied by reducing the additive constant in the log law.
    """

    if r >= radius:
        return 0.0

    y = radius - r
    y_plus = y * u_tau / nu
    k_s_plus = roughness * u_tau / nu if roughness > 0.0 else 0.0
    delta_b = roughness_shift_delta_b_plus(k_s_plus)

    if y_plus <= 0.0:
        return 0.0
    elif y_plus <= 5.0:
        u_plus = y_plus
    elif y_plus < 30.0:
        uplus_5 = 5.0
        uplus_30 = (1.0 / 0.41) * math.log(30.0) + 5.2 - delta_b
        w = (y_plus - 5.0) / 25.0
        u_plus = (1.0 - w) * uplus_5 + w * uplus_30
    else:
        u_plus = (1.0 / 0.41) * math.log(y_plus) + 5.2 - delta_b

    return max(0.0, u_tau * u_plus)


def generate_profile_points(
    layers: Sequence[RadialLayer],
    radius: float,
    u_mean: float,
    mdot_target: float,
    rho: float,
    nu: float,
    center1: float,
    center2: float,
    direction: str,
    min_points_per_ring: int,
    profile_model: str,
    power_index: float,
    u_tau_turbulent: float,
    roughness: float,
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

        if profile_model == "laminar":
            u_axial = laminar_velocity_at_radius(layer.r_center, radius, u_mean)
        elif profile_model == "turbulent-powerlaw":
            u_axial = powerlaw_velocity_at_radius(layer.r_center, radius, u_mean, power_index)
        elif profile_model == "turbulent-loglaw":
            u_axial = loglaw_velocity_at_radius(layer.r_center, radius, nu, u_tau_turbulent, roughness)
        else:
            raise ValueError(f"Unsupported profile model: {profile_model}")

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

    enforce_model_applicability(
        profile_model=args.profile_model,
        re=bulk["re"],
        allow_transitional=args.allow_transitional,
    )

    if args.profile_model == "laminar":
        wall = {
            "tau_w": bulk["tau_w_laminar"],
            "u_tau": bulk["u_tau_laminar"],
            "f_darcy": None,
            "model_used": "laminar reference",
            "rel_roughness": args.roughness / args.diameter if args.diameter > 0.0 else 0.0,
        }
    else:
        wall = compute_turbulent_wall_quantities(
            re=bulk["re"],
            u_mean=bulk["u_mean"],
            rho=args.rho,
            roughness=args.roughness,
            diameter=args.diameter,
            friction_model=args.friction_model,
        )

    layers, achieved_yplus, achieved_dr_wall, y_first_center_target = build_half_radius_layers(
        radius=radius,
        cells_diameter=args.cells_diameter,
        target_yplus=args.target_yplus,
        nu=bulk["nu"],
        u_tau=wall["u_tau"],
        growth_factor=args.growth_factor,
    )

    points, mdot_discrete, mdot_normalized = generate_profile_points(
        layers=layers,
        radius=radius,
        u_mean=bulk["u_mean"],
        mdot_target=args.mdot,
        rho=args.rho,
        nu=bulk["nu"],
        center1=args.center1,
        center2=args.center2,
        direction=args.direction,
        min_points_per_ring=args.min_points_per_ring,
        profile_model=args.profile_model,
        power_index=args.power_index,
        u_tau_turbulent=wall["u_tau"],
        roughness=args.roughness,
    )

    write_csv(points=points, direction=args.direction, output_path=args.output)

    component, transverse_labels, sign = component_name_and_labels(args.direction)

    print("=== Pipe profile summary ===")
    print(f"Output file                 : {args.output.resolve()}")
    print(f"Profile model               : {args.profile_model}")
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
    print(f"Flow regime (by Re)         : {classify_reynolds_number(bulk['re'])}")
    print(f"Wall shear stress [Pa]      : {wall['tau_w']:.12e}")
    print(f"Friction velocity [m/s]     : {wall['u_tau']:.12e}")

    if wall["f_darcy"] is not None:
        print(f"Darcy friction factor [-]   : {wall['f_darcy']:.12e}")
        print(f"Friction model              : {wall['model_used']}")
        print(f"Relative roughness [-]      : {wall['rel_roughness']:.12e}")
        print(f"Equivalent roughness [m]    : {args.roughness:.12e}")

    if args.profile_model == "turbulent-powerlaw":
        print(f"Power-law exponent n [-]    : {args.power_index:.6f}")

    print(f"Target first-center y [m]   : {y_first_center_target:.12e}")
    print(f"Achieved wall cell dr [m]   : {achieved_dr_wall:.12e}")
    print(f"Achieved first-cell y+ [-]  : {achieved_yplus:.6f}")
    print(f"Radial layers (half-radius) : {len(layers)}")
    print(f"Exported points             : {len(points)}")
    print(f"Discrete mdot before norm   : {mdot_discrete:.12e}")
    print(f"Discrete mdot after norm    : {mdot_normalized:.12e}")

    if args.profile_model == "laminar" and bulk["re"] >= 2300.0:
        print("WARNING: Reynolds number is above the classical laminar threshold.")
        print("         The Poiseuille profile may not be physically valid for this case.")


if __name__ == "__main__":
    main()
