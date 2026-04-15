# LOGOS-FLOWGEN - pipe inlet velocity generator (LOGOS CFD Software)

This folder contains small Python utilities to **generate** circular pipe inlet velocity-profile tables and to **inspect/plot** existing profile tables.

## What’s included

- **`pipe_profile_generator_LOGOS.py`**: main generator
  - Profile models:
    - `laminar` (Poiseuille)
    - `turbulent-powerlaw` (engineering power-law)
    - `turbulent-loglaw` (wall-units/log-law with friction factor and optional roughness)
  - Flow can be specified by **mass flow rate** (`--mdot`) **or** **Reynolds number** (`--re`)
  - Exports a CSV with columns compatible with coordinate-based inlet profile workflows:
    - `X / -X` → `U, YC, ZC`
    - `Y / -Y` → `V, XC, ZC`
    - `Z / -Z` → `W, XC, YC`

- **`read_velocity_profile.py`**: reader + normalizer + plotter
  - Auto-detects common delimiters (`; , \\t`)
  - Recognizes typical column names (including “Velocity magnitude” exports)
  - Produces a **PNG plot automatically next to the input file**
  - Can export a normalized CSV (`velocity,c1,c2,axis_pos,radius`)

- **`profiles_cli.py`**: small menu wrapper (interactive)
  - Option 1: run generator interactively
  - Option 2: read/plot a profile interactively (prompts for input path if omitted)

- **`laminar_pipe_profile_LOGOS.py`**: legacy laminar-only generator  
  Kept for backwards compatibility; the main generator above fully covers its functionality and more.

## Requirements

Python **3.10+** recommended.

Install dependencies:

```bash
python -m pip install numpy pandas matplotlib
```

## Quick start

### 1) Generate a profile (interactive)

```bash
python pipe_profile_generator_LOGOS.py --interactive
```

You will be prompted for:

- `profile-model` (`laminar`, `turbulent-powerlaw`, `turbulent-loglaw`)
- `D, rho, mu`
- flow spec: `mdot` **or** `Re`
- direction (`X, -X, Y, -Y, Z, -Z`)
- radial resolution & clustering (`cells-diameter`, `target-yplus`, `growth-factor`)
- optional: center coordinates and angular resolution
- output CSV path

### 2) Generate a profile (CLI)

Laminar example:

```bash
python pipe_profile_generator_LOGOS.py ^
  --profile-model laminar ^
  --re 1000 ^
  --diameter 0.05 ^
  --rho 1.2 ^
  --mu 0.001 ^
  --direction X ^
  --cells-diameter 40 ^
  --target-yplus 1 ^
  --growth-factor 1.1 ^
  --output pipe_profile.csv
```

Turbulent log-law example (with roughness):

```bash
python pipe_profile_generator_LOGOS.py ^
  --profile-model turbulent-loglaw ^
  --re 100000 ^
  --diameter 0.05 ^
  --rho 1.2 ^
  --mu 0.001 ^
  --roughness 1e-5 ^
  --friction-model auto ^
  --direction X ^
  --cells-diameter 60 ^
  --target-yplus 1 ^
  --growth-factor 1.15 ^
  --output pipe_profile.csv
```

## Inspect / plot an existing profile

### Read and auto-save a PNG next to the input CSV

```bash
python read_velocity_profile.py "path\\to\\profile.csv"
```

Notes:
- If you pass only a filename (e.g. `profile.csv`), the script will try the **current working directory** first, then the **script folder**.
- The PNG is saved as `profile_plot.png` next to the input file.
- Colorbar limits (`min/max`) are taken from the CSV data.

### Also show the plot window

```bash
python read_velocity_profile.py "path\\to\\profile.csv" --plot
```

### Export a normalized CSV

```bash
python read_velocity_profile.py "path\\to\\profile.csv" --export-normalized normalized.csv
```

## Menu wrapper

```bash
python profiles_cli.py
```

## Conventions / notes

- **Mass-flow normalization**: generated profiles are scaled so that the discrete integral matches the requested `mdot` exactly.
- **`y+` usage**: used as an engineering estimate to set near-wall spacing; the requested combination `{target y+, cells, growth factor, radius}` can be over-constrained. The scripts report the **achieved** near-wall spacing / `y+`.
- **Transitional regime**: turbulent models reject `2300 <= Re < 4000` by default unless explicitly allowed.


