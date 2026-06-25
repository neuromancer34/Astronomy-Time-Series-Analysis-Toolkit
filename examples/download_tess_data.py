"""Download a real TESS light curve and save it as a CSV astrotime can load.

This script is intentionally outside the astrotime/ package -- it's a
convenience for getting real demo data, not a core analysis capability.
It depends on `lightkurve`, which is *not* a dependency of the astrotime
package itself (see pyproject.toml); install it separately if you want
to run this script:

    uv pip install lightkurve

Usage:
    python examples/download_tess_data.py --target "TIC 307210830" --out examples/data/tic307210830.csv
"""

from __future__ import annotations

import argparse
from pathlib import Path


def download_and_save(target: str, output_path: str | Path, mission: str = "TESS") -> None:
    """Search for and download a light curve, saving it in astrotime's CSV format."""
    try:
        import lightkurve as lk
    except ImportError as exc:
        raise ImportError(
            "This script requires lightkurve, which is not installed by default. "
            "Run: uv pip install lightkurve"
        ) from exc

    print(f"Searching for {target} ({mission})...")
    search_result = lk.search_lightcurve(target, mission=mission)
    if len(search_result) == 0:
        raise ValueError(f"No light curves found for target {target!r}")

    print(f"Found {len(search_result)} result(s); downloading the first...")
    lc = search_result[0].download()

    # Remove non-finite points up front -- astrotime.loader.load_csv will
    # also drop NaNs, but doing it here keeps the saved CSV itself clean.
    lc = lc.remove_nans()

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = lc.to_pandas().reset_index()
    out = df[["time", "flux", "flux_err"]] if "flux_err" in df.columns else df[["time", "flux"]]
    out.to_csv(output_path, index=False)

    print(f"Saved {len(out)} points to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target", required=True, help='e.g. "TIC 307210830"')
    parser.add_argument("--out", required=True, help="Output CSV path")
    parser.add_argument("--mission", default="TESS", help="Mission name (default: TESS)")
    args = parser.parse_args()

    download_and_save(args.target, args.out, mission=args.mission)


if __name__ == "__main__":
    main()
