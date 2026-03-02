from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CACHE_DIR = ROOT / ".cache"
CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_DIR))

# Keep Matplotlib cache local to the repository to avoid permission warnings.
_MPLCONFIGDIR = ROOT / ".mplconfig"
_MPLCONFIGDIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_MPLCONFIGDIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from nonlinear_bvp import solve_bvp


def publication_style() -> None:
    plt.rcParams.update(
        {
            "figure.figsize": (5.2, 3.4),
            "figure.dpi": 150,
            "savefig.dpi": 600,
            "font.family": "STIXGeneral",
            "mathtext.fontset": "stix",
            "font.size": 12,
            "axes.labelsize": 13,
            "axes.titlesize": 13,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.major.size": 4.5,
            "ytick.major.size": 4.5,
            "xtick.major.width": 0.9,
            "ytick.major.width": 0.9,
            "legend.frameon": False,
        }
    )


def plot_solution(output_path: Path, num_interior_points: int = 200) -> Path:
    publication_style()

    result = solve_bvp(num_interior_points=num_interior_points, tolerance=1e-11, max_iterations=20)
    if not result.converged:
        raise RuntimeError("Newton solver did not converge")

    x = np.linspace(0.0, 1.0, num_interior_points + 2)
    u = np.zeros(num_interior_points + 2)
    u[1:-1] = result.solution

    fig, ax = plt.subplots()
    ax.plot(x, u, color="black", linewidth=2.0, label="Finite-difference solution")
    ax.fill_between(x, 0.0, u, color="#e0e0e0")
    ax.set_xlim(0.0, 1.0)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$u(x)$")
    ax.set_title(r"Solution of $-u'' + u^3 = 1$ with $u(0)=u(1)=0$")
    ax.legend(loc="upper right")
    ax.grid(color="#cfcfcf", linewidth=0.5)

    fig.tight_layout()
    fig.savefig(output_path, format="eps", bbox_inches="tight")
    plt.close(fig)
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot the computed nonlinear BVP solution.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("nonlinear_bvp_solution.eps"),
        help="Output EPS filename.",
    )
    parser.add_argument(
        "--num-interior-points",
        type=int,
        default=200,
        help="Number of interior grid points used in the finite-difference discretization.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_path = plot_solution(args.output, num_interior_points=args.num_interior_points)
    print(output_path.resolve())


if __name__ == "__main__":
    main()
