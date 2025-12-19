from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class SimpleClassificationCounts:
    n_particles: int
    n_passing: int
    n_trapped: int
    n_prompt_lost: int
    n_trapped_classified_ideal: int
    n_trapped_classified_nonideal: int
    n_trapped_classified_jpar_good: int
    n_trapped_classified_jpar_bad: int
    n_trapped_classified_both_total: int

    @property
    def trapped_classified_total_ideal(self) -> int:
        return self.n_trapped_classified_ideal + self.n_trapped_classified_nonideal

    @property
    def trapped_classified_total_jpar(self) -> int:
        return self.n_trapped_classified_jpar_good + self.n_trapped_classified_jpar_bad

    @property
    def prompt_loss_fraction(self) -> float:
        if self.n_particles <= 0:
            return 0.0
        return self.n_prompt_lost / self.n_particles

    @property
    def ideal_fraction_trapped(self) -> float:
        denom = self.trapped_classified_total_ideal
        if denom <= 0:
            return 0.0
        return self.n_trapped_classified_ideal / denom

    @property
    def jpar_good_fraction_trapped(self) -> float:
        denom = self.trapped_classified_total_jpar
        if denom <= 0:
            return 0.0
        return self.n_trapped_classified_jpar_good / denom


@dataclass(frozen=True)
class SimpleClassificationResult:
    workdir: Path
    class_parts: np.ndarray
    times_lost: np.ndarray
    confined_fraction: np.ndarray | None
    counts: SimpleClassificationCounts
    trapped_confined_fraction: float
    ideal_fraction: float
    jpar_good_fraction: float
    score: float


@dataclass(frozen=True)
class SimpleLossResult:
    workdir: Path
    confined_fraction: np.ndarray
    times_lost: np.ndarray
    loss_fraction: float
    loss_fraction_vs_time: np.ndarray


def _fortran_value(value: Any) -> str:
    if isinstance(value, bool):
        return ".True." if value else ".False."
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        text = f"{float(value):.16g}"
        return text.replace("e", "d").replace("E", "d")
    if isinstance(value, (Path, os.PathLike)):
        return f"'{Path(value)}'"
    if isinstance(value, str):
        return f"'{value}'"
    raise TypeError(f"Unsupported Fortran namelist value type: {type(value)}")


def write_simple_in(path: Path, config: Mapping[str, Any]) -> None:
    lines = ["&config"]
    for key in sorted(config.keys()):
        lines.append(f"{key} = {_fortran_value(config[key])}")
    lines.append("/")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def default_fast_classification_config(
    *,
    ntestpart: int,
    trace_time_s: float = 1.0e-2,
    tcut_s: float = -1.0,
    multharm: int = 3,
    ns_s: int = 3,
    ns_tp: int = 3,
    nturns: int | None = None,
    class_plot: bool = False,
    cut_in_per: float = 0.5,
    fast_class: bool = True,
    deterministic: bool = True,
    notrace_passing: int = 1,
) -> dict[str, Any]:
    cfg: dict[str, Any] = {
        "ntestpart": int(ntestpart),
        "trace_time": float(trace_time_s),
        "tcut": float(tcut_s),
        "multharm": int(multharm),
        "ns_s": int(ns_s),
        "ns_tp": int(ns_tp),
        "class_plot": bool(class_plot),
        "cut_in_per": float(cut_in_per),
        "fast_class": bool(fast_class),
        "deterministic": bool(deterministic),
        "notrace_passing": int(notrace_passing),
    }
    if nturns is not None:
        cfg["nturns"] = int(nturns)
    return cfg


def _read_table(path: Path) -> np.ndarray:
    data = np.loadtxt(path, ndmin=2)
    return data


def parse_class_parts(path: Path) -> np.ndarray:
    data = _read_table(path)
    if data.shape[1] < 6:
        raise ValueError(f"Expected >=6 columns in {path}, got {data.shape[1]}")
    return data


def parse_times_lost(path: Path) -> np.ndarray:
    data = _read_table(path)
    if data.shape[1] < 10:
        raise ValueError(f"Expected >=10 columns in {path}, got {data.shape[1]}")
    return data


def parse_confined_fraction(path: Path) -> np.ndarray:
    data = _read_table(path)
    if data.shape[1] < 4:
        raise ValueError(f"Expected >=4 columns in {path}, got {data.shape[1]}")
    return data


def trapped_confined_fraction(confined_fraction: np.ndarray) -> float:
    """
    Return confined fraction of trapped particles at final time.

    SIMPLE writes `confined_fraction.dat` with columns:
      time, confpart_pass, confpart_trap, ntestpart

    `confpart_trap` is normalized by total particles, so to get the fraction of
    trapped particles still confined, normalize by the initial trapped fraction
    at time index 0.
    """
    if confined_fraction.size == 0:
        return 0.0
    conf_trap0 = float(confined_fraction[0, 2])
    conf_trap_end = float(confined_fraction[-1, 2])
    if conf_trap0 <= 0.0:
        return 0.0
    return conf_trap_end / conf_trap0


def compute_weighted_simple_score(
    *,
    trapped_confined_fraction: float,
    ideal_fraction: float,
    jpar_good_fraction: float,
    weights: Mapping[str, float] | None = None,
) -> float:
    """
    Combine SIMPLE proxy metrics into one scalar score.

    Metrics (maximize all):
      - confined_trapped: fraction of trapped particles confined at end time
      - ideal_trapped: fraction of classified trapped particles with ideal=1
      - jpar_trapped: fraction of classified trapped particles with ijpar=1

    Weights default to equal contributions.
    """
    if weights is None:
        weights = {
            "confined_trapped": 1.0,
            "ideal_trapped": 1.0,
            "jpar_trapped": 1.0,
        }

    w_conf = float(weights.get("confined_trapped", 0.0))
    w_ideal = float(weights.get("ideal_trapped", 0.0))
    w_jpar = float(weights.get("jpar_trapped", 0.0))
    w_sum = w_conf + w_ideal + w_jpar
    if w_sum == 0.0:
        return 0.0

    return (
        w_conf * float(trapped_confined_fraction)
        + w_ideal * float(ideal_fraction)
        + w_jpar * float(jpar_good_fraction)
    ) / w_sum


def _count_lines(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8", errors="replace") as f:
        return sum(1 for _ in f)


def compute_classification_metric(
    *,
    class_parts: np.ndarray,
    times_lost: np.ndarray,
    workdir: Path,
    confined_fraction: np.ndarray | None = None,
    weights: Mapping[str, float] | None = None,
    w_prompt: float = 0.0,
) -> tuple[float, SimpleClassificationCounts]:
    if class_parts.shape[0] != times_lost.shape[0]:
        raise ValueError(
            "class_parts and times_lost must have same number of rows: "
            f"{class_parts.shape[0]} vs {times_lost.shape[0]}"
        )

    n_particles = int(class_parts.shape[0])

    # times_lost columns: i, time, trap_par, s0, perp_inv0, zend(1..5)
    trap_par = times_lost[:, 2]
    trapped = trap_par > 0.0
    n_trapped = int(np.count_nonzero(trapped))
    n_passing = n_particles - n_trapped

    # class_parts columns: i, s0, perp_inv0, ijpar, ideal, misc
    ijpar = class_parts[:, 3].astype(int, copy=False)
    ideal = class_parts[:, 4].astype(int, copy=False)

    trapped_ijpar = ijpar[trapped]
    trapped_ideal = ideal[trapped]

    n_trapped_classified_ideal = int(np.count_nonzero(trapped_ideal == 1))
    n_trapped_classified_nonideal = int(np.count_nonzero(trapped_ideal == 2))

    n_trapped_classified_jpar_good = int(np.count_nonzero(trapped_ijpar == 1))
    n_trapped_classified_jpar_bad = int(np.count_nonzero(trapped_ijpar == 2))

    trapped_both_classified = trapped & (ijpar != 0) & (ideal != 0)
    n_trapped_classified_both_total = int(np.count_nonzero(trapped_both_classified))

    prompt_lost = _count_lines(workdir / "fort.10001") + _count_lines(workdir / "fort.10002")

    counts = SimpleClassificationCounts(
        n_particles=n_particles,
        n_passing=n_passing,
        n_trapped=n_trapped,
        n_prompt_lost=int(prompt_lost),
        n_trapped_classified_ideal=n_trapped_classified_ideal,
        n_trapped_classified_nonideal=n_trapped_classified_nonideal,
        n_trapped_classified_jpar_good=n_trapped_classified_jpar_good,
        n_trapped_classified_jpar_bad=n_trapped_classified_jpar_bad,
        n_trapped_classified_both_total=n_trapped_classified_both_total,
    )

    trapped_confined = (
        trapped_confined_fraction(confined_fraction) if confined_fraction is not None else 0.0
    )

    score = compute_weighted_simple_score(
        trapped_confined_fraction=trapped_confined,
        ideal_fraction=counts.ideal_fraction_trapped,
        jpar_good_fraction=counts.jpar_good_fraction_trapped,
        weights=weights,
    ) - float(w_prompt) * counts.prompt_loss_fraction
    return score, counts


def _find_simple_x(explicit: str | os.PathLike[str] | None) -> Path:
    if explicit is not None:
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    env = os.environ.get("SIMPLE_X") or os.environ.get("SIMPLE_EXE")
    if env:
        path = Path(env)
        if not path.exists():
            raise FileNotFoundError(path)
        return path

    which = shutil.which("simple.x")
    if which:
        return Path(which)

    raise FileNotFoundError(
        "Could not locate SIMPLE executable. Provide simple_executable=... "
        "or set SIMPLE_X/SIMPLE_EXE, or put simple.x on PATH."
    )


def run_simple_fast_classification(
    *,
    wout_path: str | os.PathLike[str],
    config: Mapping[str, Any],
    simple_executable: str | os.PathLike[str] | None = None,
    start_dat_path: str | os.PathLike[str] | None = None,
    keep_workdir: bool = False,
    timeout_s: float | None = 300.0,
    weights: Mapping[str, float] | None = None,
    w_prompt: float = 0.0,
) -> SimpleClassificationResult:
    simple_x = _find_simple_x(simple_executable)
    wout_path = Path(wout_path)

    if not wout_path.exists():
        raise FileNotFoundError(wout_path)

    temp_ctx: Any
    if keep_workdir:
        workdir = Path(tempfile.mkdtemp(prefix="neat_simple_"))
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="neat_simple_")
        workdir = Path(temp_ctx.name)

    try:
        shutil.copyfile(wout_path, workdir / "wout.nc")

        effective_config = dict(config)
        effective_config.setdefault("netcdffile", "wout.nc")

        if start_dat_path is not None:
            shutil.copyfile(start_dat_path, workdir / "start.dat")
            effective_config["startmode"] = 2

        # Ensure class_parts.dat is written even when `tcut < 0`.
        # SIMPLE writes class_parts.dat if (ntcut > 0) or (class_plot == True).
        effective_config.setdefault("class_plot", True)

        write_simple_in(workdir / "simple.in", effective_config)

        completed = subprocess.run(
            [str(simple_x)],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        (workdir / "simple_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"SIMPLE failed with exit code {completed.returncode}. "
                f"See {workdir / 'simple_stdout.txt'}"
            )

        class_parts_path = workdir / "class_parts.dat"
        times_lost_path = workdir / "times_lost.dat"
        confined_path = workdir / "confined_fraction.dat"
        if not class_parts_path.exists():
            raise FileNotFoundError(class_parts_path)
        if not times_lost_path.exists():
            raise FileNotFoundError(times_lost_path)

        class_parts = parse_class_parts(class_parts_path)
        times_lost = parse_times_lost(times_lost_path)
        confined = parse_confined_fraction(confined_path) if confined_path.exists() else None
        score, counts = compute_classification_metric(
            class_parts=class_parts,
            times_lost=times_lost,
            workdir=workdir,
            confined_fraction=confined,
            weights=weights,
            w_prompt=w_prompt,
        )
        return SimpleClassificationResult(
            workdir=workdir,
            class_parts=class_parts,
            times_lost=times_lost,
            confined_fraction=confined,
            counts=counts,
            trapped_confined_fraction=trapped_confined_fraction(confined) if confined is not None else 0.0,
            ideal_fraction=counts.ideal_fraction_trapped,
            jpar_good_fraction=counts.jpar_good_fraction_trapped,
            score=score,
        )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()


def run_simple_loss(
    *,
    wout_path: str | os.PathLike[str],
    config: Mapping[str, Any],
    simple_executable: str | os.PathLike[str] | None = None,
    start_dat_path: str | os.PathLike[str] | None = None,
    keep_workdir: bool = False,
    timeout_s: float | None = 300.0,
) -> SimpleLossResult:
    simple_x = _find_simple_x(simple_executable)
    wout_path = Path(wout_path)

    if not wout_path.exists():
        raise FileNotFoundError(wout_path)

    temp_ctx: Any
    if keep_workdir:
        workdir = Path(tempfile.mkdtemp(prefix="neat_simple_"))
        temp_ctx = None
    else:
        temp_ctx = tempfile.TemporaryDirectory(prefix="neat_simple_")
        workdir = Path(temp_ctx.name)

    try:
        shutil.copyfile(wout_path, workdir / "wout.nc")
        effective_config = dict(config)
        effective_config.setdefault("netcdffile", "wout.nc")

        if start_dat_path is not None:
            shutil.copyfile(start_dat_path, workdir / "start.dat")
            effective_config["startmode"] = 2

        write_simple_in(workdir / "simple.in", effective_config)

        completed = subprocess.run(
            [str(simple_x)],
            cwd=str(workdir),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        (workdir / "simple_stdout.txt").write_text(completed.stdout, encoding="utf-8")
        if completed.returncode != 0:
            raise RuntimeError(
                f"SIMPLE failed with exit code {completed.returncode}. "
                f"See {workdir / 'simple_stdout.txt'}"
            )

        confined_path = workdir / "confined_fraction.dat"
        times_lost_path = workdir / "times_lost.dat"
        if not confined_path.exists():
            raise FileNotFoundError(confined_path)
        if not times_lost_path.exists():
            raise FileNotFoundError(times_lost_path)

        confined = parse_confined_fraction(confined_path)
        times_lost = parse_times_lost(times_lost_path)

        # Confined fraction columns: time, confpass, conftrap, ntestpart
        loss_fraction_vs_time = 1.0 - (confined[:, 1] + confined[:, 2])
        loss_fraction = float(loss_fraction_vs_time[-1]) if loss_fraction_vs_time.size else 0.0

        return SimpleLossResult(
            workdir=workdir,
            confined_fraction=confined,
            times_lost=times_lost,
            loss_fraction=loss_fraction,
            loss_fraction_vs_time=loss_fraction_vs_time,
        )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()
