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
    counts: SimpleClassificationCounts
    score: float


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
    trace_time_s: float,
    class_plot: bool = True,
    cut_in_per: float = 0.0,
    deterministic: bool = True,
    notrace_passing: int = 1,
) -> dict[str, Any]:
    return {
        "ntestpart": int(ntestpart),
        "trace_time": float(trace_time_s),
        "tcut": -1.0,
        "class_plot": bool(class_plot),
        "cut_in_per": float(cut_in_per),
        "fast_class": True,
        "deterministic": bool(deterministic),
        "notrace_passing": int(notrace_passing),
    }


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
    w_ideal: float = 1.0,
    w_jpar: float = 1.0,
    w_prompt: float = 1.0,
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
    )

    score = (
        w_ideal * counts.ideal_fraction_trapped
        + w_jpar * counts.jpar_good_fraction_trapped
        - w_prompt * counts.prompt_loss_fraction
    )
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
    w_ideal: float = 1.0,
    w_jpar: float = 1.0,
    w_prompt: float = 1.0,
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
        if not class_parts_path.exists():
            raise FileNotFoundError(class_parts_path)
        if not times_lost_path.exists():
            raise FileNotFoundError(times_lost_path)

        class_parts = parse_class_parts(class_parts_path)
        times_lost = parse_times_lost(times_lost_path)
        score, counts = compute_classification_metric(
            class_parts=class_parts,
            times_lost=times_lost,
            workdir=workdir,
            w_ideal=w_ideal,
            w_jpar=w_jpar,
            w_prompt=w_prompt,
        )
        return SimpleClassificationResult(
            workdir=workdir,
            class_parts=class_parts,
            times_lost=times_lost,
            counts=counts,
            score=score,
        )
    finally:
        if temp_ctx is not None:
            temp_ctx.cleanup()

