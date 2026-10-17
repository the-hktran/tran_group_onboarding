"""Module 04 Helper: Load Precomputed .npz AO Integrals or Compute via PySCF on the Fly."""
from __future__ import annotations
import importlib.util
from pathlib import Path
import sys
import numpy as np

try:
    from generate_ao_integrals import (
        build_heh_plus_sto3g_arrays,
        build_h2o_sto3g_arrays,
        save_integral_archives,
    )
except ModuleNotFoundError:
    _spec = importlib.util.spec_from_file_location(
        "generate_ao_integrals", Path(__file__).with_name("generate_ao_integrals.py")
    )
    _mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
    sys.modules[_spec.name] = _mod
    _spec.loader.exec_module(_mod)  # type: ignore[union-attr]
    build_heh_plus_sto3g_arrays = _mod.build_heh_plus_sto3g_arrays
    build_h2o_sto3g_arrays = _mod.build_h2o_sto3g_arrays
    save_integral_archives = _mod.save_integral_archives


def load_system_integrals(
    system: str = "heh_plus_sto3g",
    data_dir: str | Path = ".",
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, int]:
    """Return (S, T, V, eri, e_nuc, n_elec) for 'heh_plus_sto3g' or 'h2o_sto3g'."""
    base = Path(data_dir)
    npz_file = base / f"{system}_integrals.npz"
    if not npz_file.exists():
        save_integral_archives(base)

    if npz_file.exists():
        data = np.load(npz_file)
        return (
            data["S"],
            data["T"],
            data["V"],
            data["eri"],
            float(data["e_nuc"]),
            int(data["n_elec"]),
        )

    raw = (
        build_heh_plus_sto3g_arrays()
        if system == "heh_plus_sto3g"
        else build_h2o_sto3g_arrays()
    )
    return (
        raw["S"],
        raw["T"],
        raw["V"],
        raw["eri"],
        float(raw["e_nuc"]),
        int(raw["n_elec"]),
    )