from __future__ import annotations
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pathlib import Path
import pandas as pd


class UniaxialSakinoSunConcrete:
    """
    Uniaxial concrete inside square steel tubes per Sakino & Sun (1994).
    Units: MPa for stress (i.e., N/mm^2) and strain is unitless.

    σ(ε) = fcc' * [V·(ε/εcc) + (W−1)·(ε/εcc)^2] / [1 + (V−2)·(ε/εcc) + W·(ε/εcc)^2]
      with:
        V = Ec·εcc / fcc'
        W = 1.5 − 0.0171·fc' + 2.39·√σre
        εcc = 0.94e−3·(fc')^0.25
        fcc' = fc' (original formulation)
        σre = 2·(b/t − 1)·Fy / (b/t − 2)^3

    Parameters
    ----------
    name : str
        Material name.
    fc_prime : float
        Unconfined concrete strength f'c (MPa). Used as f'cc in the original model.
    Fy : float
        Steel tube yield stress (MPa).
    b_over_t : float
        Width-to-thickness ratio b/t (> 2).
    Ec : Optional[float], default=None
        Concrete modulus (MPa). If None, Ec = 4700*sqrt(fc').
    eps_max : float, default=0.04
        Max strain to trace.
    n_points : int, default=200
        Discretization points of the curve.
    plot : bool, default=False
        Plot automatically after creation.
    color : str, default='k'
        Matplotlib color.
    marker : Optional[str]
        Matplotlib marker.
    **plot_kwargs : Any
        Extra Matplotlib kwargs passed to `plot()`.

    Attributes
    ----------
    strain : np.ndarray
        Strain array (size n_points).
    stress : np.ndarray
        Stress array (size n_points, MPa).
    material_law : np.ndarray
        2×N array [strain_row; stress_row].
    eps_cc : float
        Peak strain of confined concrete.
    fcc_prime : float
        Confined strength (equal to fc_prime in the original model).
    sigma_re : float
        Effective confining stress (MPa).
    V, W : float
        Model shape parameters.
    """

    def __init__(
        self,
        *,
        name: str,
        fc_prime: float,
        Fy: float,
        b_over_t: float,
        Ec: Optional[float] = None,
        eps_max: float = 0.04,
        n_points: int = 200,
        plot: bool = False,
        color: str = "k",
        marker: Optional[str] = None,
        **plot_kwargs: Any,
    ) -> None:
        self.name = name
        self.fc_prime = float(fc_prime)
        self.Fy = float(Fy)
        self.b_over_t = float(b_over_t)
        self.eps_max = float(eps_max)
        self.n_points = int(n_points)
        self.plot_flag = plot
        self.color = color
        self.marker = marker
        self.plot_kwargs = dict(plot_kwargs)

        self._validate_inputs()

        # Elastic modulus
        if Ec is None:
            self.Ec = 4700.0 * (self.fc_prime ** 0.5)  # MPa
        else:
            if Ec <= 0.0:
                raise ValueError("Ec must be positive if provided.")
            self.Ec = float(Ec)

        # Peak strain & strength (original model takes fcc' = fc')
        self.eps_cc = 0.94e-3 * (self.fc_prime ** 0.25)
        self.fcc_prime = self.fc_prime

        # Confining stress and shape params
        self.sigma_re = self._compute_sigma_re(self.b_over_t, self.Fy)
        self.V = self.Ec * self.eps_cc / self.fcc_prime
        self.W = 1.5 - 0.0171 * self.fc_prime + 2.39 * np.sqrt(max(self.sigma_re, 0.0))

        # Build curve
        self.strain, self.stress, self.material_law = self._build_constitutive()

        if self.plot_flag:
            self.plot(**self.plot_kwargs)

    # ------------------------------- Validation -------------------------------

    def _validate_inputs(self) -> None:
        if self.fc_prime <= 0.0:
            raise ValueError("fc_prime must be positive.")
        if self.Fy <= 0.0:
            raise ValueError("Fy must be positive.")
        if self.b_over_t <= 2.0:
            raise ValueError("b_over_t must be > 2.0 to avoid division by zero in σre.")
        if self.eps_max <= 0.0:
            raise ValueError("eps_max must be positive.")
        if self.n_points < 2:
            raise ValueError("n_points must be >= 2.")

    # ------------------------------- Internals --------------------------------

    @staticmethod
    def _compute_sigma_re(b_over_t: float, Fy: float) -> float:
        # σre = 2·(b/t − 1)·Fy / (b/t − 2)^3
        denom = (b_over_t - 2.0) ** 3
        return 2.0 * (b_over_t - 1.0) * Fy / denom

    def _generate_constitutive_points(self) -> Tuple[np.ndarray, np.ndarray]:
        eps = np.linspace(0.0, self.eps_max, self.n_points, dtype=float)
        ratio = eps / self.eps_cc
        num = self.V * ratio + (self.W - 1.0) * (ratio ** 2)
        den = 1.0 + (self.V - 2.0) * ratio + self.W * (ratio ** 2)
        # Guard against extremely small denominators
        den = np.where(np.abs(den) < np.finfo(float).eps, np.finfo(float).eps, den)
        sig = self.fcc_prime * (num / den)
        sig = np.clip(sig, 0.0, None)  # non-negative compressive stress (MPa)
        return eps, sig

    def _build_constitutive(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        es, fs = self._generate_constitutive_points()
        return es, fs, np.vstack((es, fs))

    # ----------------------------------- API ----------------------------------

    def get_constitutive_law(self) -> np.ndarray:
        """Return 2×N array [strain_row; stress_row]."""
        return self.material_law

    def plot(
        self,
        ax: Optional[Axes] = None,
        label: Optional[str] = None,
        **kwargs: Any,
    ) -> Axes:
        """Plot the constitutive law."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
            ax.set_title("Confined Concrete – Sakino & Sun (1994)")

        label = label or f"Confined: {self.name}"
        defaults = dict(color=self.color, linewidth=1.5, marker=self.marker)
        defaults.update(kwargs)

        ax.plot(self.strain, self.stress, label=label, **defaults)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress (MPa)")
        ax.legend()
        return ax

    # ------------------------------------------------------------
    # Export methods (match your UniaxialUnconfinedConcrete style)
    # ------------------------------------------------------------

    def to_csv(self, path: Optional[str] = None) -> Path:
        """
        Export constitutive law to CSV. If path is None → `<cwd>/<name>_law.csv`.
        """
        path = Path(path) if path is not None else (Path.cwd() / f"{self.name}_law.csv")
        df = pd.DataFrame({"Strain": self.strain, "Stress": self.stress})
        df.to_csv(path, index=False)
        print(f"[INFO] CSV exported successfully: {path}")
        return path

    def to_excel(self, path: Optional[str] = None, sheet_name: str = "ConstitutiveLaw") -> Path:
        """
        Export constitutive law to XLSX. If path is None → `<cwd>/<name>_law.xlsx`.
        """
        path = Path(path) if path is not None else (Path.cwd() / f"{self.name}_law.xlsx")
        df = pd.DataFrame({"Strain": self.strain, "Stress": self.stress})
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[INFO] Excel exported successfully: {path}")
        return path

    # ------------------------------- Representation ---------------------------

    def __repr__(self) -> str:
        return (f"UniaxialSakinoSunConcrete(name='{self.name}', fc_prime={self.fc_prime}, "
                f"Fy={self.Fy}, b_over_t={self.b_over_t})")

    def __str__(self) -> str:
        return self.name
