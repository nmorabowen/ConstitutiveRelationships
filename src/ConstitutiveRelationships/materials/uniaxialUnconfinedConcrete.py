from __future__ import annotations
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes

from pathlib import Path
import pandas as pd


class UniaxialUnconfinedConcrete:
    """
    Uniaxial unconfined concrete (Mander-type ascending + linear softening).
    Units system SI [N - mm]

    Parameters
    ----------
    name : str
        Material name.
    fco : float
        Compressive strength (same units as stress output).
    eco : float, default=0.002
        Strain at compressive strength (peak).
    ec_sprall : float, default=0.006
        Ultimate strain (end of softening).
    delta : int, default=15
        Number of samples for the full curve.
    plot : bool, default=False
        If True, call `plot()` after building the curve.
    color : str, default='k'
        Default plot color.
    marker : Optional[str], default=None
        Default plot marker.
    **plot_kwargs : Any
        Extra matplotlib kwargs forwarded to `plot()`.

    Attributes
    ----------
    strain : np.ndarray
        Strain array.
    stress : np.ndarray
        Stress array.
    material_law : np.ndarray
        2×N array [strain_row; stress_row].
    Esec : float
        Secant modulus at peak = fco / eco.
    Ec : float
        Elastic modulus per sqrt(f'c) rule.
    r : float
        Mander parameter = Ec / (Ec - Esec).
    """

    def __init__(
        self,
        *,
        name: str,
        fco: float,
        eco: float = 0.002,
        ec_sprall: float = 0.006,
        delta: int = 15,
        plot: bool = False,
        color: str = "k",
        marker: Optional[str] = None,
        **plot_kwargs: Any,
    ) -> None:
        self.name = name
        self.fco = float(fco)
        self.eco = float(eco)
        self.ec_sprall = float(ec_sprall)
        self.delta = int(delta)
        self.plot_flag = plot
        self.color = color
        self.marker = marker
        self.plot_kwargs = plot_kwargs

        self._validate_inputs()
        self._compute_moduli()
        self.strain, self.stress, self.material_law = self._build_constitutive(self.delta)

        if self.plot_flag:
            self.plot(**self.plot_kwargs)

    # ------------------------------- Validation -------------------------------

    def _validate_inputs(self) -> None:
        if self.fco <= 0.0:
            raise ValueError("fco must be positive.")
        if not (0.0 < self.eco < self.ec_sprall):
            raise ValueError("Require 0 < eco < ec_sprall.")
        if self.delta < 3:
            raise ValueError("delta must be >= 3.")

    # ------------------------------- Moduli -----------------------------------

    def _compute_moduli(self) -> None:
        self.Esec = self.fco / self.eco
        self.Ec = (5000.0 * (self.fco) ** 0.5)
        # Guard against numerical issues if Ec ~ Esec
        denom = max(self.Ec - self.Esec, np.finfo(float).eps)
        self.r = self.Ec / denom

    # -------------------------- Constitutive builder --------------------------

    def _build_constitutive(
        self, delta: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build stress–strain curve:
          - Mander ascending up to 2*eco,
          - linear softening from (2*eco, fc_2eco) → (ec_sprall, 0).
        """
        es = np.linspace(0.0, self.ec_sprall, delta, dtype=float)
        two_eco = 2.0 * self.eco

        # Ascending branch (Mander)
        x = es / self.eco
        with np.errstate(divide="ignore", invalid="ignore"):
            fc_mander = (self.fco * x * self.r) / (self.r - 1.0 + np.power(x, self.r))

        # Stress at 2*eco (anchor for softening)
        x2 = two_eco / self.eco  # = 2.0
        fc_2eco = (self.fco * x2 * self.r) / (self.r - 1.0 + np.power(x2, self.r))

        # Linear softening slope (to zero at ec_sprall)
        m = fc_2eco / (two_eco - self.ec_sprall)

        # Piecewise combine
        fs = np.where(
            es <= two_eco,
            np.maximum(fc_mander, 0.0),
            m * (es - self.ec_sprall),
        )
        fs = np.clip(fs, 0.0, None)  # enforce non-negative

        constitutive = np.vstack((es, fs))
        return es, fs, constitutive

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
            ax.set_title("Unconfined Concrete – Mander + linear softening")

        label = label or f"Unconfined: {self.name}"
        defaults = dict(color=self.color, linewidth=1.5, marker=self.marker)
        defaults.update(kwargs)

        ax.plot(self.strain, self.stress, label=label, **defaults)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress")
        ax.legend()
        return ax

    # ------------------------------------------------------------
    # Export methods
    # ------------------------------------------------------------

    def to_csv(self, path: Optional[str] = None) -> Path:
        """
        Export constitutive law to a CSV file.

        Parameters
        ----------
        path : str or None, default=None
            Output file path. If None, saves as "<name>_law.csv" in the current working directory.

        Returns
        -------
        Path
            The path of the exported file.
        """
        if path is None:
            path = Path.cwd() / f"{self.name}_law.csv"
        else:
            path = Path(path)

        df = pd.DataFrame({"Strain": self.strain, "Stress": self.stress})
        df.to_csv(path, index=False)
        print(f"[INFO] CSV exported successfully: {path}")
        return path

    def to_excel(self, path: Optional[str] = None, sheet_name: str = "ConstitutiveLaw") -> Path:
        """
        Export constitutive law to an Excel (.xlsx) file.

        Parameters
        ----------
        path : str or None, default=None
            Output file path. If None, saves as "<name>_law.xlsx" in the current working directory.
        sheet_name : str, default='ConstitutiveLaw'
            Sheet name for the Excel file.

        Returns
        -------
        Path
            The path of the exported file.
        """
        if path is None:
            path = Path.cwd() / f"{self.name}_law.xlsx"
        else:
            path = Path(path)

        df = pd.DataFrame({"Strain": self.strain, "Stress": self.stress})
        with pd.ExcelWriter(path, engine="openpyxl") as writer:
            df.to_excel(writer, sheet_name=sheet_name, index=False)
        print(f"[INFO] Excel exported successfully: {path}")
        return path


    # ------------------------------- Representation ---------------------------

    def __repr__(self) -> str:
        return (f"UniaxialUnconfinedConcrete(name='{self.name}', fco={self.fco}, "
                f"eco={self.eco}, ec_sprall={self.ec_sprall})")

    def __str__(self) -> str:
        return self.name

