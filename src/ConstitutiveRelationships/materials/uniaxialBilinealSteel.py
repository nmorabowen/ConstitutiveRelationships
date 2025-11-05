from __future__ import annotations
from typing import Optional, Tuple, Any
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import pandas as pd
from pathlib import Path



class UniaxialBilinealSteel:
    """
    Uniaxial bilinear steel constitutive law (elastic + hardening branch).

    Parameters
    ----------
    name : str
        Material name.
    fy : float
        Yield strength.
    fsu : float
        Ultimate strength.
    esh : float, default=0.008
        Strain at start of hardening.
    esu : float, default=0.12
        Ultimate strain.
    Es : float, default=200000.0
        Elastic modulus.
    Esh : float, default=7000.0
        Hardening modulus.
    plot : bool, default=False
        Plot automatically after creation.
    delta : int, default=15
        Number of interpolation points in hardening branch.
    color : str, default='k'
        Matplotlib color.
    marker : Optional[str]
        Matplotlib marker.
    **plot_kwargs : Any
        Additional keyword arguments passed to `plot()` (e.g. linewidth, linestyle).

    Attributes
    ----------
    ey : float
        Yield strain = fy / Es.
    strain : np.ndarray
        Strain array.
    stress : np.ndarray
        Stress array.
    material_law : np.ndarray
        2×N array [strain_row; stress_row].
    """

    def __init__(
        self,
        *,
        name: str,
        fy: float,
        fsu: float,
        esh: float = 0.008,
        esu: float = 0.12,
        Es: float = 200_000.0,
        Esh: float = 7_000.0,
        plot: bool = False,
        delta: int = 15,
        color: str = "k",
        marker: Optional[str] = None,
        **plot_kwargs: Any,
    ) -> None:
        # Store parameters
        self.name = name
        self.fy = float(fy)
        self.fsu = float(fsu)
        self.esh = float(esh)
        self.esu = float(esu)
        self.Es = float(Es)
        self.Esh = float(Esh)
        self.plot_flag = plot
        self.delta = int(delta)
        self.color = color
        self.marker = marker
        self.plot_kwargs = plot_kwargs

        # Validate and compute
        self._validate_inputs()
        self.ey = self.fy / self.Es
        self.strain, self.stress, self.material_law = self._build_constitutive(self.delta)

        if self.plot_flag:
            self.plot(**self.plot_kwargs)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------

    def _validate_inputs(self) -> None:
        if any(v <= 0.0 for v in (self.fy, self.fsu, self.Es, self.Esh)):
            raise ValueError("Strengths and moduli must be positive.")
        if not (0.0 < self.esh < self.esu):
            raise ValueError("esh must be > 0 and < esu.")
        if self.fsu <= self.fy:
            raise ValueError("fsu must be greater than fy (hardening target).")
        if self.delta < 2:
            raise ValueError("delta must be >= 2.")

    # ------------------------------------------------------------
    # Core constitutive function
    # ------------------------------------------------------------

    def _build_constitutive(
        self, delta: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute stress–strain table."""
        # Elastic branch
        es_array = np.array([0.0, self.ey, self.esh], dtype=float)
        fs_array = np.array([0.0, self.fy, self.fy], dtype=float)

        # Hardening branch
        es_sh = np.linspace(self.esh, self.esu, delta, dtype=float)
        P = self.Esh * ((self.esu - self.esh) / (self.fsu - self.fy))

        ratio = (self.esu - es_sh) / (self.esu - self.esh)
        fs_sh = self.fsu + (self.fy - self.fsu) * (ratio**P)

        # Merge and return
        es_full = np.concatenate((es_array[:2], es_sh))
        fs_full = np.concatenate((fs_array[:2], fs_sh))
        constitutive = np.vstack((es_full, fs_full))
        return es_full, fs_full, constitutive

    # ------------------------------------------------------------
    # API
    # ------------------------------------------------------------

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
            ax.set_title("Constitutive Law of Uniaxial Bilinear Steel")

        label = label or f"Steel: {self.name}"
        plot_kwargs = {**dict(color=self.color, linewidth=1.5, marker=self.marker), **kwargs}
        ax.plot(self.strain, self.stress, label=label, **plot_kwargs)
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


    # ------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------

    def __repr__(self) -> str:
        return f"UniaxialBilinealSteel(name='{self.name}', fy={self.fy}, fsu={self.fsu})"

    def __str__(self) -> str:
        return self.name
