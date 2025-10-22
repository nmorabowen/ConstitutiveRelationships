from __future__ import annotations
from typing import Optional, Tuple, Any

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from math import pi

from pathlib import Path
import pandas as pd


class UniaxialConfinedConcrete:
    """
    Confined concrete under uniaxial compression.

    The model:
      1) Calcula parámetros de confinamiento (razones de estribos, ke, presiones efectivas).
      2) Interpola una tabla multiaxial para obtener fcc_ratio (→ fcc) según fl1/fl2.
      3) Define ecc y ecu (deformaciones de pico y última).
      4) Construye la ley constitutiva (Mander para ascenso, suavizada) con delta puntos.
      5) Unidad SI [N - mm].

    Parameters (all keyword-only)
    -----------------------------
    name : str
        Nombre del material.
    fco : float
        Resistencia a compresión no confinada (misma unidad de esfuerzo de salida).
    eco : float
        Deformación en f'c (sin confinamiento).
    b, h : float
        Dimensiones de la sección rectangular total.
    rec : float
        Recubrimiento de hormigón (cara exterior a centro de estribo).
    num_var_b, num_var_h : int
        Nº de barras longitudinales en cada dirección (cuenta por línea).
    phi_longitudinal : float
        Diámetro de barras longitudinales.
    num_est_perpendicular_b, num_est_perpendicular_h : int
        Nº de ramas de estribo perpendiculares a b y h, respectivamente.
    phi_estribo : float
        Diámetro de estribo.
    s : float
        Separación de estribos (paso).
    fye : float
        Fy del acero de estribo.
    esu_estribo : float
        Deformación última del acero de estribo.
    delta : int, default=50
        Nº de muestras para la curva constitutiva final.
    plot : bool, default=False
        Si True, llama a `plot()` al final.
    color : str, default='k'
        Color por defecto para la curva.
    marker : Optional[str], default=None
        Marcador por defecto para la curva.
    **plot_kwargs : Any
        Argumentos extra para Matplotlib (linestyle, linewidth, etc.).

    Public attributes (principales)
    -------------------------------
    bc, hc : float
        Dimensiones confinadas (libres menos diámetro de estribo).
    Ac : float
        Área confinada efectiva base.
    As : float
        Área total de barras longitudinales (en el perímetro confinado).
    rho_confinado : float
        ρ = As / Ac.
    ke : float
        Coeficiente de efectividad del confinamiento.
    fl_perpendicular_b_efectivo, fl_perpendicular_h_efectivo : float
        Presiones laterales efectivas.
    fcc_ratio, fcc : float
        Incremento por confinamiento y resistencia confinada resultante.
    ecc : float
        Deformación de pico del hormigón confinado.
    ecu : float
        Deformación última (rotura) del hormigón confinado.
    Esec, Ec, r : float
        Módulo secante (fcc/ecc), módulo elástico, y parámetro de Mander r = Ec/(Ec-Esec).
    strain, stress : np.ndarray
        Curva ε–σ generada (monótona).
    material_law : np.ndarray
        Matriz 2×N con [strain_row; stress_row].
    """

    def __init__(
        self,
        *,
        name: str,
        fco: float,
        eco: float,
        b: float,
        h: float,
        rec: float,
        num_var_b: int,
        num_var_h: int,
        phi_longitudinal: float,
        num_est_perpendicular_b: int,
        num_est_perpendicular_h: int,
        phi_estribo: float,
        s: float,
        fye: float,
        esu_estribo: float,
        delta: int = 50,
        plot: bool = False,
        color: str = "k",
        marker: Optional[str] = None,
        **plot_kwargs: Any,
    ) -> None:
        # Parámetros
        self.name = name
        self.fco = float(fco)
        self.eco = float(eco)
        self.b = float(b)
        self.h = float(h)
        self.rec = float(rec)
        self.num_var_b = int(num_var_b)
        self.num_var_h = int(num_var_h)
        self.phi_longitudinal = float(phi_longitudinal)
        self.num_est_perpendicular_b = int(num_est_perpendicular_b)
        self.num_est_perpendicular_h = int(num_est_perpendicular_h)
        self.phi_estribo = float(phi_estribo)
        self.s = float(s)
        self.fye = float(fye)
        self.esu_estribo = float(esu_estribo)
        self.delta = int(delta)
        self.plot_flag = plot
        self.color = color
        self.marker = marker
        self.plot_kwargs = plot_kwargs

        # Validación
        self._validate_inputs()

        # Tabla multiaxial (fcc_ratio en función de fl1/fl2)
        self._set_table()

        # Geometría efectiva y módulos
        self._compute_geometry()
        self._compute_confinement()
        self._compute_strengths()
        self._compute_moduli()

        # Curva constitutiva
        self.strain, self.stress, self.material_law = self._build_constitutive(self.delta)

        if self.plot_flag:
            self.plot(**self.plot_kwargs)

    # ------------------------------- Validaciones ------------------------------

    def _validate_inputs(self) -> None:
        if self.fco <= 0.0:
            raise ValueError("fco must be positive.")
        if self.eco <= 0.0:
            raise ValueError("eco must be positive.")
        if min(self.b, self.h, self.rec, self.phi_longitudinal, self.phi_estribo, self.s) <= 0.0:
            raise ValueError("Geometry and bar sizes must be positive.")
        if self.num_var_b < 2 or self.num_var_h < 2:
            raise ValueError("There must be at least 2 longitudinal bars per side (num_var_b, num_var_h >= 2).")
        if self.s <= self.phi_estribo:
            raise ValueError("Stirrup spacing must be greater than stirrup diameter.")
        if self.delta < 8:
            # se sugiere suficiente resolución para ascenso y post-pico
            raise ValueError("delta must be >= 8.")

        min_core_b = 2.0 * self.rec + 2.0 * self.phi_estribo
        min_core_h = 2.0 * self.rec + 2.0 * self.phi_estribo
        if self.b <= min_core_b or self.h <= min_core_h:
            raise ValueError("Section too small once cover and stirrups are accounted for.")

    # ------------------------------- Geometría ---------------------------------

    def _compute_geometry(self) -> None:
        # Dimensiones netas confinadas (centro a centro del estribo interior)
        self.bc = self.b - 2.0 * self.rec - self.phi_estribo
        self.hc = self.h - 2.0 * self.rec - self.phi_estribo
        if self.bc <= 0.0 or self.hc <= 0.0:
            raise ValueError("Computed confined dimensions (bc/hc) must be positive.")

        self.Ac = self.bc * self.hc

        # Nº total de barras longitudinales alrededor del perímetro
        self.num_var_long = self.num_var_b * 2 + max(self.num_var_h - 2, 0) * 2
        self.As = self.num_var_long * pi * (self.phi_longitudinal**2) / 4.0
        self.rho_confinado = self.As / self.Ac

        # Módulo elástico del hormigón (regla raíz cuadrada)
        self.Ec = (5000.0 * (self.fco) ** 0.5)

    # ----------------------------- Confinamiento --------------------------------

    def _compute_confinement(self) -> None:
        # Nº de vanos libres entre barras por cada dirección
        num_w_b = self.num_var_b - 1
        num_w_h = self.num_var_h - 1

        w_libre_b = (
            self.b - 2.0 * self.rec - 2.0 * self.phi_estribo - self.phi_longitudinal * self.num_var_b
        ) / num_w_b
        w_libre_h = (
            self.h - 2.0 * self.rec - 2.0 * self.phi_estribo - self.phi_longitudinal * self.num_var_h
        ) / num_w_h

        # Área “perdida” por vanos (triangulitos) dentro del núcleo
        Ai = 2.0 * num_w_b * (w_libre_b**2 / 6.0) + 2.0 * num_w_h * (w_libre_h**2 / 6.0)
        Acc = self.Ac * (1.0 - self.rho_confinado)

        # Efectividad geométrica (Mander) con reducción por espaciamiento libre
        s_libre = self.s - self.phi_estribo
        Ae = (self.Ac - Ai) * (1.0 - s_libre / (2.0 * self.bc)) * (1.0 - s_libre / (2.0 * self.hc))
        self.ke = Ae / Acc

        # Razones de estribo por dirección (área de ramas por paso y largo confinado)
        As_estribo_perp_b = self.num_est_perpendicular_b * pi * (self.phi_estribo**2) / 4.0
        As_estribo_perp_h = self.num_est_perpendicular_h * pi * (self.phi_estribo**2) / 4.0

        self.rho_estribo_perp_b = As_estribo_perp_b / (self.s * self.bc)
        self.rho_estribo_perp_h = As_estribo_perp_h / (self.s * self.hc)

        # Presiones laterales efectivas
        fl_b = self.rho_estribo_perp_b * self.fye
        fl_h = self.rho_estribo_perp_h * self.fye
        self.fl_perpendicular_b_efectivo = self.ke * fl_b
        self.fl_perpendicular_h_efectivo = self.ke * fl_h

    # ----------------------------- Resistencia confinada ------------------------

    def _interp_fcc_ratio(self, fl1_ratio: float, fl2_ratio: float) -> float:
        """
        Bilinear interpolation on the fcc/fco table.
        Rows = fl1/fco (smallest), Cols = fl2/fco (largest).
        """
        header = self.header
        table = self.Table

        # Clamp query to table domain
        fl1 = float(np.clip(fl1_ratio, header.min(), header.max()))
        fl2 = float(np.clip(fl2_ratio, header.min(), header.max()))

        # Find the two columns that bracket fl2
        j = int(np.searchsorted(header, fl2))
        j0 = max(j - 1, 0)
        j1 = min(j, len(header) - 1)

        x0 = header[j0]
        x1 = header[j1]
        c0 = table[:, j0]  # column at x0  (shape: 16,)
        c1 = table[:, j1]  # column at x1  (shape: 16,)

        # Linear weight in the fl2 direction (avoid div-by-zero when x0==x1)
        denom = (x1 - x0) if x1 != x0 else 1.0
        t = (fl2 - x0) / denom

        # Interpolate each row between the two columns → 1-D array (len = 16)
        interp_rows = c0 + t * (c1 - c0)

        # Now interpolate along fl1 using that 1-D array
        fcc_ratio = float(np.interp(fl1, header, interp_rows))
        return fcc_ratio


    def _compute_strengths(self) -> None:
        # Razones de presión efectiva adimensionales
        fl_b_ratio = self.fl_perpendicular_b_efectivo / self.fco
        fl_h_ratio = self.fl_perpendicular_h_efectivo / self.fco
        fl1_ratio = float(np.minimum(fl_b_ratio, fl_h_ratio))
        fl2_ratio = float(np.maximum(fl_b_ratio, fl_h_ratio))

        # Interpolación de fcc/fco
        self.fcc_ratio = self._interp_fcc_ratio(fl1_ratio, fl2_ratio)
        self.fcc = self.fco * self.fcc_ratio

        # Deformación de pico confinada (regla empírica típica)
        self.ecc_ratio = 1.0 + 5.0 * (self.fcc / self.fco - 1.0)
        self.ecc = self.ecc_ratio * self.eco

        # Deformación última (ecu) según expresión del código original
        # (escala con ρ_s, fye y ductilidad del estribo)
        denom = max(self.fcc, np.finfo(float).eps)
        self.ecu = 1.50 * (0.004 + 1.40 * ((self.rho_estribo_perp_b + self.rho_estribo_perp_h) * self.fye * self.esu_estribo) / denom)
        self.ecu_ratio = self.ecu / self.eco

    # ----------------------------- Módulos y r de Mander -----------------------

    def _compute_moduli(self) -> None:
        self.Esec = self.fcc / max(self.ecc, np.finfo(float).eps)
        denom = max(self.Ec - self.Esec, np.finfo(float).eps)
        self.r = self.Ec / denom

    # ----------------------------- Tabla multiaxial ----------------------------

    def _set_table(self) -> None:
        # Tabla de fcc/fco en función de fl1/fco (filas) y fl2/fco (columnas)
        self.Table = np.array([
            [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
            [1.062750334, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737, 1.14552737],
            [1.102803738, 1.196261682, 1.26835781, 1.26835781, 1.26835781, 1.26835781, 1.26835781, 1.26835781, 1.263017356, 1.263017356, 1.263017356, 1.263017356, 1.263017356, 1.263017356, 1.263017356, 1.263017356],
            [1.134846462, 1.23364486, 1.316421896, 1.364485981, 1.364485981, 1.364485981, 1.364485981, 1.364485981, 1.367156208, 1.367156208, 1.367156208, 1.367156208, 1.367156208, 1.367156208, 1.367156208, 1.367156208],
            [1.164218959, 1.263017356, 1.351134846, 1.417890521, 1.465954606, 1.465954606, 1.465954606, 1.465954606, 1.473965287, 1.473965287, 1.473965287, 1.473965287, 1.473965287, 1.473965287, 1.473965287, 1.473965287],
            [1.185580774, 1.284379172, 1.375166889, 1.452603471, 1.516688919, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231, 1.567423231],
            [1.204272363, 1.308411215, 1.407209613, 1.479305741, 1.551401869, 1.612817089, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175, 1.660881175],
            [1.217623498, 1.329773031, 1.428571429, 1.508678238, 1.580774366, 1.64753004, 1.703604806, 1.748998665, 1.746328438, 1.746328438, 1.746328438, 1.746328438, 1.746328438, 1.746328438, 1.746328438, 1.746328438],
            [1.238985314, 1.348464619, 1.449933244, 1.530040053, 1.610146862, 1.676902537, 1.738317757, 1.791722296, 1.834445928, 1.834445928, 1.834445928, 1.834445928, 1.834445928, 1.834445928, 1.834445928, 1.834445928],
            [1.252336449, 1.367156208, 1.468624833, 1.551401869, 1.634178905, 1.703604806, 1.765020027, 1.818424566, 1.869158879, 1.909212283, 1.909212283, 1.909212283, 1.909212283, 1.909212283, 1.909212283, 1.909212283],
            [1.265687583, 1.377837116, 1.481975968, 1.570093458, 1.655540721, 1.727636849, 1.794392523, 1.845126836, 1.898531375, 1.935914553, 1.970627503, 1.970627503, 1.970627503, 1.970627503, 1.970627503, 1.970627503],
            [1.273698264, 1.393858478, 1.503337784, 1.58611482, 1.67423231, 1.748998665, 1.818424566, 1.866488652, 1.925233645, 1.962616822, 2.005340454, 2.040053405, 2.040053405, 2.040053405, 2.040053405, 2.040053405],
            [1.281708945, 1.407209613, 1.514018692, 1.604806409, 1.690253672, 1.767690254, 1.837116155, 1.88518024, 1.946595461, 1.989319092, 2.032042724, 2.072096128, 2.109479306, 2.109479306, 2.109479306, 2.109479306],
            [1.292389853, 1.417890521, 1.530040053, 1.62082777, 1.706275033, 1.783711615, 1.855807744, 1.909212283, 1.970627503, 2.013351135, 2.06141522, 2.106809079, 2.136181575, 2.173564753, 2.173564753, 2.173564753],
            [1.300400534, 1.425901202, 1.540720961, 1.631508678, 1.719626168, 1.799732977, 1.871829105, 1.927903872, 1.989319092, 2.034712951, 2.082777036, 2.130841121, 2.160213618, 2.200267023, 2.234979973, 2.234979973],
            [1.308411215, 1.433911883, 1.554072096, 1.644859813, 1.732977303, 1.815754339, 1.887850467, 1.941255007, 2.008010681, 2.053404539, 2.106809079, 2.152202937, 2.189586115, 2.224299065, 2.259012016, 2.3]
        ])
        self.header = np.array([0.00, 0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30])

    # --------------------------- Curva constitutiva ----------------------------

    def _build_constitutive(
        self, delta: int
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Construye la ley σ(ε) confinada:
          - muestreo denso hacia ecu (geo + linspace combinados),
          - Mander en ascenso con (fcc, ecc, r),
          - sin rama explícita de ablandamiento (ecu es último punto).
        """
        # malla no uniforme (más densa hacia ecu)
        n_lin = max(delta // 4, 3)
        n_geo = max(delta - n_lin, 5)

        es_lin = np.linspace(0.0, self.ecc, n_lin, dtype=float)
        # geomspace requiere límites > 0; aseguramos mínimo eps
        start_geo = max(self.ecc, np.finfo(float).eps)
        stop_geo = max(self.ecu, start_geo * 1.0001)
        es_geo = np.geomspace(start_geo, stop_geo, n_geo, dtype=float)
        es = np.unique(np.concatenate([es_lin, es_geo]))

        # Mander vectorizado
        x = es / self.ecc
        with np.errstate(divide="ignore", invalid="ignore"):
            fs = (self.fcc * x * self.r) / (self.r - 1.0 + np.power(x, self.r))

        fs = np.clip(fs, 0.0, None)
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
        """Plot σ–ε curve."""
        if ax is None:
            _, ax = plt.subplots(figsize=(10, 5))
            ax.set_title("Confined Concrete – σ–ε")

        label = label or f"Confined: {self.name}"
        defaults = dict(color=self.color, linewidth=1.5, marker=self.marker)
        defaults.update(kwargs)

        ax.plot(self.strain, self.stress, label=label, **defaults)
        ax.set_xlabel("Strain")
        ax.set_ylabel("Stress")
        ax.legend()
        return ax

    # --------------------------- Gráficos multiaxiales -------------------------

    def plot_multiaxial_strength_2d(
        self, *, line_color: Optional[str] = None, **kwargs: Any
    ) -> Tuple[Axes, Axes]:
        """
        Visualiza la tabla fcc/fco vs. (fl1/fco, fl2/fco) en 2D (filas y columnas).
        """
        table = self.Table
        header = self.header
        c = line_color or self.color

        fig, axs = plt.subplots(1, 2, figsize=(16, 6))

        # Filas: variación con fl1 para diferentes fl2 (aprox., usando índice i)
        for i, row in enumerate(table):
            axs[0].plot(header, row, color=c, linewidth=1.2, **kwargs)
            axs[0].text(header[-1], row[-1], f"fl2/fco={header[i]:.2f}", fontsize=8, color="k", ha="left")

        axs[0].set_xlabel("Smallest confining stress ratio  fl1/fco")
        axs[0].set_ylabel("Confined strength ratio  fcc/fco")
        axs[0].grid(True)

        # Columnas: variación con fl2 para diferentes fl1 (por columna)
        for j in range(table.shape[1]):
            col = table[:, j]
            axs[1].plot(header, col, color=c, linewidth=1.2, **kwargs)
            axs[1].text(header[-1], col[-1], f"fl1/fco={header[j]:.2f}", fontsize=8, color="k", ha="left")

        axs[1].set_xlabel("Largest confining stress ratio  fl2/fco")
        axs[1].set_ylabel("Confined strength ratio  fcc/fco")
        axs[1].grid(True)

        return axs[0], axs[1]

    def plot_multiaxial_strength_3d(self) -> None:
        """
        Mapa de contornos + superficie 3D de la tabla fcc/fco (usa 'viridis').
        """
        header = self.header
        Z = self.Table
        smallest, largest = np.meshgrid(header, header)

        fig = plt.figure(figsize=(16, 6))

        ax1 = fig.add_subplot(1, 2, 1)
        contour = ax1.contourf(smallest, largest, Z, cmap="viridis", levels=40)
        contour_lines = ax1.contour(smallest, largest, Z, colors="black", linewidths=0.4, levels=20)
        ax1.clabel(contour_lines, fmt="%1.2f", colors="black", fontsize=8)
        fig.colorbar(contour, ax=ax1, label="fcc/fco")
        ax1.set_xlabel("fl1/fco (smallest)")
        ax1.set_ylabel("fl2/fco (largest)")
        ax1.set_title("Confined Strength Ratio (contours)")
        ax1.grid(True)

        ax2 = fig.add_subplot(1, 2, 2, projection="3d")
        surf = ax2.plot_surface(smallest, largest, Z, cmap="viridis", linewidth=0, antialiased=True)
        fig.colorbar(surf, ax=ax2, shrink=0.6, aspect=12, label="fcc/fco")
        ax2.set_xlabel("fl1/fco (smallest)")
        ax2.set_ylabel("fl2/fco (largest)")
        ax2.set_zlabel("fcc/fco")
        ax2.set_title("Confined Strength Ratio (surface)")

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


    # ------------------------------ Representación -----------------------------

    def __repr__(self) -> str:
        return (f"UniaxialConfinedConcrete(name='{self.name}', fco={self.fco}, "
                f"eco={self.eco}, b={self.b}, h={self.h}, s={self.s})")

    def __str__(self) -> str:
        return self.name

