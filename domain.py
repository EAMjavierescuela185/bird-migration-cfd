import numpy as np
import matplotlib.pyplot as plt


class MigrationDomain:
    """
    Defines a 400 km x 400 km Mid-Atlantic flyway domain (Eastern PA, NJ, DE, MD)
    on a uniform 2D Cartesian grid with:
    - Atlantic Ocean coastline, Delaware Bay, and Chesapeake Bay (forming the Delmarva Peninsula)
    - Appalachian / Kittatinny Ridge (Hawk Mountain corridor)
    - Physically bounded effective bird migration velocity field (u_eff, v_eff)
    - Dusk stopover takeoff density distribution rho(x, y, 0)
    """

    def __init__(self, nx=100, ny=100, lx=400.0, ly=400.0):
        self.nx = nx
        self.ny = ny
        self.lx = lx  # East-West extent [km]
        self.ly = ly  # North-South extent [km]

        # Grid spacing (Week 6/8 uniform Cartesian mesh)
        self.dx = lx / (nx - 1)
        self.dy = ly / (ny - 1)

        # 1D coordinate arrays and 2D meshgrid
        self.x = np.linspace(0.0, lx, nx)
        self.y = np.linspace(0.0, ly, ny)
        self.X, self.Y = np.meshgrid(self.x, self.y, indexing="xy")

        # Build static geography
        (
            self.is_water,
            self.coast_x,
            self.chesapeake_mask,
            self.ridge_field,
        ) = self._build_geography()

    def _build_geography(self):
        """
        Constructs realistic Mid-Atlantic water bodies and topography:
        1. Atlantic Ocean (East/Southeast) + Delaware Bay notch
        2. Chesapeake Bay (creating the Delmarva Peninsula funnel in the south)
        3. Appalachian / Kittatinny Ridge running NE-to-SW across PA
        """
        # 1. Atlantic Coastline x_coast(y): angles SW from Northern NJ to VA
        base_coast = 235.0 + 0.42 * (self.Y - 80.0)
        # Delaware Bay indentation around y = 155 km (Cape May is just north of this)
        delaware_bay = -30.0 * np.exp(-((self.Y - 155.0) / 16.0) ** 2)
        coast_x = base_coast + delaware_bay

        dist_to_ocean = coast_x - self.X  # > 0 inland, < 0 in Atlantic Ocean
        is_ocean = dist_to_ocean < 0.0

        # 2. Chesapeake Bay (narrow bay running N-S between x ~ 170-192 km for y < 175 km)
        chesapeake_center_x = 165.0 + 0.16 * self.Y
        chesapeake_half_width = 13.0 * np.exp(-np.maximum(self.Y - 165.0, 0.0) / 12.0)
        is_chesapeake = (np.abs(self.X - chesapeake_center_x) < chesapeake_half_width) & (self.Y < 185.0)

        is_water = is_ocean | is_chesapeake

        # 3. Appalachian / Kittatinny Ridge running NE-to-SW across PA
        ridge_center_x = 45.0 + 0.65 * self.Y
        dist_to_ridge = np.abs(self.X - ridge_center_x)
        ridge_field = np.exp(-(dist_to_ridge / 16.0) ** 2)
        ridge_field[is_water] = 0.0

        return is_water, coast_x, is_chesapeake, ridge_field

    def compute_velocity_field(
        self,
        bird_airspeed=28.0,      # [km/hr] (~32 km/hr net ground speed with tailwind)
        heading_deg=228.0,       # [deg] 228 = Southwest autumn migration heading
        wind_u=-6.0,             # [km/hr] background Westward wind component
        wind_v=-10.0,            # [km/hr] background Southward wind component
        drift_alpha=0.65,        # Partial wind drift factor (0 = full compensation, 1 = pure drift)
        coastal_repulsion=24.0,  # [km/hr] bounded inland steering near coastlines
        ridge_channeling=8.0     # [km/hr] favorable updraft/tailing boost along mountain ridge
    ):
        """
        Computes physically bounded 2D effective bird velocity (u_eff, v_eff) [km/hr].
        """
        # 1. Active Bird Heading Velocity
        theta = np.deg2rad(heading_deg)
        u_bird = bird_airspeed * np.cos(theta)
        v_bird = bird_airspeed * np.sin(theta)

        # 2. Spatially varying synoptic wind field (e.g., passing NW cold front)
        u_wind = wind_u * (1.0 + 0.15 * np.cos(np.pi * self.Y / self.ly))
        v_wind = wind_v * (1.0 + 0.15 * np.sin(np.pi * self.X / self.lx))

        u_eff = u_bird + drift_alpha * u_wind
        v_eff = v_bird + drift_alpha * v_wind

        # 3. Bounded Atlantic Coastal Avoidance (peaks right at shoreline, decays both inland & offshore)
        dist_to_ocean = self.coast_x - self.X
        shore_zone = np.exp(-(dist_to_ocean / 24.0) ** 2)
        u_eff = u_eff - 0.85 * coastal_repulsion * shore_zone
        v_eff = v_eff - 0.25 * coastal_repulsion * shore_zone

        # 4. Chesapeake Bay Shoreline Funneling (keeps Delmarva birds on the peninsula)
        chesapeake_center_x = 165.0 + 0.16 * self.Y
        dx_bay = self.X - chesapeake_center_x
        bay_zone = np.exp(-(dx_bay / 18.0) ** 2) * (self.Y < 185.0)
        u_eff = u_eff + 0.65 * coastal_repulsion * np.tanh(dx_bay / 8.0) * bay_zone
        v_eff = v_eff - 0.35 * coastal_repulsion * bay_zone

        # 5. Appalachian Ridge Channeling (aligns flow along SW ridge axis ~237 deg)
        u_eff = u_eff - 0.54 * ridge_channeling * self.ridge_field
        v_eff = v_eff - 0.84 * ridge_channeling * self.ridge_field

        return u_eff, v_eff, u_wind, v_wind

    def initial_density(self):
        """
        Dusk takeoff bird density rho(x, y, t=0) [birds / km^2] over forested stopover habitats.
        Shifted slightly NE so flocks traverse the full Mid-Atlantic corridor over 6-8 hours.
        """
        # Stopover A: Poconos / NE Pennsylvania forests
        poconos = 130.0 * np.exp(-((self.X - 235.0) ** 2 / (2 * 30.0**2) + (self.Y - 350.0) ** 2 / (2 * 26.0**2)))
        # Stopover B: Central PA Appalachian forests
        central_pa = 115.0 * np.exp(-((self.X - 160.0) ** 2 / (2 * 36.0**2) + (self.Y - 300.0) ** 2 / (2 * 32.0**2)))
        # Stopover C: Northern/Central NJ forests
        pine_barrens = 100.0 * np.exp(-((self.X - 300.0) ** 2 / (2 * 18.0**2) + (self.Y - 285.0) ** 2 / (2 * 22.0**2)))

        rho0 = poconos + central_pa + pine_barrens
        rho0[self.is_water] = 0.0
        return rho0

    def plot_landmarks(self, ax):
        """Adds key geographic reference points to make plots read like BirdCast maps."""
        landmarks = {
            "Philadelphia": (235, 215),
            "Hawk Mtn": (180, 260),
            "Cape May": (262, 165),
            "Delmarva\nPeninsula": (212, 95),
        }
        for name, (lx, ly) in landmarks.items():
            ax.plot(lx, ly, "ko", markersize=4)
            ax.text(lx + 4, ly + 3, name, fontsize=8, fontweight="bold", color="#1a1a1a")


# ==============================================================================
# STEP 1 VERIFICATION: Visualize Grid, Geography, Initial Density & Velocity Field
# ==============================================================================
if __name__ == "__main__":
    domain = MigrationDomain(nx=100, ny=100, lx=400.0, ly=400.0)
    u_eff, v_eff, u_wind, v_wind = domain.compute_velocity_field()
    rho0 = domain.initial_density()

    speed_eff = np.hypot(u_eff, v_eff)
    print(f"Grid initialized: {domain.nx} x {domain.ny} nodes (dx = {domain.dx:.2f} km, dy = {domain.dy:.2f} km)")
    print(f"Effective migration speed range: {np.min(speed_eff):.1f} - {np.max(speed_eff):.1f} km/hr")
    print(f"Initial peak bird density: {np.max(rho0):.2f} birds/km^2")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))

    # Panel 1: Geography + Dusk Takeoff Density
    ax1 = axes[0]
    ax1.contourf(domain.X, domain.Y, domain.is_water.astype(float), levels=[0.5, 1.5], colors=["#cde6f5"])
    ax1.contour(domain.X, domain.Y, domain.ridge_field, levels=[0.35, 0.75], colors=["darkgreen"], linestyles="--")
    cf1 = ax1.contourf(domain.X, domain.Y, rho0, levels=np.linspace(5, 130, 18), cmap="YlOrRd", extend="max")
    ax1.plot(domain.coast_x, domain.y, "b-", linewidth=1.8)
    domain.plot_landmarks(ax1)
    ax1.set_title("Step 1A: Mid-Atlantic Geography & Dusk Takeoff (t = 0 hr)", fontweight="bold")
    ax1.set_xlabel("East-West Distance [km]")
    ax1.set_ylabel("North-South Distance [km]")
    ax1.grid(True, linestyle=":", alpha=0.5)
    fig.colorbar(cf1, ax=ax1, shrink=0.82, pad=0.03, label="Initial Density [birds/km²]")

    # Panel 2: Effective Migration Velocity Field
    ax2 = axes[1]
    ax2.contourf(domain.X, domain.Y, domain.is_water.astype(float), levels=[0.5, 1.5], colors=["#cde6f5"])
    cf2 = ax2.contourf(domain.X, domain.Y, speed_eff, levels=20, cmap="viridis", alpha=0.85)
    ax2.plot(domain.coast_x, domain.y, "b-", linewidth=1.8)
    domain.plot_landmarks(ax2)
    skip = 6
    ax2.quiver(
        domain.X[::skip, ::skip], domain.Y[::skip, ::skip],
        u_eff[::skip, ::skip], v_eff[::skip, ::skip],
        color="white", scale=850
    )
    ax2.set_title("Step 1B: Effective Migration Velocity Field (u_eff, v_eff)", fontweight="bold")
    ax2.set_xlabel("East-West Distance [km]")
    ax2.grid(True, linestyle=":", alpha=0.5)
    fig.colorbar(cf2, ax=ax2, shrink=0.82, pad=0.03, label="Effective Speed [km/hr]")

    plt.subplots_adjust(wspace=0.22)
    plt.savefig("step1_domain_verification.png", dpi=200, bbox_inches="tight")
    plt.show()