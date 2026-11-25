import numpy as np
import matplotlib.pyplot as plt
from domain import MigrationDomain


class MigrationSolver:
    """
    2D Finite-Difference CFD Solver for Nocturnal Bird Migration Density rho(x, y, t).
    Implements:
    - 1st-order Upwind Differencing for convective fluxes (Week 6)
    - 2nd-order Central Differencing for turbulent/flock diffusion (Week 6)
    - 4-Stage Explicit Runge-Kutta (RK4) time integration (Week 6)
    - Automatic 2D CFL time-step sizing + optional CFL stability test
    - Northern flyway inflow BC (NY/New England stream) + dawn landing sink over land
    """

    def __init__(self, domain: MigrationDomain, diffusion_coeff=8.0, cfl=0.8):
        self.domain = domain
        self.D = diffusion_coeff  # Flock dispersion coefficient [km^2 / hr]
        self.cfl = cfl            # Courant-Friedrichs-Lewy safety number

    def compute_stable_dt(self, u, v):
        """
        Computes maximum stable time step dt from the 2D Advection-Diffusion CFL criterion:
            CFL = dt * ( max(|u|/dx + |v|/dy) + 2*D*(1/dx^2 + 1/dy^2) )
        """
        dx, dy = self.domain.dx, self.domain.dy
        max_adv_rate = np.max(np.abs(u) / dx + np.abs(v) / dy)
        diff_rate = 2.0 * self.D * (1.0 / dx**2 + 1.0 / dy**2)
        return self.cfl / (max_adv_rate + diff_rate)

    def northern_inflow(self, t, inflow_cutoff=5.5):
        """
        Dirichlet inflow profile at the northern boundary (y = Ly):
        Models nocturnal migrants streaming southward from NY & New England
        along two main corridors (Appalachian ridge x ~ 245 km & Hudson/NJ coast x ~ 335 km).
        """
        if t > inflow_cutoff:
            return np.zeros(self.domain.nx)

        # Smooth ramp-down as nocturnal migration wave tapers off before dawn
        time_envelope = np.exp(-max(t - 3.5, 0.0) ** 2 / 1.5)
        x = self.domain.x
        appalachian_stream = 85.0 * np.exp(-((x - 245.0) / 35.0) ** 2)
        coastal_stream = 75.0 * np.exp(-((x - 335.0) / 22.0) ** 2)

        inflow = time_envelope * (appalachian_stream + coastal_stream)
        inflow[self.domain.is_water[-1, :]] = 0.0
        return inflow

    def dawn_landing_sink(self, rho, t, dawn_start=6.5, landing_rate=0.75):
        """
        Ecological sink term S(x, y, t) [birds / (km^2 * hr)]:
        - For t < dawn_start: S = 0 (birds remain aloft migrating overnight)
        - For t >= dawn_start: birds descend smoothly onto land habitats (~is_water),
          while S = 0 over water (songbirds cannot land in the ocean/bay).
        """
        if t < dawn_start:
            return np.zeros_like(rho)

        ramp = 1.0 - np.exp(-(t - dawn_start) / 0.75)
        land_mask = (~self.domain.is_water).astype(float)
        return -landing_rate * ramp * land_mask * rho

    def residual(self, rho, t, u, v, enable_dawn_landing=True):
        """
        Computes spatial residual R(rho, t) = -div(V_eff * rho) + D * laplacian(rho) + S(x, y, t).
        Uses ghost-cell padding: Dirichlet inflow on the North edge (where v < 0),
        and zero-gradient (Neumann) outflow on South, West, and East edges.
        """
        dx, dy = self.domain.dx, self.domain.dy

        rho_p = np.pad(rho, 1, mode="edge")
        u_p = np.pad(u, 1, mode="edge")
        v_p = np.pad(v, 1, mode="edge")

        # Apply Northern Dirichlet Inflow BC (top row of padded array corresponds to y = Ly)
        rho_p[-1, 1:-1] = self.northern_inflow(t)

        # 1. First-Order Upwind Advection in X (Conservative flux form: d(u*rho)/dx)
        flux_x_backward = (u * rho - u_p[1:-1, :-2] * rho_p[1:-1, :-2]) / dx
        flux_x_forward  = (u_p[1:-1, 2:] * rho_p[1:-1, 2:] - u * rho) / dx
        conv_x = np.where(u >= 0.0, flux_x_backward, flux_x_forward)

        # 2. First-Order Upwind Advection in Y (Conservative flux form: d(v*rho)/dy)
        flux_y_backward = (v * rho - v_p[:-2, 1:-1] * rho_p[:-2, 1:-1]) / dy
        flux_y_forward  = (v_p[2:, 1:-1] * rho_p[2:, 1:-1] - v * rho) / dy
        conv_y = np.where(v >= 0.0, flux_y_backward, flux_y_forward)

        # 3. Second-Order Central Differencing for Flock Diffusion
        d2rho_dx2 = (rho_p[1:-1, 2:] - 2.0 * rho + rho_p[1:-1, :-2]) / (dx**2)
        d2rho_dy2 = (rho_p[2:, 1:-1] - 2.0 * rho + rho_p[:-2, 1:-1]) / (dy**2)
        diffusion = self.D * (d2rho_dx2 + d2rho_dy2)

        # 4. Dawn Landing Source/Sink Term
        sink = self.dawn_landing_sink(rho, t) if enable_dawn_landing else 0.0

        return -(conv_x + conv_y) + diffusion + sink

    def rk4_step(self, rho, t, dt, u, v, enable_dawn_landing=True):
        """
        Advances rho by one time step dt using the 4-stage Runge-Kutta scheme (Week 6, Slide 8).
        """
        q0 = rho
        q1 = q0 + (dt / 4.0) * self.residual(q0, t, u, v, enable_dawn_landing)
        q2 = q0 + (dt / 3.0) * self.residual(q1, t + dt / 4.0, u, v, enable_dawn_landing)
        q3 = q0 + (dt / 2.0) * self.residual(q2, t + dt / 3.0, u, v, enable_dawn_landing)
        q4 = q0 + dt * self.residual(q3, t + dt / 2.0, u, v, enable_dawn_landing)
        return np.maximum(q4, 0.0)

    def solve(
        self,
        rho0,
        u,
        v,
        t_end=9.0,
        save_times=(0.0, 3.0, 6.0, 9.0),
        enable_dawn_landing=True,
    ):
        """
        Integrates the 2D migration PDE from t = 0 to t = t_end.
        """
        rho = rho0.copy()
        dt_base = self.compute_stable_dt(u, v)
        cell_area = self.domain.dx * self.domain.dy

        t = 0.0
        steps = 0
        snapshots = {}
        if 0.0 in save_times:
            snapshots[0.0] = rho.copy()

        time_hist = [0.0]
        airborne_hist = [np.sum(rho) * cell_area]
        over_water_hist = [np.sum(rho[self.domain.is_water]) * cell_area]
        peak_hist = [np.max(rho)]

        remaining_targets = sorted([st for st in save_times if st > 0.0])

        while t < t_end - 1e-9:
            dt = dt_base
            if remaining_targets and (t + dt >= remaining_targets[0] - 1e-9):
                dt = remaining_targets[0] - t
                hit_target = remaining_targets.pop(0)
            else:
                hit_target = None

            rho = self.rk4_step(rho, t, dt, u, v, enable_dawn_landing)
            t += dt
            steps += 1

            if hit_target is not None:
                snapshots[hit_target] = rho.copy()

            time_hist.append(t)
            airborne_hist.append(np.sum(rho) * cell_area)
            over_water_hist.append(np.sum(rho[self.domain.is_water]) * cell_area)
            peak_hist.append(np.max(rho))

        diagnostics = {
            "time": np.array(time_hist),
            "total_airborne": np.array(airborne_hist),
            "over_water": np.array(over_water_hist),
            "peak_density": np.array(peak_hist),
            "dt": dt_base,
            "steps": steps,
        }
        return snapshots, diagnostics


# ==============================================================================
# STEP 2 VERIFICATION: Run Baseline Simulation & Check Diagnostics
# ==============================================================================
if __name__ == "__main__":
    domain = MigrationDomain(nx=100, ny=100, lx=400.0, ly=400.0)
    # Tuned realistic nocturnal flight speeds (~32 km/hr net ground speed)
    u_eff, v_eff, _, _ = domain.compute_velocity_field(
        bird_airspeed=28.0,
        heading_deg=228.0,
        wind_u=-6.0,
        wind_v=-10.0,
        drift_alpha=0.65,
        coastal_repulsion=24.0,
        ridge_channeling=8.0,
    )
    rho0 = domain.initial_density()

    solver = MigrationSolver(domain, diffusion_coeff=8.0, cfl=0.8)
    save_times = (0.0, 3.0, 6.0, 9.0)
    snapshots, diag = solver.solve(rho0, u_eff, v_eff, t_end=9.0, save_times=save_times)

    print(f"Solver completed: {diag['steps']} RK4 steps, dt = {diag['dt']:.4f} hr")
    print(f"Initial total birds aloft: {diag['total_airborne'][0]:,.0f}")
    print(f"Peak aloft (t=4 hr):       {np.max(diag['total_airborne']):,.0f}")
    print(f"After dawn landing (t=9h): {diag['total_airborne'][-1]:,.0f}")

    fig, axes = plt.subplots(2, 2, figsize=(14, 11))
    axes = axes.ravel()
    vmax = 115.0

    titles = {
        0.0: "t = 0.0 hr (Dusk Takeoff)",
        3.0: "t = 3.0 hr (Nocturnal Flyway Surge)",
        6.0: "t = 6.0 hr (Cape May & Delmarva Funneling)",
        9.0: "t = 9.0 hr (Post-Dawn Landing Over Land)",
    }

    for ax, t_snap in zip(axes, save_times):
        rho_snap = snapshots[t_snap]
        ax.contourf(domain.X, domain.Y, domain.is_water.astype(float), levels=[0.5, 1.5], colors=["#cde6f5"])
        ax.contour(domain.X, domain.Y, domain.ridge_field, levels=[0.4], colors=["darkgreen"], linestyles="--", alpha=0.6)
        cf = ax.contourf(domain.X, domain.Y, rho_snap, levels=np.linspace(3, vmax, 20), cmap="YlOrRd", extend="max")
        ax.plot(domain.coast_x, domain.y, "b-", linewidth=1.6)
        domain.plot_landmarks(ax)

        skip = 8
        ax.quiver(
            domain.X[::skip, ::skip], domain.Y[::skip, ::skip],
            u_eff[::skip, ::skip], v_eff[::skip, ::skip],
            color="black", alpha=0.4, scale=650
        )
        ax.set_title(titles[t_snap], fontweight="bold")
        ax.set_xlabel("East-West Distance [km]")
        ax.set_ylabel("North-South Distance [km]")
        ax.grid(True, linestyle=":", alpha=0.4)

    plt.subplots_adjust(right=0.88, hspace=0.28, wspace=0.22)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label="Bird Migration Density [birds / km²]")
    fig.suptitle("Step 2: 2D Nocturnal Bird Migration Evolution (Upwind + RK4)", fontsize=14, fontweight="bold")
    plt.savefig("step2_migration_evolution.png", dpi=200, bbox_inches="tight")
    plt.show()