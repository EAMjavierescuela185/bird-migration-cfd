import numpy as np
import matplotlib.pyplot as plt
from domain import MigrationDomain
from solver import MigrationSolver


def run_weather_scenarios(nx=100, ny=100):
    """
    Compares two meteorological scenarios at peak migration (t = 6.0 hr):
      1. Baseline NW Cold Front (favorable tailwinds)
      2. Strong Offshore Westerly Crosswind (coastal bottlenecking at Cape May / Delmarva)
    """
    domain = MigrationDomain(nx=nx, ny=ny, lx=400.0, ly=400.0)
    rho0 = domain.initial_density()
    solver = MigrationSolver(domain, diffusion_coeff=8.0, cfl=0.8)

    # Scenario 1: Favorable NW Cold Front
    u1, v1, _, _ = domain.compute_velocity_field(
        bird_airspeed=28.0, heading_deg=228.0, wind_u=-6.0, wind_v=-10.0, drift_alpha=0.65
    )
    snaps1, diag1 = solver.solve(rho0, u1, v1, t_end=9.0, save_times=(0.0, 6.0, 9.0))

    # Scenario 2: Strong Westerly Crosswind pushing birds toward the Atlantic Coast
    u2, v2, _, _ = domain.compute_velocity_field(
        bird_airspeed=28.0, heading_deg=225.0, wind_u=+14.0, wind_v=-8.0, drift_alpha=0.75
    )
    snaps2, diag2 = solver.solve(rho0, u2, v2, t_end=9.0, save_times=(0.0, 6.0, 9.0))

    # --- Plot Figure 1: Side-by-Side Weather Comparison at t = 6.0 hr ---
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.2))
    vmax = max(np.max(snaps1[6.0]), np.max(snaps2[6.0])) * 0.95

    scenarios = [
        (axes[0], snaps1[6.0], u1, v1, "Scenario A: NW Cold Front (Favorable Tailwinds)"),
        (axes[1], snaps2[6.0], u2, v2, "Scenario B: Westerly Crosswind (Coastal Funneling)"),
    ]

    for ax, rho_snap, u_f, v_f, title in scenarios:
        ax.contourf(domain.X, domain.Y, domain.is_water.astype(float), levels=[0.5, 1.5], colors=["#cde6f5"])
        ax.contour(domain.X, domain.Y, domain.ridge_field, levels=[0.4], colors=["darkgreen"], linestyles="--", alpha=0.6)
        cf = ax.contourf(domain.X, domain.Y, rho_snap, levels=np.linspace(3, vmax, 20), cmap="YlOrRd", extend="max")
        ax.plot(domain.coast_x, domain.y, "b-", linewidth=1.8)
        domain.plot_landmarks(ax)

        skip = 7
        ax.quiver(
            domain.X[::skip, ::skip], domain.Y[::skip, ::skip],
            u_f[::skip, ::skip], v_f[::skip, ::skip],
            color="black", alpha=0.45, scale=650
        )
        ax.set_title(title, fontweight="bold")
        ax.set_xlabel("East-West Distance [km]")
        ax.grid(True, linestyle=":", alpha=0.4)

    axes[0].set_ylabel("North-South Distance [km]")
    plt.subplots_adjust(right=0.88, wspace=0.18)
    cbar_ax = fig.add_axes([0.90, 0.15, 0.02, 0.7])
    fig.colorbar(cf, cax=cbar_ax, label="Bird Migration Density [birds / km²]")
    fig.suptitle("Weather Impact on Nocturnal Migration Corridors (t = 6.0 hr)", fontsize=14, fontweight="bold")
    plt.savefig("fig1_weather_scenarios.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    # --- Plot Figure 2: Quantitative Time-Series Diagnostics ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))

    # Panel A: Total Airborne Birds Over Time
    axes[0].plot(diag1["time"], diag1["total_airborne"] / 1e5, "b-", lw=2.2, label="Scenario A: NW Tailwind")
    axes[0].plot(diag2["time"], diag2["total_airborne"] / 1e5, "r--", lw=2.2, label="Scenario B: Westerly Crosswind")
    axes[0].axvline(5.5, color="gray", ls=":", label="Northern Inflow Cutoff (t = 5.5 hr)")
    axes[0].axvline(6.5, color="green", ls="-.", label="Dawn Landing Begins (t = 6.5 hr)")
    axes[0].set_title("Total Airborne Migrants Over Region", fontweight="bold")
    axes[0].set_xlabel("Time Since Dusk Takeoff [hr]")
    axes[0].set_ylabel("Total Birds Aloft [× 10⁵ birds]")
    axes[0].legend(fontsize=8.5)
    axes[0].grid(True, alpha=0.4)

    # Panel B: Peak Corridor Density Over Time
    axes[1].plot(diag1["time"], diag1["peak_density"], "b-", lw=2.2, label="Scenario A: NW Tailwind")
    axes[1].plot(diag2["time"], diag2["peak_density"], "r--", lw=2.2, label="Scenario B: Westerly Crosswind")
    axes[1].axvline(6.5, color="green", ls="-.", label="Dawn Landing Begins (t = 6.5 hr)")
    axes[1].set_title("Peak Flyway Concentration vs. Time", fontweight="bold")
    axes[1].set_xlabel("Time Since Dusk Takeoff [hr]")
    axes[1].set_ylabel("Peak Density [birds / km²]")
    axes[1].legend(fontsize=8.5)
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("fig2_migration_diagnostics.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    return diag1, diag2


def run_cfd_verification():
    """
    Performs two core numerical verification checks from Week 6 and Week 8:
      1. Grid Convergence Study (50x50, 100x100, 150x150) + Richardson Extrapolation
      2. RK4 CFL Stability Limit Test (CFL = 0.8 vs 2.6 vs 3.2)
    """
    print("\n--- Running Week 8 Grid Convergence Study ---")
    grid_ns = [50, 100, 150]
    slice_profiles = {}
    peak_values = []
    total_mass_values = []

    for N in grid_ns:
        dom = MigrationDomain(nx=N, ny=N, lx=400.0, ly=400.0)
        u, v, _, _ = dom.compute_velocity_field(
            bird_airspeed=28.0, heading_deg=228.0, wind_u=-6.0, wind_v=-10.0, drift_alpha=0.65
        )
        rho0 = dom.initial_density()
        sol = MigrationSolver(dom, diffusion_coeff=8.0, cfl=0.8)
        snaps, diag = sol.solve(rho0, u, v, t_end=6.0, save_times=(6.0,), enable_dawn_landing=False)

        rho_6h = snaps[6.0]
        # Extract East-West density cross-section at y = 200 km (latitude of Philadelphia)
        j_mid = N // 2
        slice_profiles[N] = (dom.x, rho_6h[j_mid, :])
        peak_values.append(np.max(rho_6h))
        total_mass_values.append(diag["total_airborne"][-1])
        print(f"  Grid {N:3d}x{N:3d} (dx={dom.dx:5.2f} km): Peak = {peak_values[-1]:6.2f} birds/km², "
              f"Total Aloft = {total_mass_values[-1]:,.0f} birds")

    # Richardson Extrapolation estimate using Coarse (50), Medium (100), Fine (150)
    # Refinement ratios: r21 = dx_50 / dx_100 ~ 2.0, r32 = dx_100 / dx_150 ~ 1.5
    F1, F2, F3 = peak_values[0], peak_values[1], peak_values[2]
    if (F3 - F2) * (F2 - F1) > 0:
        p_est = np.log(abs((F2 - F1) / (F3 - F2))) / np.log(1.5)
        p_est = np.clip(p_est, 0.5, 4.0)
        f_exact = F3 + (F3 - F2) / (1.5**p_est - 1.0)
        print(f"  Richardson Extrapolated Peak Density: {f_exact:.2f} birds/km² (estimated order p ≈ {p_est:.2f})")
    else:
        f_exact = F3
        print(f"  Fine-grid Peak Density: {F3:.2f} birds/km²")

    # --- Week 6 CFL Stability Test ---
    print("\n--- Running Week 6 RK4 CFL Stability Verification ---")
    dom_test = MigrationDomain(nx=80, ny=80, lx=400.0, ly=400.0)
    u_t, v_t, _, _ = dom_test.compute_velocity_field(
        bird_airspeed=28.0, heading_deg=228.0, wind_u=-6.0, wind_v=-10.0, drift_alpha=0.65
    )
    rho0_t = dom_test.initial_density()

    cfl_cases = [0.5, 0.9, 1.5]
    cfl_histories = {}
    for cfl_val in cfl_cases:
        sol_cfl = MigrationSolver(dom_test, diffusion_coeff=8.0, cfl=cfl_val)
        _, diag_cfl = sol_cfl.solve(rho0_t, u_t, v_t, t_end=3.0, save_times=(), enable_dawn_landing=False)
        # Cap values for clean log plotting if blow-up occurs
        peaks = np.minimum(diag_cfl["peak_density"], 1e12)
        cfl_histories[cfl_val] = (diag_cfl["time"], peaks)
        status = "STABLE" if peaks[-1] < 500.0 else "UNSTABLE (BLOW-UP)"
        print(f"  CFL = {cfl_val:.1f}: max(rho) at t=3h = {peaks[-1]:.3e}  [{status}]")

    # --- Plot Figure 3: Grid Convergence & CFL Stability ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Panel A: Grid Convergence Cross-Section at y = 200 km
    styles = {50: ("r--", "Coarse (50×50, dx=8.16 km)"),
              100: ("b-.", "Medium (100×100, dx=4.04 km)"),
              150: ("k-", "Fine (150×150, dx=2.68 km)")}
    for N in grid_ns:
        x_coords, prof = slice_profiles[N]
        ls, lbl = styles[N]
        axes[0].plot(x_coords, prof, ls, lw=2, label=lbl)
    axes[0].set_title("Week 8 Verification: Grid Convergence at y = 200 km (t = 6 hr)", fontweight="bold")
    axes[0].set_xlabel("East-West Distance [km]")
    axes[0].set_ylabel("Bird Migration Density [birds / km²]")
    axes[0].legend()
    axes[0].grid(True, alpha=0.4)

    # Panel B: CFL Stability Comparison
    cfl_styles = {
        0.5: ("g-", "CFL = 0.5 (Stable)"),
        0.9: ("b--", "CFL = 0.9 (Stable, within 2D ADR limit)"),
        1.5: ("r-", "CFL = 1.5 (Unstable Blow-Up)"),
    }
    for cfl_val in cfl_cases:
        t_arr, p_arr = cfl_histories[cfl_val]
        ls, lbl = cfl_styles[cfl_val]
        axes[1].semilogy(t_arr, p_arr, ls, lw=2, label=lbl)
    axes[1].set_title("Week 6 Verification: RK4 CFL Stability Limit Test", fontweight="bold")
    axes[1].set_xlabel("Time [hr]")
    axes[1].set_ylabel("Max Domain Density (log scale)")
    axes[1].set_ylim(10, 1e13)
    axes[1].legend()
    axes[1].grid(True, alpha=0.4)

    plt.tight_layout()
    plt.savefig("fig3_cfd_verification.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    print("=================================================================")
    print(" 2D CFD NOCTURNAL BIRD MIGRATION SIMULATOR - FULL SUITE")
    print("=================================================================")
    run_weather_scenarios(nx=100, ny=100)
    run_cfd_verification()
    print("\nAll figures saved successfully:")
    print("  1. fig1_weather_scenarios.png")
    print("  2. fig2_migration_diagnostics.png")
    print("  3. fig3_cfd_verification.png")