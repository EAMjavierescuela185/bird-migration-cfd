# 2D CFD Modeling of Nocturnal Bird Migration Corridors
**Course:** ENGR4576/5576 – Computational Fluid Dynamics  
**GitHub Repository:** [EAMjavierescuela185/bird-migration-cfd](https://github.com/EAMjavierescuela185/bird-migration-cfd)

---

## 1. The Problem Statement
Large-scale nocturnal bird migration behaves like an active fluid continuum advected by atmospheric winds and guided by geographic boundaries. Inspired by BirdCast regional migration radar forecasts, this project develops a 2D CFD solver in Python to model the evolution of nocturnal bird migration density $\rho(x, y, t)$ across a $400\text{ km} \times 400\text{ km}$ Mid-Atlantic flyway covering Eastern Pennsylvania, New Jersey, Delaware, and Maryland. The model investigates how synoptic wind direction, coastline avoidance, mountain-ridge channeling, and dawn landing behavior interact to form high-density migratory bottlenecks.

---

## 2. Method of Solution
Bird migration density $\rho(x, y, t)$ [$\text{birds/km}^2$] is governed by the 2D unsteady Advection-Diffusion-Reaction (ADR) equation:

$$\frac{\partial \rho}{\partial t} + \frac{\partial (u_{\text{eff}} \rho)}{\partial x} + \frac{\partial (v_{\text{eff}} \rho)}{\partial y} = D \left( \frac{\partial^2 \rho}{\partial x^2} + \frac{\partial^2 \rho}{\partial y^2} \right) + S(x, y, t)$$

where $\vec{V}_{\text{eff}} = \vec{V}_{\text{bird}} + \alpha \vec{V}_{\text{wind}} + \vec{V}_{\text{coast}} + \vec{V}_{\text{ridge}}$.

* **Effective Velocity Field ($\vec{V}_{\text{eff}}$):** Combines an active southwest autumn flight heading ($28\text{ km/hr}$) with partial wind drift ($\alpha \vec{V}_{\text{wind}}$), shoreline repulsion along the Atlantic Ocean and Chesapeake Bay (the Delmarva Peninsula), and topographic channeling along the Appalachian/Kittatinny Ridge.
* **Spatial Discretization *(studied in Week 6)*:** Solved on a uniform 2D Cartesian mesh using first-order upwind differencing (flux splitting based on local velocity signs) for convective terms to eliminate dispersion oscillations and maintain $\rho \ge 0$, paired with second-order central differencing for flock dispersion ($D = 8.0\text{ km}^2/\text{hr}$).
* **Boundary & Source Conditions:** A time-varying Dirichlet inflow boundary at the northern edge ($y = 400\text{ km}$) models migrants entering from NY and New England overnight ($t \le 5.5\text{ hr}$), zero-gradient Neumann conditions govern outflow boundaries, and a dawn landing sink term ($S < 0$ for $t \ge 6.5\text{ hr}$) depletes airborne density exclusively over land.
* **Time Integration & Stability *(studied in Week 6)*:** Integrated explicitly using the 4-stage Runge-Kutta (RK4) method with time steps bounded by the 2D advection-diffusion CFL stability criterion.
* **Numerical Verification *(studied in Week 6 & Week 8)*:** Validated through a 3-grid convergence study ($50\times 50$, $100\times 100$, and $150\times 150$ nodes) with Richardson extrapolation and a CFL stability sweep.

---

## 3. What Has Been Accomplished
* **Modular Python Solver Built:** Implemented `domain.py` (Mid-Atlantic coastline/bay masks, Appalachian ridge profile, bounded velocity fields, and dusk stopover initial conditions), `solver.py` (2D upwind + central finite-difference residual, RK4 integrator, northern inflow BC, and dawn landing sink), and `main.py` (scenario comparison and numerical verification suite).
* **9-Hour [3 hr. frames] Nocturnal Simulation Completed:** Simulated overnight migration from dusk takeoff ($t = 0\text{ hr}$) through peak nocturnal passage ($t = 3\text{–}6\text{ hr}$) and post-dawn landing ($t = 9\text{ hr}$).

![Domain Verification](step1_domain_verification.png)
*Figure 1: Mid-Atlantic domain geography, dusk takeoff stopover pulses (left), and bounded effective migration velocity field (right).*

![9-Hour Migration Evolution](step2_migration_evolution.png)
*Figure 2: Spatiotemporal evolution of nocturnal bird migration density over a 9-hour cycle (3-hour frames).*

* **Weather Scenario Comparison Completed:** Generated comparative BirdCast-style density/vector maps (`fig1_weather_scenarios.png`) and airborne population time-series (`fig2_migration_diagnostics.png`) contrasting favorable NW tailwinds against an offshore westerly crosswind.

![Weather Scenarios](fig1_weather_scenarios.png)
*Figure 3: Weather impact on nocturnal migration corridors at t = 6.0 hr comparing Scenario A (NW Tailwinds) and Scenario B (Westerly Crosswind coastal funneling onto the Delmarva Peninsula).*

![Migration Diagnostics](fig2_migration_diagnostics.png)
*Figure 4: Quantitative time-series of total airborne migrants over the region (left) and peak flyway concentration vs. time (right).*

* **CFD Verification Completed:** Executed a 3-grid convergence study across $50\times 50$, $100\times 100$, and $150\times 150$ meshes alongside an RK4 CFL stability limit test (`fig3_cfd_verification.png`).

![CFD Verification](fig3_cfd_verification.png)
*Figure 5: Week 8 grid convergence cross-section at y = 200 km (left) and Week 6 RK4 CFL stability limit test showing stable trajectories (CFL = 0.5, 0.9) vs. unstable blow-up (CFL = 2.3) (right).*

---

## 4. What Needs to Be Done
1. **CFL Upper-Bound Refinement:** Finalize documentation of the high-CFL instability case (`CFL = 2.3`) in `main.py` alongside the stable `CFL = 0.5` and `0.9` trajectories.
2. **Update Current Data Comparison:** Incorporate current live data and metrics to simulated travel fields of a select bird species.
3. **Final Technical Report:** Compile the final project report discussing physical flow features, Grid Convergence Index (GCI) metrics, and modeling limitations.

---

## 5. Current Bottlenecks
The Python solver and visualization pipeline are fully functional and execute in under 10 seconds locally. Remaining constraints include adjusting known solvers and equations to incorporate additional variables and finer fields, comparing against accurate tracking data, and further refining visualization details to resemble advanced radar forecast simulations.

---

## 6. Predicted Results & Preliminary Findings

| Test Case / Resolution | Grid Spacing $\Delta x$ | Peak Density ($t = 6\text{ hr}$) | Key Physical / Numerical Observation |
| :--- | :---: | :---: | :--- |
| **Coarse Grid ($50\times 50$)** | $8.16\text{ km}$ | $57.4\text{ birds/km}^2$ | Higher numerical dissipation smears corridor peak |
| **Medium Grid ($100\times 100$)** | $4.04\text{ km}$ | $68.6\text{ birds/km}^2$ | Resolves Appalachian ridge & Delmarva bottleneck |
| **Fine Grid ($150\times 150$)** | $2.68\text{ km}$ | $74.2\text{ birds/km}^2$ | Monotone convergence toward continuum limit |

Under favorable NW tailwinds (Scenario A), peak density diffuses smoothly from $130\text{ birds/km}^2$ down to $\sim 79\text{ birds/km}^2$ prior to dawn as flocks disperse evenly across Pennsylvania. Under strong westerly crosswinds (Scenario B), birds are advected toward the Atlantic shoreline and bottleneck onto the **Delmarva Peninsula**, increasing peak concentration to **$\sim 170\text{ birds/km}^2$** at $t = 6.5\text{ hr}$ before dawn landing depletes airborne density over land.

---

## How to Run Locally
```bash
pip install -r requirements.txt
python domain.py   # Generates step1_domain_verification.png
python solver.py   # Generates step2_migration_evolution.png
python main.py     # Generates fig1, fig2, fig3 and prints convergence metrics
```