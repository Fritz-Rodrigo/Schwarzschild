
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp, quad
from scipy.optimize import brentq

# ============================================================
# Parameters: Hartle's schorbits.nb
# ============================================================
signewt = 1.0       # 1: Schwarzschild, 0: Newtonian
l = 4.5             # Specific angular momentum
E = -0.25           # Energy parameter (e^2 - 1)/2
rst = 20.0          # Starting radius for unbound orbits
norbit = 6          # Number of radial cycles for bound orbits
mode = "bound"      # "bound", "scatter", or "plunge"
eps = 1e-8

# M = 1; r is measured in units of M; u = 1/r.
def V(u):
    return -u + 0.5 * l**2 * u**2 - signewt * l**2 * u**3

def dV(u):
    return -1.0 + l**2 * u - 3.0 * signewt * l**2 * u**2

def radial_speed_squared(u):
    return 2.0 * (E - V(u)) / l**2

# ============================================================
# Effective potential, extrema, and turning points
# ============================================================
print(f"l = {l}, E = {E}, signewt = {signewt}")

# V'(u) = 0. Find extrema.
extrema = np.roots(
    [-3 * signewt * l**2, l**2, -1.0]
    if signewt != 0
    else [l**2, -1.0]
)
extrema = np.sort(extrema[np.isreal(extrema)].real)
extrema = extrema[extrema > 0]

for u_ext in extrema:
    kind = "minimum" if dV(u_ext + 1e-6) > dV(u_ext - 1e-6) else "maximum"
    print(f"Potential {kind}: u = {u_ext:.8f}, "
          f"r = {1/u_ext:.8f}, V = {V(u_ext):.8f}")

# Solve V(u) = E.
coeff = (
    [-signewt * l**2, 0.5 * l**2, -1.0, -E]
    if signewt != 0
    else [0.5 * l**2, -1.0, -E]
)
roots = np.roots(coeff)
turning_points = np.sort(
    roots[np.isreal(roots)].real[
        (roots[np.isreal(roots)].real > 0)
    ]
)
print("Positive turning points in u:", turning_points)
print("Corresponding radii:", 1.0 / turning_points)

# Plot effective potential against radius.
r = np.linspace(2.001, 100, 2000)
plt.figure()
plt.plot(r, V(1.0 / r), label=r"$V(1/r)$")
plt.axhline(E, color="red", linestyle="--", label="E")
plt.ylim(bottom=-0.15)
plt.xlabel(r"$r/M$")
plt.ylabel("V")
plt.title("Schwarzschild effective potential")
plt.grid(True)
plt.legend()
plt.tight_layout()

# ============================================================
# Orbit equation
# u'' = 1/l^2 - u + 3 sigma u^2
# ============================================================
def orbit_equation(phi, y):
    u, du = y
    return [du, 1.0 / l**2 - u + 3.0 * signewt * u**2]

# Stop a plunge orbit at the Schwarzschild horizon r = 2M.
def horizon(phi, y):
    return y[0] - 0.5

horizon.terminal = True
horizon.direction = 1

# ============================================================
# Initial conditions
# ============================================================
if mode == "bound":
    if len(turning_points) < 2 or E >= 0:
        raise ValueError("No bound orbit for these parameters.")

    # For the usual bound orbit, the two outer roots bound
    # the allowed interval in u: apocenter <-> pericenter.
    u_outer, u_inner = turning_points[:2]

    if not (E >= V(u_outer + 1e-6)):
        pass

    # Radial half-period in azimuth, using a substitution
    # that removes the square-root endpoint singularities.
    midpoint = 0.5 * (u_outer + u_inner)
    halfwidth = 0.5 * (u_inner - u_outer)

    def transformed_integrand(theta):
        u = midpoint - halfwidth * np.cos(theta)
        du_dtheta = halfwidth * np.sin(theta)
        if abs(du_dtheta) < 1e-14:
            return 0.0
        return (
            l * du_dtheta
            / np.sqrt(2.0 * (E - V(u)))
        )

    # Integral from apocenter to pericenter.
    half_angle = quad(
        transformed_integrand, 0.0, np.pi,
        epsabs=1e-8, limit=300
    )[0]
    radial_period = 2.0 * half_angle
    phi_max = norbit * radial_period

    # Start just inside apocenter, moving inward.
    u0 = u_outer + eps
    du0 = np.sqrt(max(0.0, radial_speed_squared(u0)))
    y0 = [u0, du0]

elif mode in ("scatter", "plunge"):
    u0 = 1.0 / rst
    q = radial_speed_squared(u0)

    if q < 0:
        raise ValueError(
            "Initial radius is classically forbidden: E < V(1/rst)."
        )

    du0 = np.sqrt(q)  # Incoming motion: u increases with phi.
    y0 = [u0, du0]

    # Set a finite angular interval for an unbound orbit.
    phi_max = 20.0 * np.pi

else:
    raise ValueError("mode must be 'bound', 'scatter', or 'plunge'.")

# ============================================================
# Numerical integration
# ============================================================
events = horizon if mode == "plunge" else None

sol = solve_ivp(
    orbit_equation,
    (0.0, phi_max),
    y0,
    method="DOP853",
    rtol=1e-10,
    atol=1e-12,
    max_step=0.02,
    events=events,
    dense_output=False
)

if not sol.success:
    raise RuntimeError(sol.message)

u = sol.y[0]
phi = sol.t

# Convert to Cartesian coordinates.
valid = u > 0
r_orbit = 1.0 / u[valid]
x = r_orbit * np.cos(phi[valid])
y = r_orbit * np.sin(phi[valid])

# ============================================================
# Display accumulated angle accphi(z)
# ============================================================
# u1 and u2 are the two turning points bounding the bound orbit,
# ordered as u1 < u2.

# u1, u2 = turning_points[:2]

# def theta(u):
#     integrand = lambda w: l / np.sqrt(2.0 * (E - V(w)))
#     return quad(integrand, u1, u, epsabs=1e-9, limit=300)[0]

# delphi = theta(u2)
# print(f"delphi = {delphi:.8f} rad")

# def accphi_hartle(z):
#     n = int(np.floor(z))
#     zf = z - n

#     if zf < 0.5:
#         u = u1 * (1.0 - 2.0 * zf) + u2 * (2.0 * zf)
#         return 2.0 * n * delphi + theta(u)
#     else:
#         u = u1 * (2.0 * zf - 1.0) + 2.0 * u2 * (1.0 - zf)
#         return 2.0 * (n + 1) * delphi - theta(u)

# z_values = np.linspace(0.0, norbit, 1000)
# accphi_values = np.array([accphi_hartle(z) for z in z_values])

# # Hartle In[90]: Plot accphi(z) over the first radial cycle
# if norbit > 1:
#     z_plot = np.linspace(0.0, 1.0, 300)
#     phi_plot = np.array([accphi_hartle(z) for z in z_plot])

#     plt.figure(figsize=(8, 4.5))
#     plt.plot(z_plot, phi_plot)
#     plt.xlabel("z")
#     plt.ylabel("phi")
#     plt.title("Accumulated angle during the first radial cycle")
#     plt.xlim(0, 1)
#     plt.grid(True)
#     plt.tight_layout()
#     plt.show()

# # Hartle In[91]: Plot accphi(z) over all cycles
# plt.figure(figsize=(8, 4.5))
# plt.plot(z_values, accphi_values)
# plt.xlabel("z")
# plt.ylabel("accphi (radians)")
# plt.title("Accumulated azimuthal angle")
# plt.xticks(np.arange(0, norbit + 1))
# plt.grid(True)
# plt.tight_layout()
# plt.show()

# ============================================================
# Display orbit
# ============================================================
fig, ax = plt.subplots(figsize=(8, 8))
ax.plot(x, y, linewidth=1.0)
ax.set_xlabel("x / M")
ax.set_ylabel("y / M")
ax.set_title(f"Schwarzschild orbit: l={l}, E={E}, "
             f"sigma={signewt}")
ax.set_aspect("equal", adjustable="datalim")
ax.grid(True)
plt.tight_layout()
plt.show()