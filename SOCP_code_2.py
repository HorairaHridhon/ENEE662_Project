import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 14,
    'axes.labelsize': 14,
    'legend.fontsize': 16,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.3,
    'lines.linewidth': 2.0,
    'figure.figsize': (8, 5),
    'savefig.dpi': 600,
    'savefig.bbox': 'tight',
})


# PHYSICS & SOLVER FUNCTIONS

def compute_steering_vector_circ(positions, theta, phi, k):
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    return np.exp(1j * k * (positions[:, 0:1] * u + positions[:, 1:2] * v))

def compute_pattern_db_circ(positions, weights, theta_val, phi_grid, k):
    if weights is None:
        return np.zeros_like(phi_grid) - 100

    u = np.sin(theta_val) * np.cos(phi_grid)
    v = np.sin(theta_val) * np.sin(phi_grid)
    phase = np.exp(1j * k * (np.outer(positions[:, 0], u) + np.outer(positions[:, 1], v)))

    af = weights @ phase
    af_mag = np.abs(af)
    af_mag = np.maximum(af_mag, 1e-10)
    return 20 * np.log10(af_mag / np.max(af_mag))

def solve_socp(positions, k, G, beam_gap_deg, sll_near_db, sll_far_db):
    N_elements = positions.shape[0]
    w = cp.Variable(N_elements, complex=True)
    constraints = []

    theta0 = np.radians(90)
    phi0 = np.radians(30)

    # 1. Main Beam (Fixed)
    d_main = compute_steering_vector_circ(positions, theta0, phi0, k)
    constraints.append(cp.real(w @ d_main) == 1.0)
    constraints.append(cp.imag(w @ d_main) == 0.0)

    # 2. Null Constraint
    phi_null = np.radians(-150)
    d_null = compute_steering_vector_circ(positions, theta0, phi_null, k)
    constraints.append(cp.abs(w @ d_null) <= 1e-3)

    # 3. SLL Constraints
    phi_scan_deg = np.linspace(-180, 180, 180)
    limit_near = 10**(sll_near_db/20)
    limit_far = 10**(sll_far_db/20)

    for ang in phi_scan_deg:
        diff = abs(ang - 30)
        if diff <= beam_gap_deg/2: continue
        if abs(ang - (-150)) < 5: continue

        current_limit = limit_far
        if diff < 60: current_limit = limit_near

        d_sll = compute_steering_vector_circ(positions, theta0, np.radians(ang), k)
        constraints.append(cp.abs(w @ d_sll) <= current_limit)

    obj = cp.Minimize(cp.norm(G @ w, 2))
    prob = cp.Problem(obj, constraints)

    # Solver fallback sequence
    try:
        prob.solve(solver=cp.CLARABEL)
    except:
        try:
            prob.solve(solver=cp.ECOS)
        except:
            prob.solve(solver=cp.SCS, max_iters=5000)

    return prob.status, w.value


# PLOTTING FUNCTION

def plot_publication_figure(phi_deg, pat_opt, pat_classic, beam_gap, filename="figure7_pub"):

    fig, ax = plt.subplots(figsize=(7, 4.5))

    # Reference Pattern
    ax.plot(phi_deg, pat_classic, ':', color='gray', linewidth=1.5, label='W/O Opt')

    # Optimized Pattern
    ax.plot(phi_deg, pat_opt, '-', color='#004C99', linewidth=2.5, label='Proposed SOCP')

    # Visual Masks
    ax.fill_between([-180, 180], -15, -100, color='#e0e0e0', alpha=0.4, label='SLL Target (-15 dB)')

    # Beam Gap Visualization
    if beam_gap is not None:
        ax.axvspan(30 - beam_gap/2, 30 + beam_gap/2, alpha=0.1, color='orange')

    # Markers
    # Main Beam
    ax.axvline(30, color='#D32F2F', linestyle='--', linewidth=1.0, alpha=0.8)
    ax.text(30, 2, 'Main Beam', ha='center', va='bottom', fontsize=16, color='#D32F2F', weight='bold')

    # Null
    ax.axvline(-150, color='black', linestyle='--', linewidth=1.2, alpha=0.8)
    ax.text(-150, -10, 'Null', ha='center', va='bottom', fontsize=20)

    # Formatting
    ax.set_ylim(-50, 5)
    ax.set_xlim(-180, 180)
    ax.set_xlabel(r'Azimuth Angle $\phi$ (deg)', fontsize=16)
    ax.set_ylabel(r'Normalized Gain (dB)', fontsize=16)

    ax.legend(loc='lower right', frameon=True, framealpha=0.95, edgecolor='white', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)

    print(f"Saving {filename}...")
    plt.savefig(f"{filename}.pdf", format='pdf')
    plt.savefig(f"{filename}.png", format='png')
    plt.show()


# MAIN EXECUTION

def main():
    print("=== Starting Optimization ===")

    # Setup Physics
    c = 3e8; freq = 1e9; lam = c / freq; k = 2 * np.pi / lam
    N_elements = 32; radius = 4.0 * lam

    # Positions
    angles = np.linspace(0, 2*np.pi, N_elements, endpoint=False)
    positions = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])

    # Coupling Matrix (P/G)
    r_diff = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
    distances = np.linalg.norm(r_diff, axis=2)
    with np.errstate(divide='ignore', invalid='ignore'):
        P = np.sinc(2 * distances / lam)
    np.fill_diagonal(P, 1.0)
    P += 1e-8 * np.eye(N_elements)

    try:
        L = np.linalg.cholesky(P)
        G = L.T.conj()
    except np.linalg.LinAlgError:
        P += 1e-6 * np.eye(N_elements)
        L = np.linalg.cholesky(P)
        G = L.T.conj()

    # --- ATTEMPTS LOOP ---
    attempts = [
        (40, -8, -15),    # Relaxed
        (45, -10, -18),   # Medium
        (50, -12, -20),   # Target
    ]

    opt_weights = None
    best_status = None
    final_gap = None

    for i, (gap, near_db, far_db) in enumerate(attempts):
        print(f"\nAttempt {i+1}/{len(attempts)}: Gap={gap}, Near={near_db}, Far={far_db}")
        status, weights = solve_socp(positions, k, G, gap, near_db, far_db)
        print(f"  - Status: {status}")

        if status in ['optimal', 'optimal_inaccurate']:
            print(f"  - ✓ Optimization successful!")
            opt_weights = weights
            best_status = status
            final_gap = gap
            break

    # Fallback if everything fails
    if opt_weights is None:
        print("\n⚠️ All SOCP attempts failed. Using simple beamforming...")
        theta0 = np.radians(90)
        phi0 = np.radians(30)
        d_main = compute_steering_vector_circ(positions, theta0, phi0, k)
        opt_weights = d_main.conj().flatten()
        best_status = "conjugate"

    # Compute Patterns
    print("\nComputing patterns...")
    theta0 = np.radians(90)
    phi_plot_deg = np.linspace(-180, 180, 721)
    phi_plot_rad = np.radians(phi_plot_deg)

    # Proposed
    pat_opt_db = compute_pattern_db_circ(positions, opt_weights, theta0, phi_plot_rad, k)

    # Classic (Conjugate Beamforming)
    phi_beam = np.radians(30)
    d_main = compute_steering_vector_circ(positions, theta0, phi_beam, k)
    w_classic = d_main.conj().flatten()
    pat_classic_db = compute_pattern_db_circ(positions, w_classic, theta0, phi_plot_rad, k)

    # Generate Publication Plot
    plot_publication_figure(phi_plot_deg, pat_opt_db, pat_classic_db, final_gap, filename="Fig7_CircularArray")

if __name__ == "__main__":
    main()
