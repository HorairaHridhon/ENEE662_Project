import numpy as np
import cvxpy as cp
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',       # consistent math fonts
    'font.size': 12,                  # Base font size
    'axes.labelsize': 14,             # Axis label size
    'legend.fontsize': 11,            # Legend size
    'xtick.labelsize': 12,            # Tick label size
    'ytick.labelsize': 12,
    'axes.linewidth': 1.2,            # Thicker axis border
    'grid.alpha': 0.3,                # Subtle grid
    'lines.linewidth': 2.0,           # Thicker plot lines
    'figure.figsize': (8, 5),         # Standard aspect ratio
    'savefig.dpi': 600,               # High resolution for PNG
    'savefig.bbox': 'tight',          # Avoid cutting off labels
})

# ==========================================
# 1. HELPER FUNCTIONS
# ==========================================

def compute_steering_vector(positions, theta, phi, k):
    """
    Creates a steering vector for a specific direction.
    Returns shape (N_elements,)
    """
    u = np.sin(theta) * np.cos(phi)
    v = np.sin(theta) * np.sin(phi)
    # Definition: exp(j * k * (ux + vy))
    sv = np.exp(1j * k * (positions[:, 0]*u + positions[:, 1]*v))
    return sv

def compute_pattern_db(positions, weights, theta_grid, phi_grid, k):
    """
    Computes the array factor in dB.
    """
    u = np.sin(theta_grid) * np.cos(phi_grid)
    v = np.sin(theta_grid) * np.sin(phi_grid)

    # Phase matrix: (N, N_points)
    phase = np.exp(1j * k * (np.outer(positions[:,0], u) + np.outer(positions[:,1], v)))

    # Array Factor = w . exp(...)
    af = weights @ phase
    af_mag = np.abs(af)
    af_mag = np.maximum(af_mag, 1e-10) # Avoid log(0)
    return 20 * np.log10(af_mag / np.max(af_mag))

def get_geometry_and_constants():
    """
    Sets up the 16x16 square array geometry.
    Returns: positions, k, N
    """
    c = 3e8
    freq = 1e9
    lam = c / freq
    k = 2 * np.pi / lam

    Nx, Ny = 16, 16
    dx, dy = 0.3 * lam, 0.3 * lam
    x = np.arange(Nx) * dx - (Nx-1)*dx/2
    y = np.arange(Ny) * dy - (Ny-1)*dy/2
    X, Y = np.meshgrid(x, y)
    positions = np.column_stack((X.flatten(), Y.flatten()))
    N = len(positions)
    return positions, k, N

def get_coupling_matrix(positions, k, N):
    """
    Computes Matrix P (Power Matrix) for minimizing total power.
    P_mn = sinc(k * |r_m - r_n|) for isotropic elements.
    """
    r_diff = positions[:, None, :] - positions[None, :, :]
    dists = np.linalg.norm(r_diff, axis=2)
    P = np.sinc(k * dists / np.pi) # numpy sinc is sin(pi*x)/(pi*x)

    # Regularization for numerical stability
    P += 1e-6 * np.eye(N)

    # Cholesky Decomposition P = G^H * G
    L = np.linalg.cholesky(P)
    G = L.T.conj()
    return G

# ==========================================
# 2. CORE OPTIMIZATION
# ==========================================

def run_optimization(scenario_type):
    """
    Runs the SOCP optimization based on the scenario type.
    Options: 'Fig5a', 'Fig5b', 'Fig6'
    """
    positions, k, N = get_geometry_and_constants()
    G = get_coupling_matrix(positions, k, N)

    # Optimization Variable
    w = cp.Variable(N, complex=True)
    constraints = []

    # --- Common Constraint: Main Beam at 30 deg ---
    theta0 = np.radians(30)
    phi0 = 0
    d_main = compute_steering_vector(positions, theta0, phi0, k)
    constraints.append( w @ d_main == 1.0 )

    # --- Scenario Specific Constraints ---
    null_angles = [-20, 70]

    if scenario_type == 'Fig5a':
        print("\n--- Running Optimization: Figure 5a (Max Directivity, No Nulls) ---")
        # No nulls, No SLL
        pass

    elif scenario_type == 'Fig5b':
        print("\n--- Running Optimization: Figure 5b (Max Directivity + Nulls) ---")
        # Add Null Constraints
        for ang in null_angles:
            d_null = compute_steering_vector(positions, np.radians(ang), 0, k)
            constraints.append( w @ d_null == 0 )

    elif scenario_type == 'Fig6':
        print("\n--- Running Optimization: Figure 6 (Nulls + SLL -20dB) ---")
        # 1. Add Null Constraints
        for ang in null_angles:
            d_null = compute_steering_vector(positions, np.radians(ang), 0, k)
            constraints.append( w @ d_null == 0 )

        # 2. Add SLL Constraints (-20dB)
        sll_limit = 10**(-20/20)
        theta_scan = np.linspace(-90, 90, 361)

        # Mask: Exclude Main Beam (+/- 10 deg) and Nulls (+/- 2 deg)
        mask_main = np.abs(theta_scan - 30) > 10
        mask_null1 = np.abs(theta_scan - (-20)) > 2
        mask_null2 = np.abs(theta_scan - 70) > 2
        sll_indices = np.where(mask_main & mask_null1 & mask_null2)[0]

        # Subsample constraints for speed (every 4th point)
        for idx in sll_indices[::4]:
            th = np.radians(theta_scan[idx])
            d_sll = compute_steering_vector(positions, th, 0, k)
            constraints.append( cp.abs(w @ d_sll) <= sll_limit )

    # --- Objective: Minimize Power (Maximize Directivity) ---
    # Minimize Norm of G*w (Equivalent to minimizing w^H P w)
    obj = cp.Minimize(cp.norm(G @ w, 2))
    prob = cp.Problem(obj, constraints)

    print("Solving...")
    try:
        prob.solve(solver=cp.CLARABEL)
    except:
        print("CLARABEL failed or not installed, switching to SCS...")
        prob.solve(solver=cp.SCS, eps=1e-4)

    print(f"Status: {prob.status}")
    return w.value, positions, k

# ==========================================
# 3. PLOTTING FUNCTION
# ==========================================


def plot_results(w_opt, positions, k, title, filename, show_sll_mask=False, show_nulls=False):
    if w_opt is None:
        print(f"Optimization failed for {title}. No plot generated.")
        return

    theta_plot = np.linspace(-np.pi/2, np.pi/2, 1000)
    phi_plot = np.zeros_like(theta_plot)

    # 1. Compute Patterns
    pat_opt = compute_pattern_db(positions, w_opt, theta_plot, phi_plot, k)

    theta0 = np.radians(30)
    d_steer = compute_steering_vector(positions, theta0, 0, k)
    w_uni = np.conj(d_steer)
    pat_uni = compute_pattern_db(positions, w_uni, theta_plot, phi_plot, k)

    # 2. Plotting
    fig, ax = plt.subplots(figsize=(7, 4.5))

    if show_sll_mask:
        ax.fill_between([-90, 90], -20, -100, color='#e0e0e0', alpha=0.5, label='SLL Region')

    # Colors: 'gray' for reference, distinct blue for optimization
    ax.plot(np.degrees(theta_plot), pat_uni, ':', color='gray', linewidth=1.5, label='W/O Opt')
    ax.plot(np.degrees(theta_plot), pat_opt, '-', color='#004C99', linewidth=2.5, label='Proposed SOCP')

    # --- CONDITIONAL NULL MARKERS ---
    if show_nulls:
        ax.axvline(-20, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(70, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.text(-20, -20, 'Null', ha='center', va='bottom', fontsize=20)
        ax.text(70, -20, 'Null', ha='center', va='bottom', fontsize=20)

    if show_sll_mask:
        ax.axhline(-20, color='#D32F2F', linestyle='--', linewidth=1.5, label='SLL Constraint (-20 dB)')

    # Formatting
    ax.set_ylim(-50, 0)
    ax.set_xlim(-90, 90)
    ax.set_xlabel(r'Angle $\theta$ (degrees)', fontsize=16)
    ax.set_ylabel(r'Normalized Pattern (dB)', fontsize=16)
    ax.set_title(title, pad=15, fontsize=20)

    ax.legend(loc='upper left', frameon=True, framealpha=0.9, edgecolor='white', fontsize=14)
    ax.grid(True, linestyle='--', alpha=0.4)

    # 3. Saving
    print(f"Saving {filename}...")
    plt.savefig(f"{filename}.pdf", format='pdf')
    plt.savefig(f"{filename}.png", format='png')
    plt.show()

# ==========================================
# 4. MAIN EXECUTION
# ==========================================


if __name__ == "__main__":
    # --- Run Figure 5a: Max Directivity (No Nulls) ---
    w_5a, pos, k = run_optimization('Fig5a')
    # show_nulls=False here avoids the dotted lines and text
    plot_results(w_5a, pos, k, "Maximize Directivity (No Nulls)", "Fig5a_Directivity", show_nulls=False)

    # --- Run Figure 5b: Max Directivity + Nulls ---
    w_5b, pos, k = run_optimization('Fig5b')
    # show_nulls=True here adds the markers back
    plot_results(w_5b, pos, k, "Maximize Directivity + Nulls", "Fig5b_Nulls", show_nulls=True)

    # --- Run Figure 6: Max Directivity + Nulls + SLL ---
    w_6, pos, k = run_optimization('Fig6')
    plot_results(w_6, pos, k, "Broadband SLL Control", "Fig6_SLL_Control", show_sll_mask=True, show_nulls=True)
