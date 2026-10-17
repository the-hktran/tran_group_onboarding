# Module 06: Adiabatic Molecular Dynamics & Tully's Fewest-Switches Surface Hopping (FSSH)

Understanding chemical reactivity, photochemistry, and spin-phonon or electron-phonon relaxation requires moving beyond static potential energy surfaces to simulate how nuclei and electrons evolve in time. This module walks you through **Adiabatic Born–Oppenheimer Molecular Dynamics (BOMD)** using the [Velocity Verlet integrator](https://www.google.com/search?q=https%3A%2F%2Fdocs.matlantis.com%2Fatomistic-simulation-tutorial%2Fen%2F6_1_md-nve.html) and **Nonadiabatic Mixed Quantum–Classical Dynamics** via Tully's [Fewest-Switches Surface Hopping (FSSH)](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Facsodf%2Farticle%2F7%2F50%2F45810%2F461699%2FPedagogical-Overview-of-the-Fewest-Switches).

## 1. Adiabatic Born–Oppenheimer Molecular Dynamics (BOMD)

Under the Born–Oppenheimer approximation, nuclei of mass $M_I$ move classically on a single adiabatic electronic potential energy surface $E_0(\mathbf{R})$:
$$M_I \frac{d^2 \mathbf{R}_I}{dt^2} = \mathbf{F}_I(\mathbf{R}) = -\nabla_{\mathbf{R}_I} E_0(\mathbf{R})$$

### Symplectic Integration with Velocity Verlet
Standard Euler or Runge–Kutta integrators drift in energy over long molecular dynamics trajectories. The [Velocity Verlet algorithm](https://www.google.com/search?q=https%3A%2F%2Fdocs.matlantis.com%2Fatomistic-simulation-tutorial%2Fen%2F6_1_md-nve.html) is time-reversible and symplectic (conserving phase-space volume and keeping total energy $E_{\text{tot}} = K + V$ bounded over millions of steps):

1. **Half-step velocity update**:
   $$\mathbf{v}_I\!\left(t + \frac{\Delta t}{2}\right) = \mathbf{v}_I(t) + \frac{1}{2} \frac{\mathbf{F}_I(\mathbf{R}(t))}{M_I} \Delta t$$
2. **Full-step position update**:
   $$\mathbf{R}_I(t + \Delta t) = \mathbf{R}_I(t) + \mathbf{v}_I\!\left(t + \frac{\Delta t}{2}\right) \Delta t$$
3. **Electronic structure & nuclear gradient evaluation** at $\mathbf{R}(t + \Delta t)$:
   $$\mathbf{F}_I(\mathbf{R}(t + \Delta t)) = -\nabla_{\mathbf{R}_I} E_0(\mathbf{R}(t + \Delta t))$$
4. **Second half-step velocity update**:
   $$\mathbf{v}_I(t + \Delta t) = \mathbf{v}_I\!\left(t + \frac{\Delta t}{2}\right) + \frac{1}{2} \frac{\mathbf{F}_I(\mathbf{R}(t + \Delta t))}{M_I} \Delta t$$

* **Choosing the Time Step ($\Delta t$)**: The time step must resolve the fastest vibrational motion (typically $\text{O–H}$ or $\text{C–H}$ stretches with period $\tau \approx 10\text{ fs}$), requiring $\Delta t \approx 0.2\text{–}0.5\text{ fs}$ ($1\text{ fs} \approx 41.341\text{ a.u.}$ of time).
* **Running Ab Initio MD in Practice**: You can drive `nucgrad` forces from [PySCF](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fuser%2Findex.html) through the [Atomic Simulation Environment (ASE) interface](https://www.google.com/search?q=https%3A%2F%2Fyang.chem.wisc.edu%2Fwp-content%2Fuploads%2Fsites%2F1145%2F2023%2F12%2FHow-to-CNEO-MD.pdf) or accelerate trajectories by orders of magnitude using machine-learned interatomic potentials ([deep learning potentials for MD and FSSH](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2F10.1021%2Facs.jctc.6c00170)).

## 2. Breakdown of Born–Oppenheimer & Nonadiabatic Coupling Vectors (NACVs)

Near avoided crossings or conical intersections where two adiabatic electronic states $\Phi_j(\mathbf{r};\mathbf{R})$ and $\Phi_k(\mathbf{r};\mathbf{R})$ become nearly degenerate ($E_j(\mathbf{R}) \approx E_k(\mathbf{R})$), nuclear motion induces electronic transitions. The strength of this coupling is governed by the first-order **nonadiabatic derivative coupling vector (NACV)**:
$$\mathbf{d}_{jk}(\mathbf{R}) = \langle \Phi_j(\mathbf{r};\mathbf{R}) | \nabla_{\mathbf{R}} \Phi_k(\mathbf{r};\mathbf{R}) \rangle = \frac{\langle \Phi_j | \nabla_{\mathbf{R}} \hat{H}_{\text{el}}(\mathbf{R}) | \Phi_k \rangle}{E_k(\mathbf{R}) - E_j(\mathbf{R})} \qquad (j \neq k)$$

Notice the $1 / (E_k - E_j)$ denominator: as the energy gap closes, $\mathbf{d}_{jk}(\mathbf{R})$ spikes sharply! In multi-reference quantum chemistry, analytical NACVs are computed via coupled-perturbed equations in [pyscf.nac](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2Fpyscf_api_docs%2Fpyscf.nac.html) ([SA-CASSCF NAC implementation](https://www.google.com/search?q=https%3A%2F%2Fpyscf.org%2F_modules%2Fpyscf%2Fnac%2Fsacasscf.html)) and interfaced with production surface-hopping packages such as [SHARC](https://www.google.com/search?q=https%3A%2F%2Fsharc-md.org%2F) and [Newton-X](https://www.google.com/search?q=https%3A%2F%2Fnewtonx.org%2F).

### Diabatic vs. Adiabatic Basis in 2-State Model Systems
In a 2-state diabatic basis $\{|\chi_1\rangle, |\chi_2\rangle\}$, the electronic Hamiltonian is a smooth $2 \times 2$ matrix:
$$\mathbf{V}_{\text{diab}}(R) = \begin{pmatrix} V_{11}(R) & V_{12}(R) \\ V_{12}(R) & V_{22}(R) \end{pmatrix}$$
Diagonalizing $\mathbf{V}_{\text{diab}}(R) \mathbf{U}(R) = \mathbf{U}(R) \text{diag}(E_0(R), E_1(R))$ with mixing angle $\theta(R) = \frac{1}{2} \arctan\!\left(\frac{2 V_{12}(R)}{V_{22}(R) - V_{11}(R)}\right)$ yields the adiabatic energies $E_{0,1}(R)$ and an exact analytical formula for the nonadiabatic coupling:
$$d_{01}(R) = \langle \Phi_0 | \frac{d}{dR} \Phi_1 \rangle = \frac{d\theta(R)}{dR}$$

## 3. Tully's Fewest-Switches Surface Hopping ([FSSH](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Facsodf%2Farticle%2F7%2F50%2F45810%2F461699%2FPedagogical-Overview-of-the-Fewest-Switches))

In [Tully's FSSH algorithm](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Facsodf%2Farticle%2F7%2F50%2F45810%2F461699%2FPedagogical-Overview-of-the-Fewest-Switches) (and modern variants like [FSSH-2](https://www.google.com/search?q=https%3A%2F%2Fpubs.acs.org%2Fdoi%2F10.1021%2Facs.jctc.4c00089)), an ensemble of independent nuclear trajectories is propagated. At any instant $t$, each trajectory moves classically on a single **active adiabatic surface** $k$:
$$M \ddot{\mathbf{R}}(t) = -\nabla_{\mathbf{R}} E_k(\mathbf{R}(t))$$
Simultaneously, the complex electronic wavefunction $\Psi_{\text{el}}(\mathbf{r}, t) = \sum_j c_j(t) \Phi_j(\mathbf{r}; \mathbf{R}(t))$ is propagated along the trajectory via the time-dependent Schrödinger equation (TDSE) in the adiabatic basis (in atomic units, $\hbar = 1$):
$$i \dot{c}_j(t) = E_j(\mathbf{R}) c_j(t) - i \sum_l \left( \dot{\mathbf{R}} \cdot \mathbf{d}_{jl}(\mathbf{R}) \right) c_l(t)$$

### Step-by-Step FSSH Algorithm at Each Time Step $\Delta t$

1. **Nuclear Velocity Verlet Step on Active State $k$**:
   Advance $\mathbf{R}(t) \to \mathbf{R}(t + \Delta t)$ and $\mathbf{v}(t) \to \mathbf{v}(t + \Delta t)$ using the active-surface gradient $-\nabla E_k(\mathbf{R})$.
2. **Electronic Sub-Stepping (RK4)**:
   Because electronic phases oscillate faster than nuclear motion, integrate $i \dot{c}_j(t)$ from $t$ to $t + \Delta t$ using $N_{\text{sub}} \approx 10\text{–}20$ fourth-order Runge–Kutta (RK4) sub-steps.
3. **Evaluate Tully's Fewest-Switches Hopping Probability ($g_{k \to j}$)**:
   From the rate of change of the active state population $\dot{\rho}_{kk}$, the probability of hopping from active state $k$ to state $j \neq k$ during step $\Delta t$ is:
   $$g_{k \to j} = \max\!\left(0, \, \frac{2 \, \Delta t}{|c_k|^2} \, \text{Re}\!\left( c_j^* c_k \, \mathbf{v} \cdot \mathbf{d}_{jk}(\mathbf{R}) \right) \right)$$
4. **Stochastic Surface Hop Decision**:
   Draw a uniform random number $\xi \in [0, 1)$. If $\xi < g_{k \to j}$ (in a 2-state system), a surface hop $k \to j$ is attempted!
5. **Momentum Rescaling & Frustrated Hops**:
   To conserve total energy $E_{\text{tot}} = \frac{1}{2} M \mathbf{v}^2 + E_k(\mathbf{R})$ when jumping to surface $j$, the potential energy change $\Delta E = E_j(\mathbf{R}) - E_k(\mathbf{R})$ must be compensated by rescaling the nuclear velocity along the nonadiabatic coupling direction $\hat{\mathbf{d}}_{kj}$:
   * **Allowed Hop ($\frac{1}{2} M v_{\parallel}^2 \ge \Delta E$)**: Rescale the parallel velocity component so that $\frac{1}{2} M v_{\parallel,\text{new}}^2 = \frac{1}{2} M v_{\parallel}^2 - \Delta E$ (preserving the sign of $v_{\parallel}$) and switch the active state $k \leftarrow j$.
   * **Frustrated Hop ($\frac{1}{2} M v_{\parallel}^2 < \Delta E$)**: The nuclei lack sufficient kinetic energy along the coupling vector to climb onto the upper surface $j$. Reject the hop and remain on state $k$.
6. **Energy-Based Decoherence Correction (Granucci–Persico)**:
   After wavepackets pass through an avoided crossing and separate in phase space, standard FSSH stays artificially coherent. Applying an empirical decoherence damping timescale $\tau_{jk} = \frac{\hbar}{|E_j - E_k|}\left(1 + \frac{C}{E_{\text{kin}}}\right)$ (with $C = 0.1\text{ E}_\text{h}$) decays non-active amplitudes $c_{j \neq k} \leftarrow c_j e^{-\Delta t / \tau_{jk}}$ and renormalizes $c_k$.

## 4. Example Project 6: Coding Velocity Verlet & Tully's Avoided Crossing (`fssh_tully_solver.py`)

Open `fssh_tully_solver.py` and `test_md_fssh.py`:
1. Run a microcanonical (NVE) Velocity Verlet simulation of an $\text{H}_2$ Morse oscillator and verify that total energy drift is $< 10^{-6}\text{ E}_\text{h}$.
2. Simulate Tully's Model I (Single Avoided Crossing: $V_{11}(R) = A\,\text{sgn}(R)(1 - e^{-B|R|})$, $V_{22}(R) = -V_{11}(R)$, $V_{12}(R) = C e^{-D R^2}$ with $A=0.01, B=1.6, C=0.005, D=1.0$, nuclear mass $M=2000\text{ a.u.}$) across low ($k_0 = 10\text{ a.u.}$) and high ($k_0 = 25\text{ a.u.}$) initial nuclear momenta to observe how nonadiabatic transmission onto the upper state increases with nuclear velocity!
