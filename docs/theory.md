# OpenNDM theory manual

Every equation the code currently solves, with its discretisation written out
(NFR-QA-7). This covers what is implemented; see
[`status.md`](status.md) for what is not.

Notation: $G$ energy groups indexed $g$, nodes indexed $i$, surfaces indexed
$s$. A surface has a normal direction and two sides, lo and hi, with the
normal pointing from lo to hi.

---

## 1. The continuous problem

Multi-group neutron diffusion at steady state:

$$ -\nabla\cdot D_g \nabla\phi_g + \Sigma_{r,g}\,\phi_g
   = \sum_{g'\neq g} \Sigma_{s,g'\to g}\,\phi_{g'}
   + \frac{\chi_g}{k} \sum_{g'} \nu\Sigma_{f,g'}\,\phi_{g'} $$

with the removal cross section

$$ \Sigma_{r,g} = \Sigma_{a,g} + \sum_{g'\neq g} \Sigma_{s,g\to g'} $$

Within-group scattering never leaves the node and so appears on neither side.
$\Sigma_r$ is cached at library finalisation.

---

## 2. Coarse mesh finite difference

### 2.1 Node balance

Integrating over node $i$ gives, for each group,

$$ \sum_{s\in\text{faces}(i)} \pm J_s A_s + \Sigma_{r,g} V_i\,\phi_{i,g}
   = \sum_{g'\neq g} \Sigma_{s,g'\to g} V_i\,\phi_{i,g'}
   + \frac{\chi_g}{k} \sum_{g'} \nu\Sigma_{f,g'} V_i\,\phi_{i,g'} $$

with $+$ on the node's high faces and $-$ on its low faces, so the sum is the
net leakage out of the node.

### 2.2 Interface current

The coarse-mesh current across an interior surface is written

$$ J_s = -\tilde{D}_s\left(\phi_\text{hi} - \phi_\text{lo}\right)
   - \hat{D}_s\left(\phi_\text{hi} + \phi_\text{lo}\right) $$

$\tilde{D}$ is the finite difference coupling and $\hat{D}$ is the correction
the nodal kernel supplies. With $\hat{D} = 0$ this is exactly finite
differences.

### 2.3 Discontinuity factors

Let $f_\text{lo}$ and $f_\text{hi}$ be the discontinuity factors of the two
nodes on their shared face. Continuity of current and of *factor-weighted*
surface flux,

$$ f_\text{lo}\,\phi_{s,\text{lo}} = f_\text{hi}\,\phi_{s,\text{hi}}, \qquad
   J = \frac{2 D_\text{lo}\left(\phi_\text{lo} - \phi_{s,\text{lo}}\right)}{h_\text{lo}}
     = \frac{2 D_\text{hi}\left(\phi_{s,\text{hi}} - \phi_\text{hi}\right)}{h_\text{hi}} $$

eliminates the surface fluxes to give

$$ J = \Lambda\left(f_\text{lo}\,\phi_\text{lo} - f_\text{hi}\,\phi_\text{hi}\right),
   \qquad
   \Lambda = \frac{2 D_\text{lo} D_\text{hi}}
                  {f_\text{lo} h_\text{lo} D_\text{hi} + f_\text{hi} h_\text{hi} D_\text{lo}} $$

Matching this to the canonical form above gives the coupling and the initial
correction, before any nodal update:

$$ \tilde{D}_s = \frac{\Lambda\left(f_\text{lo} + f_\text{hi}\right)}{2}, \qquad
   \hat{D}_s = \frac{\Lambda\left(f_\text{hi} - f_\text{lo}\right)}{2} $$

So the discontinuity factors are folded into the coupling before the nonlinear
iteration starts, and the finite difference kernel already honours them.

### 2.4 Boundary faces

Write the boundary condition as $J_\text{out} = \gamma f \phi_s$ with the
node's outward normal, where $\gamma$ follows from the condition:

| Condition | $\gamma$ |
|---|---|
| reflective | $0$ |
| vacuum (Marshak, $J_\text{in} = 0$) | $1/2$ |
| albedo $\beta = J^-/J^+$ | $\dfrac{1 - \beta}{2(1 + \beta)}$ |
| zero flux | $\to \infty$ |

Combining with $J = 2D(\phi - \phi_s)/h$:

$$ \tilde{D}_\text{bc} = \frac{2 D \gamma f}{2 D + \gamma f h} $$

and in the zero-flux limit $\tilde{D}_\text{bc} = 2D/h$, where the
discontinuity factor drops out because the surface flux is zero either way.
The boundary contributes $+A\tilde{D}_\text{bc}$ to the node's diagonal.

$\hat{D}$ **is** updated on boundary faces, from the one-node problem of §3.6.
The same $\gamma$ serves both: the finite difference coupling above and the
boundary condition the nodal kernel imposes are the one relation, so they
cannot drift apart.

---

## 3. Transverse-integrated nodal kernels

### 3.1 The one-dimensional equation

Integrating the diffusion equation over the cross-section normal to axis $u$
and dividing by the transverse area gives, per node and group,

$$ -D\,\frac{d^2\phi}{du^2} + \Sigma_r\,\phi = Q(u) - L(u) $$

where $Q$ collects scattering and fission from the other groups and $L$ is the
transverse leakage,

$$ L(u) = \sum_{b\neq u} \frac{J_b(\text{high}) - J_b(\text{low})}{h_b} $$

evaluated from the coarse-mesh currents.

Map the node onto $\xi \in [-1/2, 1/2]$ with $u = h(\xi + 1/2)$ and use the
basis

$$ P_0 = 1, \qquad P_1 = 2\xi, \qquad P_2 = 6\xi^2 - \tfrac{1}{2} $$

all of which except $P_0$ integrate to zero over the node.

### 3.2 Transverse leakage fit

The leakage shape in node $i$ is the quadratic through the node-average
leakages of $i-1$, $i$, $i+1$:

$$ L(\xi) = L_i + l_1 P_1(\xi) + l_2 P_2(\xi) $$

with $l_1$, $l_2$ chosen so that the polynomial, extended over the neighbours,
reproduces their averages. With $a = h_{i-1}/h_i$, $b = h_{i+1}/h_i$,
$t = 1/2 + a$ and $s = 1/2 + b$:

$$ \begin{bmatrix}
     -(1+a) & (2t^3 - t/2)/a \\
     1+b    & (2s^3 - s/2)/b
   \end{bmatrix}
   \begin{bmatrix} l_1 \\ l_2 \end{bmatrix}
   =
   \begin{bmatrix} L_{i-1} - L_i \\ L_{i+1} - L_i \end{bmatrix} $$

At a core boundary one of the three averages does not exist, and what is put
in its place matters more than it looks.

**Reflective face.** The solution is mirror-symmetric about the face, so the
ghost node genuinely has $h_{i-1} = h_i$ and $L_{i-1} = L_i$. The quadratic
fit is kept and it is exact.

**Any other face.** There is no third average and nothing to invent one from,
so the quadratic drops to the linear fit through the two averages that do
exist:

$$ l_1 = \frac{L_{i+1} - L_i}{1 + h_{i+1}/h_i}, \qquad l_2 = 0 $$

with the mirrored form at a low-side face. This is what KOMODO does, and the
alternative — repeating the boundary node's own average, as though the face
were reflective when it is not — asserts a symmetry the solution does not
have. It costs more than the missing quadratic term does: on the bare cuboid
the observed order of both nodal kernels falls from 4 to 3, on the
manufactured solution the error at 32 nodes per side is 27 times larger, and
on a vacuum-bounded absorber box the nodal kernels track the finite
difference answer instead of converging away from it.

### 3.3 In-group fission

The fission source contains the group's own flux. Moving that term to the left
gives the effective removal

$$ \Sigma_{r,\text{eff}} = \Sigma_{r,g} - \frac{\chi_g\,\nu\Sigma_{f,g}}{k} $$

which can be negative in a strongly multiplying group. Both kernels handle
that; SANM switches to a trigonometric basis.

### 3.4 SANM

With $\kappa^2 = \Sigma_{r,\text{eff}} h^2 / D$ and the source projected onto
the quadratic basis as $S = (h^2/D)(Q - L) = s_0 + s_1 P_1 + s_2 P_2$, the
equation becomes

$$ -\phi'' + \kappa^2 \phi = S(\xi) $$

The particular solution is polynomial, because the basis closes under a second
derivative ($P_2'' = 12$, $P_1'' = 0$):

$$ b_2 = \frac{s_2}{\kappa^2}, \qquad
   b_1 = \frac{s_1}{\kappa^2}, \qquad
   b_0 = \frac{s_0 + 12 b_2}{\kappa^2} $$

The homogeneous solution uses the analytic pair. For $\kappa^2 > 0$ that is
$\sinh(\kappa\xi)$ and $\cosh(\kappa\xi)$; for $\kappa^2 < 0$,
$\sin(\omega\xi)$ and $\cos(\omega\xi)$ with $\omega = \sqrt{-\kappa^2}$. So

$$ \phi(\xi) = A\,\text{odd}(\xi) + C\,\text{even}(\xi)
   + b_0 + b_1 P_1 + b_2 P_2 $$

$\text{odd}$ integrates to zero over the node, so the node-average constraint
$\int\phi\,d\xi = \bar\phi$ fixes the even coefficient outright:

$$ C = \frac{\bar\phi - b_0}{\int \text{even}\,d\xi} $$

leaving exactly **one free coefficient per node**. A two-node problem
therefore has two unknowns, $A_\text{lo}$ and $A_\text{hi}$, closed by
factor-weighted flux continuity and current continuity at the interface:

$$ f_\text{lo}\,\phi_\text{lo}(+\tfrac{1}{2}) = f_\text{hi}\,\phi_\text{hi}(-\tfrac{1}{2}),
   \qquad
   -\frac{D_\text{lo}}{h_\text{lo}}\,\phi_\text{lo}'(+\tfrac{1}{2})
   = -\frac{D_\text{hi}}{h_\text{hi}}\,\phi_\text{hi}'(-\tfrac{1}{2}) $$

That is a 2×2 solve per surface per group. No outer boundary condition is
needed: the outer faces enter through the node averages, which the coarse-mesh
solution already knows.

The source $Q$ needs the *shapes* of the other groups, so the two-node problem
sweeps Gauss-Seidel over groups, projecting each group's current expansion
onto $\{P_0, P_1, P_2\}$ in closed form. The moments of the analytic functions
are, with $x = |\kappa|$, $\text{sh} = \sinh(x/2)$, $\text{ch} = \cosh(x/2)$:

$$ \begin{aligned}
   \int \cosh(x\xi)\,d\xi     &= \frac{2}{x}\,\text{sh} \\
   \int \sinh(x\xi)\,P_1\,d\xi &= \frac{2}{x}\,\text{ch} - \frac{4}{x^2}\,\text{sh} \\
   \int \cosh(x\xi)\,P_2\,d\xi &= \frac{2}{x}\,\text{sh} - \frac{12}{x^2}\,\text{ch}
                                  + \frac{24}{x^3}\,\text{sh}
   \end{aligned} $$

and on the trigonometric branch, with $\text{sn} = \sin(x/2)$,
$\text{cs} = \cos(x/2)$:

$$ \begin{aligned}
   \int \cos(x\xi)\,d\xi     &= \frac{2}{x}\,\text{sn} \\
   \int \sin(x\xi)\,P_1\,d\xi &= \frac{4}{x^2}\,\text{sn} - \frac{2}{x}\,\text{cs} \\
   \int \cos(x\xi)\,P_2\,d\xi &= \frac{2}{x}\,\text{sn} + \frac{12}{x^2}\,\text{cs}
                                 - \frac{24}{x^3}\,\text{sn}
   \end{aligned} $$

$|\kappa^2|$ is floored at $10^{-8}$ so the basis stays conditioned in the
physically irrelevant limit of vanishing removal.

### 3.5 NEM

The flux is a quartic:

$$ \phi(\xi) = \bar\phi + a_1 f_1 + a_2 f_2 + a_3 f_3 + a_4 f_4 $$

$$ f_1 = 2\xi, \qquad f_2 = 6\xi^2 - \tfrac{1}{2}, \qquad
   f_3 = \xi^3 - \tfrac{\xi}{4}, \qquad f_4 = \xi^4 - 0.3\,\xi^2 + 0.0125 $$

All four integrate to zero, and $f_3$, $f_4$ additionally vanish at both
faces, so the surface fluxes are $\phi(\pm 1/2) = \bar\phi \pm a_1 + a_2$.

The weighted residual (moment) equations, taking $f_1$ and $f_2$ as weights,
give after evaluating the integrals

$$ \begin{aligned}
   -\frac{D}{h^2}\,a_3 + \Sigma_{r,\text{eff}}\left(\frac{a_1}{3} - \frac{a_3}{60}\right)
     &= \frac{q_1}{3} \\
   -\frac{D}{h^2}\,0.4\,a_4 + \Sigma_{r,\text{eff}}\left(\frac{a_2}{5} - \frac{a_4}{350}\right)
     &= \frac{q_2}{5}
   \end{aligned} $$

which express $a_1$ and $a_2$ linearly in $a_3$ and $a_4$:

$$ \begin{aligned}
   a_1 &= \frac{q_1}{\Sigma_{r,\text{eff}}}
          + a_3\left(\frac{3D}{\Sigma_{r,\text{eff}} h^2} + \frac{1}{20}\right) \\
   a_2 &= \frac{q_2}{\Sigma_{r,\text{eff}}}
          + a_4\left(\frac{2D}{\Sigma_{r,\text{eff}} h^2} + \frac{1}{70}\right)
   \end{aligned} $$

Two free coefficients per node remain, four for a two-node problem. Unlike
SANM the node-average constraint is already built into the basis, so the two
interface conditions are not enough; the two outer faces are closed with the
coarse-mesh net currents. That is a 4×4 solve per surface per group.

The moment equations divide by $\Sigma_{r,\text{eff}}$, so it is floored at
$10^{-10}$ in magnitude rather than allowed to blow the coefficients up.

### 3.6 The one-node boundary problem

A boundary face has no second node to be continuous with, so the two-node
problem of §3.4 and §3.5 does not apply to it. The one-node problem replaces
it. The node-average constraint and the boundary condition of §2.4 between
them determine the within-node shape, and the current that shape produces at
the face is what the nonlinear update matches.

For SANM the node average fixes $C$ and the boundary condition fixes $A$, the
only coefficient left. Writing the face at $\xi = s/2$, with $s = +1$ for a
high face and $-1$ for a low one, $J_\text{out} = \gamma f \phi(s/2)$ gives

$$ A = -s\,\frac{\gamma f K_\phi + s\,(D/h)\,K_{\phi'}}
               {(D/h)\,\text{odd}'(\tfrac{1}{2}) + \gamma f\,\text{odd}(\tfrac{1}{2})} $$

where $K_\phi$ and $K_{\phi'}$ are the parts of the face flux and its
derivative that do not involve $A$. In the zero-flux limit the condition
degenerates to $\phi(s/2) = 0$ and $A = -s K_\phi / \text{odd}(\tfrac{1}{2})$,
with $D$ dropping out.

For NEM the two free coefficients $a_3, a_4$ need two equations: the boundary
condition, and the coarse-mesh net current at the node's other face. That is
the same row NEM already uses at the outer faces of a two-node problem.

Both are exactly determined, with no continuity rows and no iteration between
two nodes. A node whose other face along the same axis is also a boundary
keeps the finite difference coupling: it spans the core in that direction, and
there is no interior surface to carry information into it.

The consequence is that a problem whose exact solution lies in the kernel's
basis is now reproduced exactly rather than nearly. A one-dimensional slab,
homogeneous or reflected, is a cosine and a hyperbolic sine, both of which the
SANM basis spans; with the boundary faces closed this way and the transverse
leakage identically zero, SANM returns the analytic eigenvalue to round-off on
any mesh, five nodes included.

### 3.7 The nonlinear update

Whichever kernel is used, the two-node problem returns the interface current
$J_\text{nodal}$. The corrected coupling coefficient is whatever makes the
coarse-mesh system reproduce it:

$$ \hat{D}_s = -\frac{J_\text{nodal} + \tilde{D}_s\left(\phi_\text{hi} - \phi_\text{lo}\right)}
                     {\phi_\text{hi} + \phi_\text{lo}} $$

clamped to $|\hat{D}| \le \texttt{dhat\_limit}\cdot\tilde{D}$. Without a clamp
a large correction costs the coarse-mesh matrix its diagonal dominance; PARCS
and KOMODO clamp for the same reason.

Boundary faces are treated differently. Their $\hat{D}$ is the one-node face
current divided by a single node flux, and in an intermediate group of a
down-scatter chain at a zero-flux face that flux is small next to the source
driving it, so the ratio is badly conditioned. Two things follow:

- the step is under-relaxed by `Settings.boundary_relaxation` (default 0.5);
- an update that lands outside the band is **discarded** rather than clamped,
  keeping the value the face already had. Saturating it at the limit replaces
  an untrustworthy number with a large wrong one; discarding it on the first
  update falls back to the finite difference coupling.

Interior faces are neither damped nor discarded: their $\hat{D}$ divides by
the sum of two node fluxes and is tied to a neighbour by continuity, so it does
not have the same conditioning.

At convergence the coarse-mesh currents equal the nodal currents at every
interior surface, so the coarse-mesh solution carries the nodal accuracy while
the linear algebra stays a 7-point stencil.

---

## 4. Eigenvalue iteration

### 4.1 Power iteration with a Wielandt shift

Write the system as $A\phi = \frac{1}{k} F\phi$. The shifted form is

$$ \left(A - \frac{F}{k_s}\right)\phi^{m+1}
   = \left(\frac{1}{k^m} - \frac{1}{k_s}\right) F\phi^m $$

with $k_s = k + \text{shift}$. The eigenvalue update follows from requiring
consistency at the fixed point:

$$ \frac{1}{k^{m+1}} = \frac{1}{k_s}
   + \left(\frac{1}{k^m} - \frac{1}{k_s}\right)
     \frac{\langle F\phi^m \rangle}{\langle F\phi^{m+1} \rangle} $$

At convergence $\phi^{m+1} = \phi^m$ and this collapses to
$k^{m+1} = k^m$, so the shift changes the iteration path and nothing else.

**The shift must not be lagged.** The term $-F/k_s$ couples every group to
every other through $\chi_g\,\nu\Sigma_{f,g'}$. Where $\chi$ and
$\nu\Sigma_f$ occupy different groups, as in any two-group LWR library, that
coupling is *entirely* off the group diagonal: a single Gauss-Seidel sweep over
groups then adds the shift to one group's source and removes it again through
the eigenvalue term, leaving the flux update algebraically identical to the
unshifted one. The group sweep therefore repeats until the flux settles. On
the IAEA-2D core this is the difference between 526 and 103 outer iterations.

**The shift must be capped.** Where $\chi$ and $\nu\Sigma_f$ *do* share a
group, as in a one-group model, the shift subtracts directly from the diagonal
and a tight shift cancels removal outright. The system computes

$$ \frac{1}{k_s} \le \min_{i,g}\; 0.75\,
   \frac{\Sigma_{r,g} V_i}{\chi_g\,\nu\Sigma_{f,g} V_i} $$

and never exceeds it, which keeps the acceleration where it helps and backs
off where it would produce a singular operator.

### 4.2 Inner solves

Within group $g$, the system is a 7-point stencil on the node adjacency graph.
The right-hand side is

$$ b_g = \sum_{g'\neq g} \Sigma_{s,g'\to g}\,\phi_{g'}
   + \chi_g\left(\frac{1}{k_s}\sum_{g'\neq g} \nu\Sigma_{f,g'}\,\phi_{g'}
   + \lambda S\right), \qquad
   \lambda = \frac{1}{k} - \frac{1}{k_s} $$

with $S$ the fission source held fixed across the outer iteration. Each
group's system is solved by BiCGSTAB preconditioned by a zero-fill incomplete
LU using the same sparsity pattern. A zero pivot degrades that row to Jacobi
rather than failing, so a decoupled node surfaces as non-convergence and not as
a crash.

### 4.3 Determinism

Every inner product and norm sums the vector in fixed-size chunks of 512,
combining the chunk partials in index order. The reduction tree is therefore
identical for any thread count, and results are bit-identical on 1, 2, 4 and 8
threads (FR-OPT-4). Floating-point addition is not associative, so a naive
`omp reduction` would not have this property.

A cold solve discards the nonlinear correction along with the flux, so the
same model solved twice retraces the same iteration path rather than
depending on what the object happened to hold. Warm starting (FR-OPT-3) opts
out of that deliberately, and an adjoint run is the one case that must keep
the $\hat{D}$ a forward solve converged.

`Model.refresh()` rebuilds $\tilde{D}$ from the new diffusion coefficients and
keeps $\hat{D}$ and the flux, so a warm start survives a change to the model.
The retained $\hat{D}$ is the correction for the old model; the first nodal
update replaces it. Keeping it rather than resetting it to the
discontinuity-factor value saves outers: 18 against 22 on the shuffled case in
`benchmarks/performance`, 2 against 15 on a swap that changes nothing.

---

## 5. Adjoint

The adjoint operator is the transpose. Concretely:

- the scattering matrix is transposed, $\Sigma_{s,g\to g'}$ in place of
  $\Sigma_{s,g'\to g}$;
- the emission spectrum and the production cross section exchange roles, so
  the fission term becomes $\nu\Sigma_{f,g} \sum_{g'} \chi_{g'}\,\phi^\dagger_{g'}$;
- the two off-diagonal leakage entries of each interior surface are swapped,
  which is what transposes the $\hat{D}$ asymmetry.

Everything downstream, including the power iteration, is unchanged, so the
adjoint returns the same eigenvalue with a different flux shape. The nodal
kernels solve the *forward* transverse-integrated problem, so the nonlinear
update is not re-run for an adjoint solve: the coupling coefficients converged
by the forward solve are kept, and the transpose of the corrected forward
operator is exactly the corrected adjoint operator.

---

## 6. Fixed source

For subcritical multiplication the eigenvalue is fixed at one and the external
source $S_\text{ext}$ is added:

$$ A\phi - F\phi = S_\text{ext} V $$

assembled by taking the shifted operator at $1/k_s = 1$, which puts the
in-group fission term on the diagonal, and leaving the off-group fission on the
right-hand side. In a leakage-free box with a uniform source this reproduces
$\phi = S / (\Sigma_a - \nu\Sigma_f)$ exactly.

Because the in-group fission term sits on the diagonal here rather than in
the source, one Gauss-Seidel sweep over groups is exact unless the library
upscatters — the opposite of the shifted eigenvalue case in §4.1, where the
shift is entirely off the group diagonal and the sweep has to repeat.

The kernels see $S_\text{ext}$ through the same quadratic expansion as the
transverse leakage. For NEM only its first and second moments reach the
unknowns: a flat source cannot change a shape whose node average the
coarse-mesh solution has already fixed.

A single-node delta source is a different matter. The two-node closure
overshoots against a discontinuity that steep, and SANM can undershoot
slightly negative in the tail. That is a property of nodal methods rather than
of this implementation, and it is what `Settings.dhat_limit` exists to bound.

---

## 6a. Scattering multiplicity

OpenMC's `absorption` score counts fission and capture. It does **not** count
(n,2n) or (n,3n): those destroy one neutron and create several, and the extra
neutrons appear only in the row sums of the `nu-scatter matrix`, as an excess
over the plain `scatter matrix`.

The operator in §2 cannot see that production. Summing the group balance over
$g$, the in-scatter and out-scatter terms are the same double sum with the
indices relabelled, so they cancel identically:

$$ \sum_g \sum_{g'\neq g} \Sigma_{s,g\to g'}\,\phi_g
   = \sum_g \sum_{g'\neq g} \Sigma_{s,g'\to g}\,\phi_{g'} $$

leaving $k = \sum \nu\Sigma_f\phi \,/ \sum \Sigma_a\phi$ with the (n,xn)
neutrons nowhere in it, whichever scattering matrix was supplied.

The fix is exact rather than a correction factor. The true balance with
multiplicity is

$$ \Sigma_{a,g}\,\phi_g + \Sigma^\text{tot}_{s,g}\,\phi_g
   = \sum_{g'} \nu\Sigma_{s,g'\to g}\,\phi_{g'} + \frac{\chi_g}{k} F $$

and moving the self-scatter term across gives a removal cross section
$\Sigma_a + \Sigma_s^\text{tot} - \nu\Sigma_{s,g\to g}$. Supplying the
nu-weighted matrix to §2 instead forms
$\Sigma_a + \nu\Sigma_s^\text{tot} - \nu\Sigma_{s,g\to g}$. The two differ by
exactly the multiplicity excess $\nu\Sigma_s^\text{tot} - \Sigma_s^\text{tot}$,
so subtracting that excess from the absorption cross section reproduces the
correct operator **group by group**, not merely in the global balance.

`openndm.gc.from_mgxs_library` does this by default when the library carries
both matrices, and warns when it cannot. On a homogenised UO2 and water
mixture the effect is 230 pcm, and the eigenvalue is the only symptom: the
scattering orientation, the computed spectrum and the solver's internal
consistency all look perfect without it. See `tests/validation/README.md`.

---

## 7. Critical spectrum and buckling search

For a homogeneous medium with buckling $B^2$:

$$ \left(\Sigma_{t,g} + D_g(B^2)\,B^2\right)\phi_g
   - \sum_{g'} \Sigma_{s0,g'\to g}\,\phi_{g'}
   = \frac{\chi_g}{k} \sum_{g'} \nu\Sigma_{f,g'}\,\phi_{g'} $$

The fission source is rank one, so $k(B^2)$ is available in closed form:

$$ k(B^2) = \nu\Sigma_f^{T} M(B^2)^{-1} \chi, \qquad
   M = \operatorname{diag}\!\left(\Sigma_t + D B^2\right) - \Sigma_{s0}^{T} $$

and the search reduces to a scalar root find. The two conventions differ only
in $D$:

- **P1**: $D_g = 1 / (3\Sigma_{tr,g})$, independent of buckling.
- **B1**: $D_g = \gamma_g / (3\Sigma_{tr,g})$ with, for
  $x_g = B^2/\Sigma_{t,g}^2$,

$$ \alpha_g = \begin{cases}
     \dfrac{\arctan\sqrt{x_g}}{\sqrt{x_g}} & x_g > 0 \\[1.5ex]
     \dfrac{\operatorname{artanh}\sqrt{-x_g}}{\sqrt{-x_g}} & x_g < 0
   \end{cases}
   \qquad
   \gamma_g = \frac{x_g}{3\left(1/\alpha_g - 1\right)} $$

$\alpha \approx 1 - x/3$ for small $x$, so $\gamma \to 1$ and B1 reduces to P1
as $B \to 0$. This is verified directly in the test suite.

The root is bracketed outward from zero on whichever side reduces $k$ toward
the target: positive for a medium supercritical at infinite dilution, negative
otherwise. On the negative side the physical region is bounded — far enough
below zero the leakage term cancels removal and the flux solution changes sign
— so the search walks *inward* until the residual is defined before expanding
outward.

---

## 8. Derived quantities

Node power is $P_i = V_i \sum_g \kappa\Sigma_{f,g}\,\phi_{i,g}$, normalised to
a mean of one over the nodes that carry power. Radial and axial profiles are
volume-weighted collapses of that onto the lattice. $F_q$ is the peak node
power and $F_{\Delta H}$ the peak radial power.

Because $P_i$ is integrated over the node, its mean over nodes equals the
volume-weighted core average only when every node has the same volume. On a
non-uniform mesh $F_q$ is therefore not a peak-to-average power density: on
the NEACRP A1 core, whose axial layers run from 7.7 to 30 cm, it is 4.03
where the volume-weighted figure is 3.18.

Flux is normalised so the volume-averaged total flux over all groups is one,
which makes results comparable between meshes and between runs.

---

## 9. Delayed neutron precursors

For precursor group $d$ in node $i$,

$$ \frac{dC_{d,i}}{dt} = \beta_d F_i(t) - \lambda_d C_{d,i}, \qquad
   F_i = \sum_g \nu\Sigma_{f,g,i}\,\phi_{g,i} $$

This is linear in $C$ once $F$ is known, so it is integrated in **closed
form** across a step rather than with the scheme used for the flux. Taking
$F$ linear across the step, $F(s) = F_0 + (F_1-F_0)s/\Delta t$, the integrating
factor gives exactly

$$ C_d(\Delta t) = C_d(0)\,e^{-\lambda_d \Delta t}
   + \beta_d\left[F_0 I_0 + \frac{F_1-F_0}{\Delta t} I_1\right] $$

$$ I_0 = \int_0^{\Delta t}\! e^{-\lambda(\Delta t-s)}\,ds
      = \frac{1-e^{-\lambda \Delta t}}{\lambda}, \qquad
   I_1 = \int_0^{\Delta t}\! s\,e^{-\lambda(\Delta t-s)}\,ds
      = \frac{\Delta t - I_0}{\lambda} $$

Two consequences are worth stating because they are what the tests check.

**A linear fission source is reproduced exactly**, for any step size. The
scheme's only approximation is the shape of $F$ within the step, so its error
is entirely the error in that assumption — there is no separate time
discretisation error in the precursor equation itself.

**Equilibrium is a fixed point.** Setting $F_0 = F_1 = F$ and
$C_d(0) = \beta_d F/\lambda_d$ returns the same value, again for any step
size, since $e^{-x} + (1-e^{-x}) = 1$. A reactor at steady state therefore
stays there. A scheme that misses this starts every transient with a jump
that looks like physics.

### 9.1 Evaluating the integrals

Both closed forms are catastrophic cancellations as $\lambda \Delta t \to 0$:
$I_0$ loses digits, and $I_1$, being a difference of two nearly equal
quantities divided by a small number, loses all of them well before
$\lambda\Delta t$ underflows. Their limits are $\Delta t$ and
$\Delta t^2/2$, and below $\lambda\Delta t = 10^{-4}$ they are evaluated by
series

$$ I_0 \simeq \Delta t\left(1 - \frac{x}{2} + \frac{x^2}{6}
     - \frac{x^3}{24}\right), \qquad
   I_1 \simeq \Delta t^2\left(\frac{1}{2} - \frac{x}{3} + \frac{x^2}{8}
     - \frac{x^3}{30}\right), \qquad x = \lambda\Delta t $$

which is why a precursor group with a very long half-life and a short step
does not quietly poison the delayed source.

### 9.2 The delayed source

$$ S^{\text{delayed}}_{g,i} = \sum_d \lambda_d C_{d,i}\, \chi^d_g $$

with $\chi^d$ the delayed spectrum of group $d$ (FR-XS-3). A library without
one puts every delayed neutron in the top group, which is right for a
one-group problem and wrong for any real multi-group library.

## 10. Time integration

The flux is advanced by a $\theta$-weighted scheme (FR-KIN-2). Writing the
time derivative as $R(\phi)$ — leakage, removal, scattering, prompt fission
and the delayed source — and dividing through by $\theta$,

$$ T\left(\phi^{n+1} - \phi^n\right) = R^{n+1}
   + \frac{1-\theta}{\theta} R^n, \qquad
   T_{g,i} = \frac{V_i}{v_g\,\theta\,\Delta t} $$

so each step is a fixed-source solve on the static operator with $T$ added to
the diagonal. $\theta = 1$ is fully implicit, $\theta = 1/2$ Crank-Nicolson.

**$R^n$ is evaluated against the cross sections this step runs with, not the
ones the previous step ended with.** A perturbation applied between steps
belongs to the interval that follows it, so the operator either half of the
scheme sees is the new one; only the flux and the precursors come from the
previous step. Using the stale operator instead injects a local $O(1)$ error
at the step where the perturbation lands — one step, so $O(\Delta t)$
overall, which drags Crank-Nicolson down to first order while leaving
$\theta = 1$ untouched, because it never reads this term. Half the scheme
then looks correct, which is what makes the error worth stating here.

### 10.1 Folding the precursors into the fission spectrum

The analytic precursor solution of §9 is *linear* in the new fission source,

$$ C_d^{n+1} = \underbrace{C_d^n e^{-\lambda_d \Delta t}
   + \beta_d \tilde F^n\!\left(I_0 - \frac{I_1}{\Delta t}\right)}_{\text{known}}
   + \beta_d \tilde F^{n+1} \frac{I_1}{\Delta t} $$

so the part of the delayed source that depends on the new flux is
proportional to $\tilde F^{n+1}$ and merges into the fission term:

$$ \chi^{\text{eff}}_g = \chi^p_g (1-\beta)
   + \sum_d \lambda_d \chi^d_g \beta_d \frac{I_1}{\Delta t} $$

The step is then a single fixed-source solve with an effective emission
spectrum, rather than an iteration between the flux and the precursors.

### 10.2 Criticality normalisation

A static solve generally returns $k \neq 1$. The fission source is divided by
that eigenvalue for the whole transient, which makes the initial state
exactly critical. Without it a core at $k = 1.03$ ramps from the first step,
and the ramp looks like physics rather than like an inconsistent initial
condition. The prompt spectrum follows from the total and delayed ones,

$$ \chi^p_g = \frac{\chi_g - \sum_d \beta_d \chi^d_g}{1 - \beta} $$

so that $\chi^p(1-\beta) + \sum_d \beta_d \chi^d = \chi$ exactly and the
null transient closes.

### 10.3 What is not done

The nonlinear nodal coupling coefficients are **not** re-converged inside a
step: $\widehat{D}$ is held at the value the static solve left. For FDM there
is nothing to freeze; for the nodal kernels it is an approximation that grows
with how far the flux shape moves from the static one. Making it consistent
means carrying the time and delayed terms into the two-node problem of §3. Adaptive time stepping (FR-KIN-3) and the exponential
transformation (FR-KIN-2) are not implemented.

## 10.4 Square-root temperature feedback

Doppler broadening of a capture resonance widens it as the square root of the
fuel temperature, so the resonance-integral change follows

$$ \Sigma(T) = \Sigma_0\left[1 + \gamma\left(\sqrt{T} -
   \sqrt{T_0}\right)\right] $$

A per-Kelvin coefficient is a linearisation of this about $T_0$, and over the
hundreds of Kelvin a transient covers the two part company. The LRA BWR
specification defines its feedback in exactly the form above, so running it
as specified requires the law rather than the linearisation.

Cross sections live per composition rather than per node, so a temperature
*distribution* needs one composition per region that can hold its own
temperature.

## 11. Verification of the time integration

A leakage-free box reduces the spatial solve to exact point kinetics, which
gives closed-form answers to check against — no other code and no
transcribed deck.

| Check | Result |
|---|---|
| Null transient | Power constant to 1 part in $10^{10}$ over 2 s, both $\theta$ |
| Prompt jump | $\beta/(\beta-\rho)$ to 0.5% |
| Asymptotic period | Inhour root to 0.2% |
| Observed order, $\theta = 1$ | 1.00 |
| Observed order, $\theta = 1/2$ | 2.00 |

The order has to be measured **inside the prompt layer**. Later on the
amplitude is set by the precursor equation, which is integrated in closed
form, and that masks the order of the flux scheme entirely: both weightings
look second order there. Measuring in the wrong window is how an error of
this kind stays hidden.

### 11.1 The order an operational transient actually sees

The orders above are asymptotic, and a reactor transient is not run in the
regime where they hold. The prompt time constant is
$\Lambda / (\beta - \rho)$, of order a millisecond; an operational transient
runs at a quarter of a second, some two hundred times larger. That is the
stiff regime, and the $\theta$ method loses an order in it.

Measured on the LMW core with a smooth cross section ramp, so that nothing
but the time discretisation is in play:

| $\Delta t$ | Relative to $\Lambda/(\beta-\rho)$ | $\theta = 1/2$ | $\theta = 1$ |
|---|---|---|---|
| $5\times10^{-5}$ – $8\times10^{-4}$ s | order 1 | **2.05** | 1.03 |
| $0.0625$ – $1$ s | order $10^{2}$ | **1.0** | 1.0 |

Both weightings converge, and at a quarter of a second Crank-Nicolson is
still the more accurate of the two, but it converges first order and not
second. It does not ring: the scheme is not L-stable, so the prompt mode is
damped only weakly, yet on this problem the power history stays monotone
under refinement.

The cost is quantified in `benchmarks/README.md`: on LMW the mesh is
converged to 0.06% at the peak while the deck's own time step is about 3%
away from the extrapolated answer. This is the argument for the exponential
transformation of §10.3 — factoring the fast exponential out of the flux is
what lets a large step stay accurate, and it is the piece that would move
this table.



## 12. Water properties

IAPWS-IF97 from IAPWS R7-97(2012): region 1 (compressed liquid), region 2
(vapour) and region 4 (the saturation line). Region 1 is a fundamental
equation for the specific Gibbs free energy in dimensionless form,

$$ \gamma(\pi, \tau) = \sum_i n_i\,(7.1 - \pi)^{I_i}\,(\tau - 1.222)^{J_i},
   \qquad \pi = \frac{p}{16.53\ \text{MPa}}, \quad \tau = \frac{1386\ \text{K}}{T} $$

from which

$$ v = \frac{RT}{p}\,\pi\gamma_\pi, \qquad h = RT\,\tau\gamma_\tau, \qquad
   c_p = -R\,\tau^2\gamma_{\tau\tau}, \qquad R = 461.526\ \text{J/(kg K)} $$

Both $7.1 - \pi$ and $\tau - 1.222$ are strictly positive everywhere in region
1 — $\pi$ reaches 6.05 at 100 MPa and $\tau$ falls to 2.224 at 623.15 K — so
the negative exponents in the table need no special handling.

Region 2 splits the Gibbs energy into an ideal-gas part and a residual,

$$ \gamma = \ln\pi + \sum_i n^o_i\,\tau^{J^o_i}
   + \sum_i n_i\,\pi^{I_i}\,(\tau - 0.5)^{J_i},
   \qquad \pi = \frac{p}{1\ \text{MPa}}, \quad \tau = \frac{540\ \text{K}}{T} $$

with $v$, $h$ and $c_p$ following from the same derivatives as region 1.
Region 4 gives the saturation pressure and temperature explicitly, and the
saturated liquid and vapour states are region 1 and region 2 evaluated on that
line. Region 3 is not implemented, which bounds the saturation line at
16.529 MPa, where it meets the B23 boundary; above that the saturation
accessors refuse rather than extrapolate.

### Why the inverse is iterated rather than tabulated

The release gives a backward equation `T(p,h)` with its own 20 coefficients,
whose purpose is to avoid iteration. It is not used here. The release permits
it to differ from the basic equation by up to 25 mK, and it is a second table
that has to be kept consistent with the first.

Instead `T(p,h)` is a Newton iteration on the basic equation, whose derivative
is the specific heat the same equation already provides. Enthalpy rises
monotonically with temperature throughout region 1, so there is one root and
it is reached in a few steps. The result is exact against the equation it
inverts, which is a stronger guarantee than 25 mK, and against the backward
equation's own published test values it lands 6.5 to 16.8 mK away — inside the
allowance, as it must be.

The iteration is clipped to the saturation temperature at that pressure rather
than to region 1's flat upper bound, because the ceiling is where water stops
being liquid and that depends on pressure.

---

## References

The formulations above follow the standard nodal literature:

- Smith, K.S., *Assembly homogenization techniques for light water reactor
  analysis* — discontinuity factors and the two-node problem.
- Lawrence, R.D., *Progress in nodal methods for the solution of the neutron
  diffusion and transport equations* — transverse integration and the
  polynomial and analytic kernels.
- Downar, T. et al., PARCS theory manual — the nonlinear coupling coefficient
  and its clamping.
- Imron, M., *Development and verification of open reactor simulator ADPRES*
  (KOMODO) — the SANM formulation used as the default kernel.
- Stamm'ler, R. and Abbate, M., *Methods of Steady-State Reactor Physics in
  Nuclear Design* — the B1 correction factor.

The water properties follow:

- IAPWS R7-97(2012), *Revised Release on the IAPWS Industrial Formulation 1997
  for the Thermodynamic Properties of Water and Steam*, International
  Association for the Properties of Water and Steam, Lucerne, August 2007.
  Coefficients from Tables 2, 10, 11 and 34; verification values from
  Tables 5, 7, 15, 35 and 36.
