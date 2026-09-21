# OpenNDM theory manual

Every equation the code currently solves, with its discretisation written out
(NFR-QA-7). This covers what is implemented; see
[`status.md`](status.md) for what is not.

Notation: `G` energy groups indexed `g`, nodes indexed `i`, surfaces indexed
`s`. A surface has a normal direction and two sides, `lo` and `hi`, with the
normal pointing from `lo` to `hi`.

---

## 1. The continuous problem

Multi-group neutron diffusion at steady state:

```
-∇·D_g ∇φ_g + Σ_r,g φ_g = Σ_{g'≠g} Σ_s,g'→g φ_g' + (χ_g / k) Σ_g' νΣ_f,g' φ_g'
```

with the removal cross section

```
Σ_r,g = Σ_a,g + Σ_{g'≠g} Σ_s,g→g'
```

Within-group scattering never leaves the node and so appears on neither side.
`Σ_r` is cached at library finalisation.

---

## 2. Coarse mesh finite difference

### 2.1 Node balance

Integrating over node `i` gives, for each group,

```
Σ_s∈faces(i)  ±J_s A_s  +  Σ_r,g V_i φ_{i,g}
    = Σ_{g'≠g} Σ_s,g'→g V_i φ_{i,g'} + (χ_g / k) Σ_g' νΣ_f,g' V_i φ_{i,g'}
```

with `+` on the node's high faces and `−` on its low faces, so the sum is the
net leakage out of the node.

### 2.2 Interface current

The coarse-mesh current across an interior surface is written

```
J_s = −D̃_s (φ_hi − φ_lo) − D̂_s (φ_hi + φ_lo)
```

`D̃` is the finite difference coupling and `D̂` is the correction the nodal
kernel supplies. With `D̂ = 0` this is exactly finite differences.

### 2.3 Discontinuity factors

Let `f_lo` and `f_hi` be the discontinuity factors of the two nodes on their
shared face. Continuity of current and of *factor-weighted* surface flux,

```
f_lo φ_s,lo = f_hi φ_s,hi
J = 2 D_lo (φ_lo − φ_s,lo) / h_lo = 2 D_hi (φ_s,hi − φ_hi) / h_hi
```

eliminates the surface fluxes to give

```
J = Λ (f_lo φ_lo − f_hi φ_hi),    Λ = 2 D_lo D_hi / (f_lo h_lo D_hi + f_hi h_hi D_lo)
```

Matching this to the canonical form above:

```
D̃_s = Λ (f_lo + f_hi) / 2
D̂_s = Λ (f_hi − f_lo) / 2      (the initial value, before any nodal update)
```

So the discontinuity factors are folded into the coupling before the nonlinear
iteration starts, and the finite difference kernel already honours them.

### 2.4 Boundary faces

Write the boundary condition as `J_out = γ f φ_s` with the node's outward
normal, where `γ` follows from the condition:

| Condition | γ |
|---|---|
| reflective | `0` |
| vacuum (Marshak, `J_in = 0`) | `1/2` |
| albedo `β = J⁻/J⁺` | `(1 − β) / (2(1 + β))` |
| zero flux | `→ ∞` |

Combining with `J = 2D(φ − φ_s)/h`:

```
D̃_bc = 2 D γ f / (2 D + γ f h)
```

and in the zero-flux limit `D̃_bc = 2D/h`, where the discontinuity factor drops
out because the surface flux is zero either way. The boundary contributes
`+ A D̃_bc` to the node's diagonal.

`D̂` is **not** updated on boundary faces; see the limitation in
[`status.md`](status.md).

---

## 3. Transverse-integrated nodal kernels

### 3.1 The one-dimensional equation

Integrating the diffusion equation over the cross-section normal to axis `u`
and dividing by the transverse area gives, per node and group,

```
−D d²φ/du² + Σ_r φ = Q(u) − L(u)
```

where `Q` collects scattering and fission from the other groups and `L` is the
transverse leakage,

```
L(u) = Σ_{b≠u} [ J_b(high) − J_b(low) ] / h_b
```

evaluated from the coarse-mesh currents.

Map the node onto `ξ ∈ [−1/2, 1/2]` with `u = h(ξ + 1/2)` and use the basis

```
P_0 = 1,    P_1 = 2ξ,    P_2 = 6ξ² − 1/2
```

all of which except `P_0` integrate to zero over the node.

### 3.2 Transverse leakage fit

The leakage shape in node `i` is the quadratic through the node-average
leakages of `i−1`, `i`, `i+1`:

```
L(ξ) = L_i + l_1 P_1(ξ) + l_2 P_2(ξ)
```

with `l_1`, `l_2` chosen so that the polynomial, extended over the neighbours,
reproduces their averages. With `a = h_{i−1}/h_i`, `b = h_{i+1}/h_i`,
`t = 1/2 + a` and `s = 1/2 + b`:

```
[ −(1+a)        (2t³ − t/2)/a ] [ l_1 ]   [ L_{i−1} − L_i ]
[  (1+b)        (2s³ − s/2)/b ] [ l_2 ] = [ L_{i+1} − L_i ]
```

At a core boundary the missing neighbour is replaced by the node itself, which
is the usual flat extrapolation.

### 3.3 In-group fission

The fission source contains the group's own flux. Moving that term to the left
gives the effective removal

```
Σ_r,eff = Σ_r,g − χ_g νΣ_f,g / k
```

which can be negative in a strongly multiplying group. Both kernels handle
that; SANM switches to a trigonometric basis.

### 3.4 SANM

With `κ² = Σ_r,eff h² / D` and the source projected onto the quadratic basis as
`S = (h²/D)(Q − L) = s_0 + s_1 P_1 + s_2 P_2`, the equation becomes

```
−φ'' + κ² φ = S(ξ)
```

The particular solution is polynomial, because the basis closes under a second
derivative (`P_2'' = 12`, `P_1'' = 0`):

```
b_2 = s_2 / κ²
b_1 = s_1 / κ²
b_0 = (s_0 + 12 b_2) / κ²
```

The homogeneous solution uses the analytic pair. For `κ² > 0` that is
`sinh(κξ)` and `cosh(κξ)`; for `κ² < 0`, `sin(ωξ)` and `cos(ωξ)` with
`ω = √(−κ²)`. So

```
φ(ξ) = A·odd(ξ) + C·even(ξ) + b_0 + b_1 P_1 + b_2 P_2
```

`odd` integrates to zero over the node, so the node-average constraint
`∫φ dξ = φ̄` fixes the even coefficient outright:

```
C = (φ̄ − b_0) / ∫even dξ
```

leaving exactly **one free coefficient per node**. A two-node problem
therefore has two unknowns, `A_lo` and `A_hi`, closed by the two interface
conditions:

```
f_lo φ_lo(+1/2) = f_hi φ_hi(−1/2)          (factor-weighted flux continuity)
−(D_lo/h_lo) φ_lo'(+1/2) = −(D_hi/h_hi) φ_hi'(−1/2)     (current continuity)
```

That is a 2×2 solve per surface per group. No outer boundary condition is
needed: the outer faces enter through the node averages, which the coarse-mesh
solution already knows.

The source `Q` needs the *shapes* of the other groups, so the two-node problem
sweeps Gauss-Seidel over groups, projecting each group's current expansion
onto `{P_0, P_1, P_2}` in closed form. The moments of the analytic functions
are, with `x = |κ|`, `sh = sinh(x/2)`, `ch = cosh(x/2)`:

```
∫ cosh(xξ) dξ        = (2/x) sh
∫ sinh(xξ) P_1 dξ    = (2/x) ch − (4/x²) sh
∫ cosh(xξ) P_2 dξ    = (2/x) sh − (12/x²) ch + (24/x³) sh
```

and on the trigonometric branch, with `sn = sin(x/2)`, `cs = cos(x/2)`:

```
∫ cos(xξ) dξ         = (2/x) sn
∫ sin(xξ) P_1 dξ     = (4/x²) sn − (2/x) cs
∫ cos(xξ) P_2 dξ     = (2/x) sn + (12/x²) cs − (24/x³) sn
```

`|κ²|` is floored at `1e-8` so the basis stays conditioned in the physically
irrelevant limit of vanishing removal.

### 3.5 NEM

The flux is a quartic:

```
φ(ξ) = φ̄ + a_1 f_1 + a_2 f_2 + a_3 f_3 + a_4 f_4
f_1 = 2ξ                 f_2 = 6ξ² − 1/2
f_3 = ξ³ − ξ/4           f_4 = ξ⁴ − 0.3ξ² + 0.0125
```

All four integrate to zero, and `f_3`, `f_4` additionally vanish at both faces,
so the surface fluxes are `φ(±1/2) = φ̄ ± a_1 + a_2`.

The weighted residual (moment) equations, taking `f_1` and `f_2` as weights,
give after evaluating the integrals

```
−(D/h²) a_3            + Σ_r,eff (a_1/3 − a_3/60)   = q_1/3
−(D/h²)(0.4 a_4)       + Σ_r,eff (a_2/5 − a_4/350)  = q_2/5
```

which express `a_1` and `a_2` linearly in `a_3` and `a_4`:

```
a_1 = q_1/Σ_r,eff + a_3 (3D/(Σ_r,eff h²) + 1/20)
a_2 = q_2/Σ_r,eff + a_4 (2D/(Σ_r,eff h²) + 1/70)
```

Two free coefficients per node remain, four for a two-node problem. Unlike
SANM the node-average constraint is already built into the basis, so the two
interface conditions are not enough; the two outer faces are closed with the
coarse-mesh net currents. That is a 4×4 solve per surface per group.

The moment equations divide by `Σ_r,eff`, so it is floored at `1e-10` in
magnitude rather than allowed to blow the coefficients up.

### 3.6 The nonlinear update

Whichever kernel is used, the two-node problem returns the interface current
`J_nodal`. The corrected coupling coefficient is whatever makes the
coarse-mesh system reproduce it:

```
D̂_s = −( J_nodal + D̃_s (φ_hi − φ_lo) ) / (φ_hi + φ_lo)
```

clamped to `|D̂| ≤ dhat_limit · D̃`. Without a clamp a large correction costs
the coarse-mesh matrix its diagonal dominance; PARCS and KOMODO clamp for the
same reason.

At convergence the coarse-mesh currents equal the nodal currents at every
interior surface, so the coarse-mesh solution carries the nodal accuracy while
the linear algebra stays a 7-point stencil.

---

## 4. Eigenvalue iteration

### 4.1 Power iteration with a Wielandt shift

Write the system as `A φ = (1/k) F φ`. The shifted form is

```
(A − F/k_s) φ^{m+1} = (1/k^m − 1/k_s) F φ^m
```

with `k_s = k + shift`. The eigenvalue update follows from requiring
consistency at the fixed point:

```
1/k^{m+1} = 1/k_s + (1/k^m − 1/k_s) · ⟨F φ^m⟩ / ⟨F φ^{m+1}⟩
```

At convergence `φ^{m+1} = φ^m` and this collapses to `k^{m+1} = k^m`, so the
shift changes the iteration path and nothing else.

**The shift must not be lagged.** The term `−F/k_s` couples every group to
every other through `χ_g νΣ_f,g'`. Where `χ` and `νΣ_f` occupy different
groups, as in any two-group LWR library, that coupling is *entirely* off the
group diagonal: a single Gauss-Seidel sweep over groups then adds the shift to
one group's source and removes it again through the eigenvalue term, leaving
the flux update algebraically identical to the unshifted one. The group sweep
therefore repeats until the flux settles. On the IAEA-2D core this is the
difference between 526 and 103 outer iterations.

**The shift must be capped.** Where `χ` and `νΣ_f` *do* share a group, as in a
one-group model, the shift subtracts directly from the diagonal and a tight
shift cancels removal outright. The system computes

```
1/k_s ≤ min over nodes and groups of  0.75 · Σ_r V / (χ νΣ_f V)
```

and never exceeds it, which keeps the acceleration where it helps and backs
off where it would produce a singular operator.

### 4.2 Inner solves

Within group `g`, the system is a 7-point stencil on the node adjacency graph.
The right-hand side is

```
b_g = Σ_{g'≠g} Σ_s,g'→g φ_g'  +  χ_g ( (1/k_s) Σ_{g'≠g} νΣ_f,g' φ_g' + λ S )
λ = 1/k − 1/k_s
```

with `S` the fission source held fixed across the outer iteration. Each group's
system is solved by BiCGSTAB preconditioned by a zero-fill incomplete LU using
the same sparsity pattern. A zero pivot degrades that row to Jacobi rather than
failing, so a decoupled node surfaces as non-convergence and not as a crash.

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
the `D̂` a forward solve converged.

---

## 5. Adjoint

The adjoint operator is the transpose. Concretely:

- the scattering matrix is transposed, `Σ_s,g→g'` in place of `Σ_s,g'→g`;
- the emission spectrum and the production cross section exchange roles, so
  the fission term becomes `νΣ_f,g Σ_g' χ_g' φ†_g'`;
- the two off-diagonal leakage entries of each interior surface are swapped,
  which is what transposes the `D̂` asymmetry.

Everything downstream, including the power iteration, is unchanged, so the
adjoint returns the same eigenvalue with a different flux shape. The nodal
kernels solve the *forward* transverse-integrated problem, so the nonlinear
update is not re-run for an adjoint solve: the coupling coefficients converged
by the forward solve are kept, and the transpose of the corrected forward
operator is exactly the corrected adjoint operator.

---

## 6. Fixed source

For subcritical multiplication the eigenvalue is fixed at one and the external
source `S_ext` is added:

```
A φ − F φ = S_ext V
```

assembled by taking the shifted operator at `1/k_s = 1`, which puts the
in-group fission term on the diagonal, and leaving the off-group fission on the
right-hand side. In a leakage-free box with a uniform source this reproduces
`φ = S / (Σ_a − νΣ_f)` exactly.

Because the in-group fission term sits on the diagonal here rather than in
the source, one Gauss-Seidel sweep over groups is exact unless the library
upscatters -- the opposite of the shifted eigenvalue case in §4.1, where the
shift is entirely off the group diagonal and the sweep has to repeat.

The kernels see `S_ext` through the same quadratic expansion as the transverse
leakage. For NEM only its first and second moments reach the unknowns: a flat
source cannot change a shape whose node average the coarse-mesh solution has
already fixed.

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
`g`, the in-scatter and out-scatter terms are the same double sum with the
indices relabelled, so they cancel identically:

```
sum_g sum_{g'!=g} Sigma_s,g->g' phi_g  ==  sum_g sum_{g'!=g} Sigma_s,g'->g phi_g'
```

leaving `k = sum(nuSf phi) / sum(Sigma_a phi)` with the (n,xn) neutrons
nowhere in it, whichever scattering matrix was supplied.

The fix is exact rather than a correction factor. The true balance with
multiplicity is

```
Sigma_a,g phi_g + Sigma_s^tot,g phi_g
    = sum_{g'} nuSigma_s,g'->g phi_g' + chi_g/k F
```

and moving the self-scatter term across gives a removal cross section
`Sigma_a + Sigma_s^tot - nuSigma_{s,g->g}`. Supplying the nu-weighted matrix
to §2 instead forms `Sigma_a + nuSigma_s^tot - nuSigma_{s,g->g}`. The two
differ by exactly the multiplicity excess `nuSigma_s^tot - Sigma_s^tot`, so
subtracting that excess from the absorption cross section reproduces the
correct operator **group by group**, not merely in the global balance.

`openndm.gc.from_mgxs_library` does this by default when the library carries
both matrices, and warns when it cannot. On a homogenised UO2 and water
mixture the effect is 230 pcm, and the eigenvalue is the only symptom: the
scattering orientation, the computed spectrum and the solver's internal
consistency all look perfect without it. See `tests/validation/README.md`.

---

## 7. Critical spectrum and buckling search

For a homogeneous medium with buckling `B²`:

```
( Σ_t,g + D_g(B²) B² ) φ_g − Σ_g' Σ_s0,g'→g φ_g' = (χ_g/k) Σ_g' νΣ_f,g' φ_g'
```

The fission source is rank one, so `k(B²)` is available in closed form:

```
k(B²) = νΣ_f^T M(B²)^{-1} χ,      M = diag(Σ_t + D B²) − Σ_s0^T
```

and the search reduces to a scalar root find. The two conventions differ only
in `D`:

- **P1**: `D_g = 1 / (3 Σ_tr,g)`, independent of buckling.
- **B1**: `D_g = γ_g / (3 Σ_tr,g)` with, for `x_g = B²/Σ_t,g²`,

```
α_g = arctan(√x_g) / √x_g        (x > 0)
α_g = artanh(√−x_g) / √−x_g      (x < 0)
γ_g = x_g / (3 (1/α_g − 1))
```

`α ≈ 1 − x/3` for small `x`, so `γ → 1` and B1 reduces to P1 as `B → 0`. This
is verified directly in the test suite.

The root is bracketed outward from zero on whichever side reduces `k` toward
the target: positive for a medium supercritical at infinite dilution, negative
otherwise. On the negative side the physical region is bounded — far enough
below zero the leakage term cancels removal and the flux solution changes sign
— so the search walks *inward* until the residual is defined before expanding
outward.

---

## 8. Derived quantities

Node power is `P_i = V_i Σ_g κΣ_f,g φ_{i,g}`, normalised to a mean of one over
the nodes that carry power. Radial and axial profiles are volume-weighted
collapses of that onto the lattice. `F_q` is the peak node power and `F_ΔH` the
peak radial power, both relative to the core average.

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
