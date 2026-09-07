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

A single-node delta source is a different matter. The two-node closure
overshoots against a discontinuity that steep, and SANM can undershoot
slightly negative in the tail. That is a property of nodal methods rather than
of this implementation, and it is what `Settings.dhat_limit` exists to bound.

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
