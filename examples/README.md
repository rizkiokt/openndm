# Worked examples

Four notebooks, each executed end to end with its outputs committed, so you
can read them without running anything.

| Notebook | Needs OpenMC | What it covers |
|---|---|---|
| [`01_getting_started.ipynb`](01_getting_started.ipynb) | no | The five objects, a complete calculation, kernel comparison, results and plots |
| [`02_openmc_to_openndm.ipynb`](02_openmc_to_openndm.ipynb) | **yes** | An OpenMC lattice run to a nodal core calculation, with no format conversion in between |
| [`03_iaea_benchmark.ipynb`](03_iaea_benchmark.ipynb) | no | The IAEA-2D benchmark, mesh refinement, and what a nodal method buys you |
| [`04_branch_library_and_boron_search.ipynb`](04_branch_library_and_boron_search.ipynb) | no | Branch-parameterised libraries, Doppler and boron feedback, critical boron search, loading pattern studies |

Start with **02** if you came here for the OpenMC coupling; it is the workflow
the package exists for. Start with **01** otherwise.

## Running them

```bash
pip install '.[dev,plot]'
jupyter lab examples/
```

Notebook 2 additionally needs OpenMC, the `openmc` executable on `PATH`, and a
nuclear data library:

```bash
export OPENMC_CROSS_SECTIONS=/path/to/cross_sections.xml
```

To re-execute them all in place:

```bash
jupyter nbconvert --to notebook --execute --inplace examples/*.ipynb
```

## What they produce

Notebook 2 ends with the comparison the package is judged on. On the run
committed here, a 5x5 UO2 pin lattice in ENDF/B-VIII.1:

```
OpenMC  k_inf = 1.42158 +/- 73 pcm
OpenNDM k_eff = 1.42146
difference    = -12 pcm  (0.2 sigma)
```

Notebook 3 shows why a nodal kernel is worth the extra work. On the IAEA-2D
core at one node per assembly, a 20 cm mesh:

```
FDM  1.033324  (+373.4 pcm)
NEM  1.028412  (-117.8 pcm)
SANM 1.029553  (  -3.7 pcm)
```

Notebook 4 finds the critical boron concentration in five solves and then
trades 3400 pcm of reactivity for a drop in F_dH from 2.96 to 1.86 across four
assembly shuffles.

## See also

- [`../docs/user-guide.md`](../docs/user-guide.md) — the reference for
  everything the notebooks use
- [`../benchmarks/`](../benchmarks/) — runnable benchmark decks
- [`../tests/validation/`](../tests/validation/) — the OpenMC coupling
  validation cases
