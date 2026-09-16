Worked examples
===============

Four notebooks, each executed end to end with its outputs committed, so they
read without being run. They are rendered here as they were executed; the
sources live in ``examples/`` in the repository.

Start with **02** if you came here for the OpenMC coupling — it is the
workflow the package exists for. Start with **01** otherwise.

.. list-table::
   :header-rows: 1
   :widths: 30 12 58

   * - Notebook
     - Needs OpenMC
     - What it covers
   * - :doc:`01_getting_started`
     - no
     - The five objects, a complete calculation, kernel comparison, results
       and plots
   * - :doc:`02_openmc_to_openndm`
     - **yes**
     - An OpenMC lattice run to a nodal core calculation, with no format
       conversion in between
   * - :doc:`03_iaea_benchmark`
     - no
     - The IAEA-2D benchmark, mesh refinement, and what a nodal method buys you
   * - :doc:`04_branch_library_and_boron_search`
     - no
     - Branch-parameterised libraries, Doppler and boron feedback, critical
       boron search, loading pattern studies

.. toctree::
   :maxdepth: 1
   :hidden:

   01_getting_started
   02_openmc_to_openndm
   03_iaea_benchmark
   04_branch_library_and_boron_search
