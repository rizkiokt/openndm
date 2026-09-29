API reference
=============

The Python API is the primary interface. The classes and functions a
calculation needs are importable from the top-level ``openndm`` namespace.
Module constants, :mod:`openndm.plots` and :func:`openndm.water.external_backend`
are imported from their modules, and so is :mod:`openndm.gc`, the
OpenMC-dependent group constant generation path.

.. toctree::
   :maxdepth: 2

   model
   geometry
   rods
   xslib
   feedback
   thermal
   channel
   pin
   water
   settings
   statepoint
   vtk
   plots
   exceptions
   gc

The five objects
----------------

A calculation is assembled from five things, in this order:

.. list-table::
   :header-rows: 1
   :widths: 20 80

   * - Object
     - What it holds
   * - :class:`openndm.XSLibrary`
     - Group constants: cross sections, discontinuity factors, branch tables.
   * - :class:`openndm.Geometry`
     - The node and surface graph, and the boundary conditions on it.
   * - :class:`openndm.Settings`
     - Kernel choice, tolerances, iteration limits, threading.
   * - :class:`openndm.Model`
     - The three above, bound together and solvable.
   * - :class:`openndm.Result`
     - Eigenvalue, fluxes, powers and currents from one solve.
