OpenNDM
=======

**Open Nodal Diffusion Method** — a three-dimensional, multi-group nodal
diffusion solver for reactor core analysis, with a C++17 core and a
group-constant generation path built natively on the OpenMC stack.

OpenNDM provides the second stage of a two-step lattice-to-core calculation:
OpenMC runs the lattice physics, OpenNDM runs the core. There is no format
conversion between them.

.. code-block:: python

   import numpy as np
   import openndm

   lib = openndm.XSLibrary(n_groups=2, n_compositions=1)
   lib.set_composition(0, D=[1.5, 0.4], absorption=[0.01, 0.085],
                       nu_fission=[0.0, 0.135], kappa_fission=[0.0, 0.135],
                       chi=[1.0, 0.0], scatter=[[0.0, 0.02], [0.0, 0.0]])
   lib.finalize()

   geom = openndm.Geometry.from_lattice(np.zeros((10, 9, 9), int), pitch=20.0)
   result = openndm.Model(geom, lib).solve()
   print(result.k_eff)

.. warning::

   OpenNDM is pre-1.0 research software. It carries no nuclear quality
   assurance pedigree and is not qualified for licensing or safety analysis.
   See :doc:`status` for what is and is not implemented.

Installation
------------

.. code-block:: bash

   conda install -c conda-forge openndm   # recommended
   pip install openndm

The OpenMC coupling in :mod:`openndm.gc` needs OpenMC, which is only
distributed through conda-forge. The solver itself imports and runs without it.

.. toctree::
   :maxdepth: 2
   :caption: Using OpenNDM

   user-guide
   examples/index
   theory

.. toctree::
   :maxdepth: 2
   :caption: API reference

   api/index

.. toctree::
   :maxdepth: 2
   :caption: Project

   status
   requirements
   architecture
   contributing
   releasing
   changelog

Indices
-------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
