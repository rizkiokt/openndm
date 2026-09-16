Group constants from OpenMC
===========================

.. module:: openndm.gc

Everything needed to turn an OpenMC lattice calculation into a nodal library,
with no user-written format conversion. This is the only part of OpenNDM that
requires OpenMC; importing :mod:`openndm` never pulls it in.

.. code-block:: python

   import openndm.gc as gc

   lib = gc.from_statepoint("statepoint.100.h5", mgxs_lib)

.. note::

   OpenMC is mocked when these pages are built, so signatures and docstrings
   are rendered but OpenMC types in annotations do not resolve to the OpenMC
   documentation.

Ingestion
---------

.. automodule:: openndm.gc.mgxs
   :members:
   :undoc-members:
   :show-inheritance:

Discontinuity factors
---------------------

.. automodule:: openndm.gc.adf
   :members:
   :undoc-members:
   :show-inheritance:

Critical spectrum and leakage
-----------------------------

.. automodule:: openndm.gc.leakage
   :members:
   :undoc-members:
   :show-inheritance:

Kinetics parameters
-------------------

.. automodule:: openndm.gc.kinetics
   :members:
   :undoc-members:
   :show-inheritance:

Pin form functions
------------------

.. automodule:: openndm.gc.formfunc
   :members:
   :undoc-members:
   :show-inheritance:

Branch libraries
----------------

.. automodule:: openndm.gc.driver
   :members:
   :undoc-members:
   :show-inheritance:
