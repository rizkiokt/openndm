//! \file bindings.cpp
//! pybind11 bridge between the C++ core and the openndm Python package.
//!
//! OpenMC reaches its C++ core through a hand-written C API called by ctypes.
//! OpenNDM uses pybind11 instead (see docs/architecture.md): it removes the
//! hand-written marshalling layer and gives zero-copy numpy views onto solver
//! results for free, which is what FR-OPT-5 and DP-4 need. A thin extern "C"
//! layer remains available for non-Python coupling.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "openndm/cmfd.h"
#include "openndm/error.h"
#include "openndm/geometry.h"
#include "openndm/kernel.h"
#include "openndm/settings.h"
#include "openndm/solver.h"
#include "openndm/xslib.h"

namespace py = pybind11;
using namespace openndm;

namespace {

//! Simpler zero-copy view keyed on a base object handle.
py::array_t<double> array_view(const std::vector<double>& data, py::object base,
    std::vector<py::ssize_t> shape)
{
  return py::array_t<double>(std::move(shape), data.data(), std::move(base));
}

py::array_t<int> int_array(const std::vector<int>& data)
{
  return py::array_t<int>({static_cast<py::ssize_t>(data.size())}, data.data());
}

std::vector<double> to_vector(
    const py::array_t<double, py::array::c_style | py::array::forcecast>& a)
{
  return std::vector<double>(a.data(), a.data() + a.size());
}

}  // namespace

PYBIND11_MODULE(_core, m)
{
  m.doc() = "OpenNDM C++ core (pybind11 bindings)";
#ifdef OPENNDM_VERSION
  m.attr("__version__") = OPENNDM_VERSION;
#else
  m.attr("__version__") = "0.0.0";
#endif

  // ------------------------------------------------------------- exceptions
  //
  // Every C++ error surfaces as the matching class from openndm.exceptions
  // rather than as a separate extension-local type, so that
  // `except openndm.InputError` catches what the core raises and users can
  // subclass the hierarchy. The lookup is lazy because the package is still
  // importing this module when it loads.
  py::register_exception_translator([](std::exception_ptr p) {
    static const auto raise_as = [](const char* name, const py::object& exc) {
      py::object module = py::module_::import("openndm.exceptions");
      py::object cls = module.attr(name);
      PyErr_SetObject(cls.ptr(), exc.ptr());
    };
    try {
      if (p) std::rethrow_exception(p);
    } catch (const ConvergenceError& e) {
      py::object module = py::module_::import("openndm.exceptions");
      py::object cls = module.attr("ConvergenceError");
      py::object exc = cls(e.what(), py::arg("iterations") = e.iterations(),
          py::arg("residual") = e.residual());
      PyErr_SetObject(cls.ptr(), exc.ptr());
    } catch (const InputError& e) {
      py::object module = py::module_::import("openndm.exceptions");
      raise_as("InputError", module.attr("InputError")(e.what()));
    } catch (const LibraryError& e) {
      py::object module = py::module_::import("openndm.exceptions");
      raise_as("LibraryError", module.attr("LibraryError")(e.what()));
    } catch (const NotImplementedError& e) {
      py::object module = py::module_::import("openndm.exceptions");
      raise_as("NotImplementedError_",
          module.attr("NotImplementedError_")(e.what()));
    } catch (const Error& e) {
      py::object module = py::module_::import("openndm.exceptions");
      raise_as("OpenNDMError", module.attr("OpenNDMError")(e.what()));
    }
  });

  // ------------------------------------------------------------- enumerations
  py::enum_<BoundaryType>(m, "BoundaryType")
      .value("interior", BoundaryType::interior)
      .value("zero_flux", BoundaryType::zero_flux)
      .value("vacuum", BoundaryType::vacuum)
      .value("reflective", BoundaryType::reflective)
      .value("albedo", BoundaryType::albedo);

  py::enum_<KernelType>(m, "KernelType")
      .value("fdm", KernelType::fdm)
      .value("nem", KernelType::nem)
      .value("sanm", KernelType::sanm);

  py::enum_<SolveMode>(m, "SolveMode")
      .value("forward", SolveMode::forward)
      .value("adjoint", SolveMode::adjoint)
      .value("fixed_source", SolveMode::fixed_source);

  py::enum_<Extrapolation>(m, "Extrapolation")
      .value("clamp", Extrapolation::clamp)
      .value("linear", Extrapolation::linear)
      .value("error", Extrapolation::error);

  // ---------------------------------------------------------------- geometry
  py::class_<CartesianSpec>(m, "CartesianSpec")
      .def(py::init<>())
      .def_readwrite("dx", &CartesianSpec::dx)
      .def_readwrite("dy", &CartesianSpec::dy)
      .def_readwrite("dz", &CartesianSpec::dz)
      .def_readwrite("composition", &CartesianSpec::composition)
      .def_readwrite("bc", &CartesianSpec::bc)
      .def_readwrite("albedo", &CartesianSpec::albedo)
      .def_readwrite("inactive_bc", &CartesianSpec::inactive_bc)
      .def_readwrite("inactive_albedo", &CartesianSpec::inactive_albedo);

  py::class_<Node>(m, "Node")
      .def_readonly("volume", &Node::volume)
      .def_readonly("composition", &Node::composition)
      .def_property_readonly("width",
          [](const Node& n) {
            return std::vector<double>(n.width.begin(), n.width.end());
          })
      .def_property_readonly("ijk", [](const Node& n) {
        return std::vector<int>(n.ijk.begin(), n.ijk.end());
      });

  py::class_<Surface>(m, "Surface")
      .def_readonly("lo", &Surface::lo)
      .def_readonly("hi", &Surface::hi)
      .def_readonly("area", &Surface::area)
      .def_readonly("axis", &Surface::axis)
      .def_readonly("bc", &Surface::bc)
      .def("is_boundary", &Surface::is_boundary);

  py::class_<Geometry>(m, "Geometry")
      .def_static("from_cartesian", &Geometry::from_cartesian, py::arg("spec"))
      .def_property_readonly("n_nodes", &Geometry::n_nodes)
      .def_property_readonly("n_surfaces", &Geometry::n_surfaces)
      .def_property_readonly("n_axes", &Geometry::n_axes)
      .def_property_readonly("n_compositions", &Geometry::n_compositions)
      .def_property_readonly("total_volume", &Geometry::total_volume)
      .def_property_readonly("nodes", &Geometry::nodes)
      .def_property_readonly("surfaces", &Geometry::surfaces)
      .def_property_readonly("lattice_shape",
          [](const Geometry& g) {
            const auto& s = g.lattice_shape();
            return py::make_tuple(s[0], s[1], s[2]);
          })
      .def_property_readonly("lattice_to_node",
          [](const Geometry& g) { return int_array(g.lattice_to_node()); })
      .def_property_readonly("volumes",
          [](const Geometry& g) {
            std::vector<double> v;
            v.reserve(static_cast<std::size_t>(g.n_nodes()));
            for (const auto& n : g.nodes()) v.push_back(n.volume);
            return py::array_t<double>(
                {static_cast<py::ssize_t>(v.size())}, v.data());
          })
      .def_property_readonly("compositions",
          [](const Geometry& g) {
            std::vector<int> c;
            c.reserve(static_cast<std::size_t>(g.n_nodes()));
            for (const auto& n : g.nodes()) c.push_back(n.composition);
            return int_array(c);
          })
      .def("set_composition", &Geometry::set_composition, py::arg("node"),
          py::arg("composition"));

  // -------------------------------------------------------------- xs library
  py::class_<Composition>(m, "Composition")
      .def_readwrite("D", &Composition::D)
      .def_readwrite("absorption", &Composition::absorption)
      .def_readwrite("nu_fission", &Composition::nu_fission)
      .def_readwrite("kappa_fission", &Composition::kappa_fission)
      .def_readwrite("chi", &Composition::chi)
      .def_readwrite("scatter", &Composition::scatter)
      .def_readwrite("inv_velocity", &Composition::inv_velocity)
      .def_readonly("removal", &Composition::removal)
      .def_readwrite("D_std", &Composition::D_std)
      .def_readwrite("absorption_std", &Composition::absorption_std)
      .def_readwrite("nu_fission_std", &Composition::nu_fission_std)
      .def_readwrite("scatter_std", &Composition::scatter_std)
      .def_property_readonly("has_uncertainty", &Composition::has_uncertainty);

  py::class_<BranchAxis>(m, "BranchAxis")
      .def(py::init<>())
      .def_readwrite("name", &BranchAxis::name)
      .def_readwrite("points", &BranchAxis::points);

  py::class_<DelayedData>(m, "DelayedData")
      .def(py::init<>())
      .def_readwrite("beta", &DelayedData::beta)
      .def_readwrite("lambda_", &DelayedData::lambda)
      .def_readwrite("chi_delayed", &DelayedData::chi_delayed)
      .def_property_readonly("n_precursors", &DelayedData::n_precursors)
      .def_property_readonly("beta_total", &DelayedData::beta_total);

  py::class_<XSLibrary>(m, "XSLibrary")
      .def(py::init<int, int>(), py::arg("n_groups"), py::arg("n_compositions"))
      .def_property_readonly("n_groups", &XSLibrary::n_groups)
      .def_property_readonly("n_compositions", &XSLibrary::n_compositions)
      .def_property_readonly("n_states", &XSLibrary::n_states)
      .def_property_readonly("finalized", &XSLibrary::finalized)
      .def_property_readonly("has_uncertainty", &XSLibrary::has_uncertainty)
      .def_property_readonly("axes", &XSLibrary::axes)
      .def("set_axes", &XSLibrary::set_axes, py::arg("axes"))
      .def_property("extrapolation", &XSLibrary::extrapolation,
          &XSLibrary::set_extrapolation)
      // Reading returns a snapshot, so inspecting a library can never
      // invalidate it; mutation goes through the explicitly named accessor.
      .def("composition",
          py::overload_cast<int, int>(&XSLibrary::composition, py::const_),
          py::arg("composition"), py::arg("state") = 0,
          py::return_value_policy::copy)
      .def("mutable_composition",
          py::overload_cast<int, int>(&XSLibrary::composition),
          py::arg("composition"), py::arg("state") = 0,
          py::return_value_policy::reference_internal)
      .def_property_readonly("delayed",
          py::overload_cast<>(&XSLibrary::delayed),
          py::return_value_policy::reference_internal)
      .def(
          "set_adf",
          [](XSLibrary& lib, int comp,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  values,
              int n_axes) {
            AdfSet& set = lib.adf(comp, n_axes);
            const auto v = to_vector(values);
            if (v.size() != set.value.size()) {
              throw InputError(
                  "ADF array must have 2 * n_axes * n_groups entries");
            }
            set.value = v;
          },
          py::arg("composition"), py::arg("values"), py::arg("n_axes") = 3)
      .def("adf_value", &XSLibrary::adf_value, py::arg("composition"),
          py::arg("face"), py::arg("group"))
      .def("finalize",
          [](XSLibrary& lib) {
            std::vector<std::string> warnings;
            lib.finalize(&warnings);
            return warnings;
          })
      .def("interpolate", &XSLibrary::interpolate, py::arg("state"));

  // ---------------------------------------------------------------- settings
  py::class_<Settings>(m, "Settings")
      .def(py::init<>())
      .def_readwrite("kernel", &Settings::kernel)
      .def_readwrite("mode", &Settings::mode)
      .def_readwrite("k_tolerance", &Settings::k_tolerance)
      .def_readwrite(
          "fission_source_tolerance", &Settings::fission_source_tolerance)
      .def_readwrite("max_outer", &Settings::max_outer)
      .def_readwrite("min_outer", &Settings::min_outer)
      .def_readwrite("inner_tolerance", &Settings::inner_tolerance)
      .def_readwrite("max_inner", &Settings::max_inner)
      .def_readwrite("group_sweeps", &Settings::group_sweeps)
      .def_readwrite("group_sweep_tolerance", &Settings::group_sweep_tolerance)
      .def_readwrite("wielandt_shift", &Settings::wielandt_shift)
      .def_readwrite("wielandt_start", &Settings::wielandt_start)
      .def_readwrite("nodal_update_interval", &Settings::nodal_update_interval)
      .def_readwrite("nodal_start", &Settings::nodal_start)
      .def_readwrite("two_node_sweeps", &Settings::two_node_sweeps)
      .def_readwrite("dhat_limit", &Settings::dhat_limit)
      .def_readwrite("warm_start", &Settings::warm_start)
      .def_readwrite("verbosity", &Settings::verbosity)
      .def_readwrite("threads", &Settings::threads);

  py::class_<IterationRecord>(m, "IterationRecord")
      .def_readonly("outer", &IterationRecord::outer)
      .def_readonly("k_eff", &IterationRecord::k_eff)
      .def_readonly("k_change", &IterationRecord::k_change)
      .def_readonly("source_change", &IterationRecord::source_change)
      .def_readonly("inner_iterations", &IterationRecord::inner_iterations);

  // ------------------------------------------------------------------ result
  py::class_<Result>(m, "Result")
      .def_readonly("k_eff", &Result::k_eff)
      .def_readonly("converged", &Result::converged)
      .def_readonly("outer_iterations", &Result::outer_iterations)
      .def_readonly("runtime_seconds", &Result::runtime_seconds)
      .def_readonly("kernel", &Result::kernel)
      .def_readonly("history", &Result::history)
      .def_property_readonly("flux",
          [](py::object self) {
            const Result& r = self.cast<const Result&>();
            const py::ssize_t n =
                r.power.empty() ? 0 : static_cast<py::ssize_t>(r.power.size());
            const py::ssize_t g =
                n ? static_cast<py::ssize_t>(r.flux.size()) / n : 0;
            return array_view(r.flux, self, {n, g});
          })
      .def_property_readonly("power", [](py::object self) {
        const Result& r = self.cast<const Result&>();
        return array_view(
            r.power, self, {static_cast<py::ssize_t>(r.power.size())});
      });

  // ------------------------------------------------------------------ solver
  py::class_<Solver>(m, "Solver")
      .def(py::init<const Geometry&, const XSLibrary&>(), py::arg("geometry"),
          py::arg("library"), py::keep_alive<1, 2>(), py::keep_alive<1, 3>())
      // The GIL is released for the whole solve so that several models can be
      // driven from Python threads (FR-OPT-2).
      .def("solve", &Solver::solve, py::arg("settings"),
          py::call_guard<py::gil_scoped_release>())
      .def(
          "solve_fixed_source",
          [](Solver& s,
              py::array_t<double, py::array::c_style | py::array::forcecast>
                  source,
              const Settings& settings) {
            const auto v = to_vector(source);
            py::gil_scoped_release release;
            return s.solve_fixed_source(v, settings);
          },
          py::arg("source"), py::arg("settings"))
      .def("reset", &Solver::reset)
      .def_property_readonly("k_eff", &Solver::k_eff)
      .def_property_readonly("flux", [](py::object self) {
        const Solver& s = self.cast<const Solver&>();
        return array_view(
            s.flux(), self, {static_cast<py::ssize_t>(s.flux().size())});
      });
}
