// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Python/PowerMonitors.hpp"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/LinearOperators/PowerMonitors.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace PowerMonitors::py_bindings {

namespace {
template <size_t Dim>
void bind_power_monitors_impl(py::module& m) {  // NOLINT
  m.def("power_monitors",
        py::overload_cast<const DataVector&, const Mesh<Dim>&> (
          &power_monitors<Dim>),
        py::arg("data_vector"), py::arg("mesh"));
  m.def(
      "truncation_error",
      py::overload_cast<const std::array<DataVector, Dim>&, const DataVector&>(
          &truncation_error<Dim>),
      py::arg("power_monitors_of_u"), py::arg("data_vector"));
}
}  // namespace

void bind_power_monitors(py::module& m) {
  bind_power_monitors_impl<1>(m);
  bind_power_monitors_impl<2>(m);
  bind_power_monitors_impl<3>(m);
  m.def("relative_truncation_error",
        py::overload_cast<const DataVector&, const size_t>(
            &relative_truncation_error),
        py::arg("power_monitor"), py::arg("num_modes_to_use"));
}

}  // namespace PowerMonitors::py_bindings
