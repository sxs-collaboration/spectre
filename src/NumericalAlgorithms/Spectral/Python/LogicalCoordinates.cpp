// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Python/LogicalCoordinates.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace Spectral::py_bindings {
namespace {
template <size_t Dim>
void bind_logical_coordinates_impl(py::module& m) {  // NOLINT
  m.def("logical_coordinates",
        py::overload_cast<const Mesh<Dim>&>(&logical_coordinates<Dim>),
        py::arg("mesh"));
}
}  // namespace

void bind_logical_coordinates(py::module& m) {
  bind_logical_coordinates_impl<1>(m);
  bind_logical_coordinates_impl<2>(m);
  bind_logical_coordinates_impl<3>(m);
}

}  // namespace Spectral::py_bindings
