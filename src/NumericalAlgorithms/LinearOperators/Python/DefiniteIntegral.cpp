// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/Python/DefiniteIntegral.hpp"

#include <cstddef>
#include <pybind11/pybind11.h>
#include <string>

#include "DataStructures/DataVector.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"

namespace py = pybind11;

namespace py_bindings {

namespace {
template <size_t Dim>
void bind_definite_integral_impl(py::module& m) {  // NOLINT
  m.def("definite_integral", &definite_integral<Dim>, py::arg("integrand"),
        py::arg("mesh"));
}
}  // namespace

void bind_definite_integral(py::module& m) {  // NOLINT
  bind_definite_integral_impl<1>(m);
  bind_definite_integral_impl<2>(m);
  bind_definite_integral_impl<3>(m);
}

}  // namespace py_bindings
