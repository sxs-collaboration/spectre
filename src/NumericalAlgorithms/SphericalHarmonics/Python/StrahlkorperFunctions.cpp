// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/SphericalHarmonics/Python/StrahlkorperFunctions.hpp"

#include <pybind11/pybind11.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/StrahlkorperFunctions.hpp"

namespace py = pybind11;

namespace ylm::py_bindings {
void bind_strahlkorper_functions(pybind11::module& m) {  // NOLINT
  m.def("cartesian_coords",
        py::overload_cast<const ylm::Strahlkorper<Frame::Inertial>&>(
            &ylm::cartesian_coords<Frame::Inertial>),
        py::arg("strahlkorper"));
}
}  // namespace ylm::py_bindings
