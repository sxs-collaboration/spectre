// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Strahlkorper/Python/StrahlkorperFunctions.hpp"

#include <pybind11/pybind11.h>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"

namespace py = pybind11;

namespace ylm::py_bindings {
namespace {
template <typename Frame>
void bind_strahlkorper_functions_impl(pybind11::module& m) {  // NOLINT
  using Strahlkorper = ylm::Strahlkorper<Frame>;
  m.def("cartesian_coords",
        py::overload_cast<const Strahlkorper&>(&ylm::cartesian_coords<Frame>),
        py::arg("strahlkorper"));
  m.def("power_monitor",
        py::overload_cast<const Strahlkorper&>(&ylm::power_monitor<Frame>),
        py::arg("strahlkorper"));
}
}  // namespace

void bind_strahlkorper_functions(pybind11::module& m) {  // NOLINT
  bind_strahlkorper_functions_impl<Frame::Grid>(m);
  bind_strahlkorper_functions_impl<Frame::Inertial>(m);
}
}  // namespace ylm::py_bindings
