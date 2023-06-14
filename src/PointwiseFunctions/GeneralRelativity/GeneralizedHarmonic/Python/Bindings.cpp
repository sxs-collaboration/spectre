// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace GeneralizedHarmonic::py_bindings {

namespace {
template <size_t Dim>
void bind_impl(py::module& m) {  // NOLINT
  m.def(
      "time_deriv_of_shift",
      static_cast<tnsr::I<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::II<DataVector, Dim>&, const tnsr::A<DataVector, Dim>&,
          const tnsr::iaa<DataVector, Dim>&, const tnsr::aa<DataVector, Dim>&)>(
          &::gh::time_deriv_of_shift),
      py::arg("lapse"), py::arg("shift"), py::arg("inverse_spatial_metric"),
      py::arg("spacetime_unit_normal"), py::arg("phi"), py::arg("pi"));
}
}  // namespace

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py_bindings::bind_impl<1>(m);
  py_bindings::bind_impl<2>(m);
  py_bindings::bind_impl<3>(m);
}
}  // namespace GeneralizedHarmonic::py_bindings
