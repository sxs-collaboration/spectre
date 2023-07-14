// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace GeneralizedHarmonic::py_bindings {

namespace {
template <size_t Dim>
void bind_impl(py::module& m) {  // NOLINT
  m.def(
      "christoffel_second_kind",
      static_cast<tnsr::Ijj<DataVector, Dim> (*)(
          const tnsr::iaa<DataVector, Dim>&, const tnsr::II<DataVector, Dim>&)>(
          &::gh::christoffel_second_kind),
      py::arg("phi"), py::arg("inv_metric"));

  m.def("deriv_spatial_metric",
        static_cast<tnsr::ijj<DataVector, Dim> (*)(
            const tnsr::iaa<DataVector, Dim>&)>(&::gh::deriv_spatial_metric),
        py::arg("phi"));

  m.def(
      "spatial_ricci_tensor",
      static_cast<tnsr::ii<DataVector, Dim> (*)(
          const tnsr::iaa<DataVector, Dim>&, const tnsr::ijaa<DataVector, Dim>&,
          const tnsr::II<DataVector, Dim>&)>(&::gh::spatial_ricci_tensor),
      py::arg("phi"), py::arg("deriv_phi"), py::arg("inverse_spatial_metric"));

  m.def(
      "time_deriv_of_shift",
      static_cast<tnsr::I<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::II<DataVector, Dim>&, const tnsr::A<DataVector, Dim>&,
          const tnsr::iaa<DataVector, Dim>&, const tnsr::aa<DataVector, Dim>&)>(
          &::gh::time_deriv_of_shift),
      py::arg("lapse"), py::arg("shift"), py::arg("inverse_spatial_metric"),
      py::arg("spacetime_unit_normal"), py::arg("phi"), py::arg("pi"));

  m.def(
      "trace_christoffel",
      static_cast<tnsr::a<DataVector, Dim> (*)(
          const tnsr::a<DataVector, Dim>&, const tnsr::A<DataVector, Dim>&,
          const tnsr::II<DataVector, Dim>&, const tnsr::AA<DataVector, Dim>&,
          const tnsr::aa<DataVector, Dim>&, const tnsr::iaa<DataVector, Dim>&)>(
          &::gh::trace_christoffel),
      py::arg("spacetime_normal_one_form"), py::arg("spacetime_normal_vector"),
      py::arg("inverse_spatial_metric"), py::arg("inverse_spacetime_metric"),
      py::arg("pi"), py::arg("phi"));
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
