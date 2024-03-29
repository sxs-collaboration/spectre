// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/CovariantDerivOfExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/DerivSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/GaugeSource.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Phi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Pi.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfDetSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpacetimeDerivOfNormOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/SpatialDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfLowerShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfShift.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/TimeDerivativeOfSpacetimeMetric.hpp"
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

  m.def(
      "covariant_deriv_of_extrinsic_curvature",
      static_cast<tnsr::ijj<DataVector, Dim> (*)(
          const tnsr::ii<DataVector, Dim>&, const tnsr::A<DataVector, Dim>&,
          const tnsr::Ijj<DataVector, Dim>&, const tnsr::AA<DataVector, Dim>&,
          const tnsr::iaa<DataVector, Dim>&, const tnsr::iaa<DataVector, Dim>&,
          const tnsr::ijaa<DataVector, Dim>&)>(
          &::gh::covariant_deriv_of_extrinsic_curvature),
      py::arg("extrinsic_curvature"), py::arg("spacetime_unit_normal_vector"),
      py::arg("spatial_christoffel_second_kind"),
      py::arg("inverse_spacetime_metric"), py::arg("phi"), py::arg("d_pi"),
      py::arg("d_phi"));

  m.def("deriv_spatial_metric",
        static_cast<tnsr::ijj<DataVector, Dim> (*)(
            const tnsr::iaa<DataVector, Dim>&)>(&::gh::deriv_spatial_metric),
        py::arg("phi"));

  m.def("extrinsic_curvature",
        static_cast<tnsr::ii<DataVector, Dim> (*)(
            const tnsr::A<DataVector, Dim>&, const tnsr::aa<DataVector, Dim>&,
            const tnsr::iaa<DataVector, Dim>&)>(&::gh::extrinsic_curvature),
        py::arg("spacetime_normal_vector"), py::arg("pi"), py::arg("phi"));

  m.def("gauge_source",
        static_cast<tnsr::a<DataVector, Dim> (*)(
            const Scalar<DataVector>&, const Scalar<DataVector>&,
            const tnsr::i<DataVector, Dim>&, const tnsr::I<DataVector, Dim>&,
            const tnsr::I<DataVector, Dim>&, const tnsr::iJ<DataVector, Dim>&,
            const tnsr::ii<DataVector, Dim>&, const Scalar<DataVector>&,
            const tnsr::i<DataVector, Dim>&)>(&::gh::gauge_source),
        py::arg("lapse"), py::arg("dt_lapse"), py::arg("deriv_lapse"),
        py::arg("shift"), py::arg("dt_shift"), py::arg("deriv_shift"),
        py::arg("spatial_metric"), py::arg("trace_extrinsic_curvature"),
        py::arg("trace_christoffel_last_indices"));

  m.def(
      "phi",
      static_cast<tnsr::iaa<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::i<DataVector, Dim>&,
          const tnsr::I<DataVector, Dim>&, const tnsr::iJ<DataVector, Dim>&,
          const tnsr::ii<DataVector, Dim>&, const tnsr::ijj<DataVector, Dim>&)>(
          &::gh::phi),
      py::arg("lapse"), py::arg("deriv_lapse"), py::arg("shift"),
      py::arg("deriv_shift"), py::arg("spatial_metric"),
      py::arg("deriv_spatial_metric"));

  m.def("pi",
        static_cast<tnsr::aa<DataVector, Dim> (*)(
            const Scalar<DataVector>&, const Scalar<DataVector>&,
            const tnsr::I<DataVector, Dim>&, const tnsr::I<DataVector, Dim>&,
            const tnsr::ii<DataVector, Dim>&, const tnsr::ii<DataVector, Dim>&,
            const tnsr::iaa<DataVector, Dim>&)>(&::gh::pi),
        py::arg("lapse"), py::arg("dt_lapse"), py::arg("shift"),
        py::arg("dt_shift"), py::arg("spatial_metric"),
        py::arg("dt_spatial_metric"), py::arg("phi"));

  m.def(
      "spacetime_deriv_of_det_spatial_metric",
      static_cast<tnsr::a<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::II<DataVector, Dim>&,
          const tnsr::ii<DataVector, Dim>&, const tnsr::iaa<DataVector, Dim>&)>(
          &::gh::spacetime_deriv_of_det_spatial_metric),
      py::arg("sqrt_det_spatial_metric"), py::arg("inverse_spatial_metric"),
      py::arg("dt_spatial_metric"), py::arg("phi"));

  m.def(
      "spacetime_deriv_of_norm_of_shift",
      static_cast<tnsr::a<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::ii<DataVector, Dim>&, const tnsr::II<DataVector, Dim>&,
          const tnsr::AA<DataVector, Dim>&, const tnsr::A<DataVector, Dim>&,
          const tnsr::iaa<DataVector, Dim>&, const tnsr::aa<DataVector, Dim>&)>(
          &::gh::spacetime_deriv_of_norm_of_shift),
      py::arg("lapse"), py::arg("shift"), py::arg("spatial_metric"),
      py::arg("inverse_spatial_metric"), py::arg("inverse_spacetime_metric"),
      py::arg("spacetime_unit_normal"), py::arg("phi"), py::arg("pi"));

  m.def("spatial_deriv_of_lapse",
        static_cast<tnsr::i<DataVector, Dim> (*)(
            const Scalar<DataVector>&, const tnsr::A<DataVector, Dim>&,
            const tnsr::iaa<DataVector, Dim>&)>(&::gh::spatial_deriv_of_lapse),
        py::arg("lapse"), py::arg("spacetime_unit_normal"), py::arg("phi"));

  m.def(
      "time_deriv_of_lower_shift",
      static_cast<tnsr::i<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::ii<DataVector, Dim>&, const tnsr::A<DataVector, Dim>&,
          const tnsr::iaa<DataVector, Dim>&, const tnsr::aa<DataVector, Dim>&)>(
          &::gh::time_deriv_of_lower_shift),
      py::arg("lapse"), py::arg("shift"), py::arg("spatial_metric"),
      py::arg("spacetime_unit_normal"), py::arg("phi"), py::arg("pi"));

  m.def(
      "spatial_deriv_of_shift",
      static_cast<tnsr::iJ<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::AA<DataVector, Dim>&,
          const tnsr::A<DataVector, Dim>&, const tnsr::iaa<DataVector, Dim>&)>(
          &::gh::spatial_deriv_of_shift),
      py::arg("lapse"), py::arg("inverse_spacetime_metric"),
      py::arg("spacetime_unit_normal"), py::arg("phi"));

  m.def(
      "spatial_ricci_tensor",
      static_cast<tnsr::ii<DataVector, Dim> (*)(
          const tnsr::iaa<DataVector, Dim>&, const tnsr::ijaa<DataVector, Dim>&,
          const tnsr::II<DataVector, Dim>&)>(&::gh::spatial_ricci_tensor),
      py::arg("phi"), py::arg("deriv_phi"), py::arg("inverse_spatial_metric"));

  m.def("time_deriv_of_lapse",
        static_cast<Scalar<DataVector> (*)(
            const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
            const tnsr::A<DataVector, Dim>&, const tnsr::iaa<DataVector, Dim>&,
            const tnsr::aa<DataVector, Dim>&)>(&::gh::time_deriv_of_lapse),
        py::arg("lapse"), py::arg("shift"), py::arg("spacetime_unit_normal"),
        py::arg("phi"), py::arg("pi"));

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
      "time_deriv_of_spatial_metric",
      static_cast<tnsr::ii<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::iaa<DataVector, Dim>&, const tnsr::aa<DataVector, Dim>&)>(
          &::gh::time_deriv_of_spatial_metric),
      py::arg("lapse"), py::arg("shift"), py::arg("phi"), py::arg("pi"));

  m.def(
      "time_derivative_of_spacetime_metric",
      static_cast<tnsr::aa<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::aa<DataVector, Dim>&, const tnsr::iaa<DataVector, Dim>&)>(
          &::gh::time_derivative_of_spacetime_metric),
      py::arg("lapse"), py::arg("shift"), py::arg("pi"), py::arg("phi"));

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
