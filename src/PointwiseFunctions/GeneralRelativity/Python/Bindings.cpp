// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <pybind11/pybind11.h>
#include <type_traits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativeSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/InverseSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/ProjectionOperators.hpp"
#include "PointwiseFunctions/GeneralRelativity/Psi4Real.hpp"
#include "PointwiseFunctions/GeneralRelativity/Ricci.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/TimeDerivativeOfSpatialMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylElectric.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylMagnetic.hpp"
#include "PointwiseFunctions/GeneralRelativity/WeylPropagating.hpp"
#include "Utilities/ErrorHandling/SegfaultHandler.hpp"

namespace py = pybind11;

namespace GeneralRelativity::py_bindings {

namespace {
template <size_t Dim, IndexType Index>
void bind_spacetime_impl(py::module& m) {  // NOLINT
  m.def("christoffel_first_kind",
        static_cast<tnsr::abb<DataVector, Dim, Frame::Inertial, Index> (*)(
            const tnsr::abb<DataVector, Dim, Frame::Inertial, Index>&)>(
            &::gr::christoffel_first_kind),
        py::arg("d_metric"));

  m.def("christoffel_second_kind",
        static_cast<tnsr::Abb<DataVector, Dim, Frame::Inertial, Index> (*)(
            const tnsr::abb<DataVector, Dim, Frame::Inertial, Index>&,
            const tnsr::AA<DataVector, Dim, Frame::Inertial, Index>&)>(
            &::gr::christoffel_second_kind),
        py::arg("d_metric"), py::arg("inverse_metric"));

  m.def("ricci_scalar",
        static_cast<Scalar<DataVector> (*)(
            const tnsr::aa<DataVector, Dim, Frame::Inertial, Index>&,
            const tnsr::AA<DataVector, Dim, Frame::Inertial, Index>&)>(
            &::gr::ricci_scalar),
        py::arg("ricci_tensor"), py::arg("inverse_metric"));

  m.def("ricci_tensor",
        static_cast<tnsr::aa<DataVector, Dim, Frame::Inertial, Index> (*)(
            const tnsr::Abb<DataVector, Dim, Frame::Inertial, Index>&,
            const tnsr::aBcc<DataVector, Dim, Frame::Inertial, Index>&)>(
            &::gr::ricci_tensor),
        py::arg("christoffel_2nd_kind"), py::arg("d_christoffel_2nd_kind"));
}

template <size_t Dim>
void bind_impl(py::module& m) {  // NOLINT
  m.def(
      "deriv_inverse_spatial_metric",
      static_cast<tnsr::iJJ<DataVector, Dim> (*)(
          const tnsr::II<DataVector, Dim>&, const tnsr::ijj<DataVector, Dim>&)>(
          &::gr::deriv_inverse_spatial_metric),
      py::arg("inverse_spatial_metric"), py::arg("d_spatial_metric"));

  m.def(
      "extrinsic_curvature",
      py::overload_cast<
          const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::iJ<DataVector, Dim>&, const tnsr::ii<DataVector, Dim>&,
          const tnsr::ii<DataVector, Dim>&, const tnsr::ijj<DataVector, Dim>&>(
          &gr::extrinsic_curvature<DataVector, Dim, Frame::Inertial>),
      py::arg("lapse"), py::arg("shift"), py::arg("deriv_shift"),
      py::arg("spatial_metric"), py::arg("dt_spatial_metric"),
      py::arg("deriv_spatial_metric"));

  m.def("inverse_spacetime_metric",
        static_cast<tnsr::AA<DataVector, Dim> (*)(
            const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
            const tnsr::II<DataVector, Dim>&)>(&::gr::inverse_spacetime_metric),
        py::arg("lapse"), py::arg("shift"), py::arg("inverse_spatial_metric"));

  m.def("lapse",
        static_cast<Scalar<DataVector> (*)(const tnsr::I<DataVector, Dim>&,
                                           const tnsr::aa<DataVector, Dim>&)>(
            &::gr::lapse),
        py::arg("shift"), py::arg("spacetime_metric"));

  m.def("derivatives_of_spacetime_metric",
        static_cast<tnsr::abb<DataVector, Dim> (*)(
            const Scalar<DataVector>&, const Scalar<DataVector>&,
            const tnsr::i<DataVector, Dim>&, const tnsr::I<DataVector, Dim>&,
            const tnsr::I<DataVector, Dim>&, const tnsr::iJ<DataVector, Dim>&,
            const tnsr::ii<DataVector, Dim>&, const tnsr::ii<DataVector, Dim>&,
            const tnsr::ijj<DataVector, Dim>&)>(
            &::gr::derivatives_of_spacetime_metric),
        py::arg("lapse"), py::arg("dt_lapse"), py::arg("deriv_lapse"),
        py::arg("shift"), py::arg("dt_shift"), py::arg("deriv_shift"),
        py::arg("spatial_metric"), py::arg("dt_spatial_metric"),
        py::arg("deriv_spatial_metric"));

  m.def("spacetime_metric",
        static_cast<tnsr::aa<DataVector, Dim> (*)(
            const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
            const tnsr::ii<DataVector, Dim>&)>(&::gr::spacetime_metric),
        py::arg("lapse"), py::arg("shift"), py::arg("spatial_metric"));

  m.def("spatial_metric",
        static_cast<tnsr::ii<DataVector, Dim> (*)(
            const tnsr::aa<DataVector, Dim>&)>(&::gr::spatial_metric),
        py::arg("spacetime_metric"));

  m.def(
      "time_derivative_of_spacetime_metric",
      static_cast<tnsr::aa<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const Scalar<DataVector>&,
          const tnsr::I<DataVector, Dim>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::ii<DataVector, Dim>&, const tnsr::ii<DataVector, Dim>&)>(
          &::gr::time_derivative_of_spacetime_metric),
      py::arg("lapse"), py::arg("dt_lapse"), py::arg("shift"),
      py::arg("dt_shift"), py::arg("spatial_metric"),
      py::arg("dt_spatial_metric"));

  m.def(
      "time_derivative_of_spatial_metric",
      static_cast<tnsr::ii<DataVector, Dim> (*)(
          const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&,
          const tnsr::iJ<DataVector, Dim>&, const tnsr::ii<DataVector, Dim>&,
          const tnsr::ijj<DataVector, Dim>&, const tnsr::ii<DataVector, Dim>&)>(
          &::gr::time_derivative_of_spatial_metric),
      py::arg("lapse"), py::arg("shift"), py::arg("deriv_shift"),
      py::arg("spatial_metric"), py::arg("deriv_spatial_metric"),
      py::arg("extrinsic_curvature"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::II<DataVector, Dim> (*)(
            const tnsr::II<DataVector, Dim>&, const tnsr::I<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("inverse_spatial_metric"), py::arg("normal_vector"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::ii<DataVector, Dim> (*)(
            const tnsr::ii<DataVector, Dim>&, const tnsr::i<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("spatial_metric"), py::arg("normal_one_form"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::Ij<DataVector, Dim> (*)(
            const tnsr::I<DataVector, Dim>&, const tnsr::i<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("normal_vector"), py::arg("normal_one_form"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::aa<DataVector, Dim> (*)(
            const tnsr::aa<DataVector, Dim>&, const tnsr::a<DataVector, Dim>&,
            const tnsr::i<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("spacetime_metric"), py::arg("spacetime_normal_one_form"),
        py::arg("interface_unit_normal_one_form"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::AA<DataVector, Dim> (*)(
            const tnsr::AA<DataVector, Dim>&, const tnsr::A<DataVector, Dim>&,
            const tnsr::I<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("inverse_spacetime_metric"), py::arg("spacetime_normal_vector"),
        py::arg("interface_unit_normal_vector"));

  m.def("transverse_projection_operator",
        static_cast<tnsr::Ab<DataVector, Dim> (*)(
            const tnsr::A<DataVector, Dim>&, const tnsr::a<DataVector, Dim>&,
            const tnsr::I<DataVector, Dim>&, const tnsr::i<DataVector, Dim>&)>(
            &::gr::transverse_projection_operator),
        py::arg("spacetime_normal_vector"),
        py::arg("spacetime_normal_one_form"),
        py::arg("interface_unit_normal_vector"),
        py::arg("interface_unit_normal_one_form"));

  m.def(
      "shift",
      static_cast<tnsr::I<DataVector, Dim> (*)(
          const tnsr::aa<DataVector, Dim>&, const tnsr::II<DataVector, Dim>&)>(
          &::gr::shift),
      py::arg("spacetime_metric"), py::arg("inverse_spatial_metric"));

  m.def("spacetime_normal_one_form",
        static_cast<tnsr::a<DataVector, Dim> (*)(const Scalar<DataVector>&)>(
            &::gr::spacetime_normal_one_form),
        py::arg("lapse"));

  m.def("spacetime_normal_vector",
        static_cast<tnsr::A<DataVector, Dim> (*)(
            const Scalar<DataVector>&, const tnsr::I<DataVector, Dim>&)>(
            &::gr::spacetime_normal_vector),
        py::arg("lapse"), py::arg("shift"));

  m.def("weyl_electric",
        static_cast<tnsr::ii<DataVector, Dim> (*)(
            const tnsr::ii<DataVector, Dim>&, const tnsr::ii<DataVector, Dim>&,
            const tnsr::II<DataVector, Dim>&)>(&::gr::weyl_electric),
        py::arg("spatial_ricci"), py::arg("extrinsic_curvature"),
        py::arg("inverse_spatial_metric"));

  m.def("weyl_electric_scalar",
        static_cast<Scalar<DataVector> (*)(const tnsr::ii<DataVector, Dim>&,
                                           const tnsr::II<DataVector, Dim>&)>(
            &::gr::weyl_electric_scalar),
        py::arg("weyl_electric"), py::arg("inverse_spatial_metric"));

  m.def("weyl_magnetic",
        static_cast<tnsr::ii<DataVector, 3, Frame::Inertial> (*)(
            const tnsr::ijj<DataVector, 3, Frame::Inertial>&,
            const tnsr::ii<DataVector, 3, Frame::Inertial>&,
            const Scalar<DataVector>&)>(&gr::weyl_magnetic),
        py::arg("grad_extrinsic_curvature"), py::arg("spatial_metric"),
        py::arg("sqrt_det_spatial_metric"));

  m.def("weyl_magnetic_scalar",
        static_cast<Scalar<DataVector> (*)(
            const tnsr::ii<DataVector, 3, Frame::Inertial>&,
            const tnsr::II<DataVector, 3, Frame::Inertial>&)>(
            &gr::weyl_magnetic_scalar),
        py::arg("weyl_magnetic"), py::arg("inverse_spatial_metric"));

  m.def("weyl_propagating",
        py::overload_cast<
            const tnsr::ii<DataVector, Dim>&, const tnsr::ii<DataVector, Dim>&,
            const tnsr::II<DataVector, Dim>&, const tnsr::ijj<DataVector, Dim>&,
            const tnsr::I<DataVector, Dim>&, const tnsr::II<DataVector, Dim>&,
            const tnsr::ii<DataVector, Dim>&, const tnsr::Ij<DataVector, Dim>&,
            const double>(
            &gr::weyl_propagating<DataVector, Dim, Frame::Inertial>),
        py::arg("ricci"), py::arg("extrinsic_curvature"),
        py::arg("inverse_spatial_metric"),
        py::arg("cov_deriv_extrinsic_curvature"),
        py::arg("unit_interface_normal_vector"), py::arg("projection_IJ"),
        py::arg("projection_ij"), py::arg("projection_Ij"), py::arg("sign"));
}
}  // namespace

PYBIND11_MODULE(_Pybindings, m) {  // NOLINT
  enable_segfault_handler();
  py::module_::import("spectre.DataStructures");
  py::module_::import("spectre.DataStructures.Tensor");
  py::module_::import("spectre.Spectral");
  py_bindings::bind_spacetime_impl<1, IndexType::Spatial>(m);
  py_bindings::bind_spacetime_impl<2, IndexType::Spatial>(m);
  py_bindings::bind_spacetime_impl<3, IndexType::Spatial>(m);
  py_bindings::bind_spacetime_impl<1, IndexType::Spacetime>(m);
  py_bindings::bind_spacetime_impl<2, IndexType::Spacetime>(m);
  py_bindings::bind_spacetime_impl<3, IndexType::Spacetime>(m);
  py_bindings::bind_impl<1>(m);
  py_bindings::bind_impl<2>(m);
  py_bindings::bind_impl<3>(m);
  m.def("psi4real",
        static_cast<Scalar<DataVector> (*)(
            const tnsr::ii<DataVector, 3>&, const tnsr::ii<DataVector, 3>&,
            const tnsr::ijj<DataVector, 3>&, const tnsr::ii<DataVector, 3>&,
            const tnsr::II<DataVector, 3>&, const tnsr::I<DataVector, 3>&)>(
            &::gr::psi_4_real),
        py::arg("spatial_ricci"), py::arg("extrinsic_curvature"),
        py::arg("cov_deriv_extrinsic_curvature"), py::arg("spatial_metric"),
        py::arg("inverse_spatial_metric"), py::arg("inertial_coords"));
}
}  // namespace GeneralRelativity::py_bindings
