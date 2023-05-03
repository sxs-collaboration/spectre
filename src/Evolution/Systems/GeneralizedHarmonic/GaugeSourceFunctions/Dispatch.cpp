// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/AnalyticChristoffel.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/DampedHarmonic.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Harmonic.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace gh::gauges {
template <size_t Dim>
void dispatch(
    const gsl::not_null<tnsr::a<DataVector, Dim, Frame::Inertial>*> gauge_h,
    const gsl::not_null<tnsr::ab<DataVector, Dim, Frame::Inertial>*> d4_gauge_h,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const tnsr::a<DataVector, Dim, Frame::Inertial>&
        spacetime_unit_normal_one_form,
    const tnsr::A<DataVector, Dim, Frame::Inertial>& spacetime_unit_normal,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inverse_spatial_metric,
    const tnsr::abb<DataVector, Dim, Frame::Inertial>& d4_spacetime_metric,
    const Scalar<DataVector>& half_pi_two_normals,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& half_phi_two_normals,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& spacetime_metric,
    const tnsr::aa<DataVector, Dim, Frame::Inertial>& pi,
    const tnsr::iaa<DataVector, Dim, Frame::Inertial>& phi,
    const Mesh<Dim>& mesh, double time,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords,
    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>& inverse_jacobian,
    const GaugeCondition& gauge_condition) {
  if (const auto* harmonic_gauge =
          dynamic_cast<const Harmonic*>(&gauge_condition);
      harmonic_gauge != nullptr) {
    harmonic_gauge->gauge_and_spacetime_derivative(gauge_h, d4_gauge_h, time,
                                                   inertial_coords);
  } else if (const auto* damped_harmonic_gauge =
                 dynamic_cast<const DampedHarmonic*>(&gauge_condition);
             damped_harmonic_gauge != nullptr) {
    damped_harmonic_gauge->gauge_and_spacetime_derivative(
        gauge_h, d4_gauge_h, lapse, shift, spacetime_unit_normal_one_form,
        spacetime_unit_normal, sqrt_det_spatial_metric, inverse_spatial_metric,
        d4_spacetime_metric, half_pi_two_normals, half_phi_two_normals,
        spacetime_metric, pi, phi, time, inertial_coords);
  } else if (const auto* analytic_gauge =
                 dynamic_cast<const AnalyticChristoffel*>(&gauge_condition);
             analytic_gauge != nullptr) {
    analytic_gauge->gauge_and_spacetime_derivative(
        gauge_h, d4_gauge_h, mesh, time, inertial_coords, inverse_jacobian);
  } else {
    // LCOV_EXCL_START
    ERROR(
        "Failed to dispatch to Harmonic, DampedHarmonic, or Analytic gauge "
        "condition.");
    // LCOV_EXCL_STOP
  }
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                   \
  template void dispatch(                                                      \
      gsl::not_null<tnsr::a<DataVector, DIM(data), Frame::Inertial>*> gauge_h, \
      gsl::not_null<tnsr::ab<DataVector, DIM(data), Frame::Inertial>*>         \
          d4_gauge_h,                                                          \
      const Scalar<DataVector>& lapse,                                         \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& shift,            \
      const tnsr::a<DataVector, DIM(data), Frame::Inertial>&                   \
          spacetime_unit_normal_one_form,                                      \
      const tnsr::A<DataVector, DIM(data), Frame::Inertial>&                   \
          spacetime_unit_normal,                                               \
      const Scalar<DataVector>& sqrt_det_spatial_metric,                       \
      const tnsr::II<DataVector, DIM(data), Frame::Inertial>&                  \
          inverse_spatial_metric,                                              \
      const tnsr::abb<DataVector, DIM(data), Frame::Inertial>&                 \
          d4_spacetime_metric,                                                 \
      const Scalar<DataVector>& half_pi_two_normals,                           \
      const tnsr::i<DataVector, DIM(data), Frame::Inertial>&                   \
          half_phi_two_normals,                                                \
      const tnsr::aa<DataVector, DIM(data), Frame::Inertial>&                  \
          spacetime_metric,                                                    \
      const tnsr::aa<DataVector, DIM(data), Frame::Inertial>& pi,              \
      const tnsr::iaa<DataVector, DIM(data), Frame::Inertial>& phi,            \
      const Mesh<DIM(data)>& mesh, double time,                                \
      const tnsr::I<DataVector, DIM(data), Frame::Inertial>& inertial_coords,  \
      const InverseJacobian<DataVector, DIM(data), Frame::ElementLogical,      \
                            Frame::Inertial>& inverse_jacobian,                \
      const GaugeCondition& gauge_condition);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace gh::gauges
