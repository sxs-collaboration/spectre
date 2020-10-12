// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/TimeDerivativeTerms.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace RelativisticEuler::Valencia {
namespace {
template <size_t Dim>
void forward_to_time_deriv(
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
        non_flux_terms_dt_tilde_s,

    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*>
        tilde_s_flux,

    // For fluxes and sources
    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& tilde_s,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& spatial_velocity,

    // For sources
    const tnsr::i<DataVector, Dim, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, Dim, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, Dim, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, Dim, Frame::Inertial>& inv_spatial_metric,
    const tnsr::ii<DataVector, Dim, Frame::Inertial>&
        extrinsic_curvature) noexcept {
  Variables<typename TimeDerivativeTerms<Dim>::temporary_tags> temp{
      get(lapse).size(), 0.0};

  TimeDerivativeTerms<Dim>::apply(
      non_flux_terms_dt_tilde_d, non_flux_terms_dt_tilde_tau,
      non_flux_terms_dt_tilde_s,

      tilde_d_flux, tilde_tau_flux, tilde_s_flux,

      // from temporaries
      make_not_null(&get<typename TimeDerivativeTerms<
                        Dim>::PressureLapseSqrtDetSpatialMetric>(temp)),
      make_not_null(
          &get<typename TimeDerivativeTerms<Dim>::TransportVelocity>(temp)),
      make_not_null(&get<typename TimeDerivativeTerms<Dim>::TildeSUp>(temp)),
      make_not_null(
          &get<typename TimeDerivativeTerms<Dim>::DensitizedStress>(temp)),

      // For fluxes
      tilde_d, tilde_tau, tilde_s, lapse, shift, sqrt_det_spatial_metric,
      pressure, spatial_velocity,

      // For sources
      d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
      extrinsic_curvature);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.TimeDerivativeTerms",
                  "[Unit][RelativisticEuler]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  pypp::check_with_random_values<1>(
      &forward_to_time_deriv<1>, "TimeDerivative",
      {"source_tilde_d", "source_tilde_tau", "source_tilde_s", "tilde_d_flux",
       "tilde_tau_flux", "tilde_s_flux"},
      {{{0.0, 1.0}}}, DataVector{5});
  pypp::check_with_random_values<1>(
      &forward_to_time_deriv<2>, "TimeDerivative",
      {"source_tilde_d", "source_tilde_tau", "source_tilde_s", "tilde_d_flux",
       "tilde_tau_flux", "tilde_s_flux"},
      {{{0.0, 1.0}}}, DataVector{5});
  pypp::check_with_random_values<1>(
      &forward_to_time_deriv<3>, "TimeDerivative",
      {"source_tilde_d", "source_tilde_tau", "source_tilde_s", "tilde_d_flux",
       "tilde_tau_flux", "tilde_s_flux"},
      {{{0.0, 1.0}}}, DataVector{5});
}
}  // namespace RelativisticEuler::Valencia
