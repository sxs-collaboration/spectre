// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ForceFree/TimeDerivativeTerms.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void forward_to_time_deriv(
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_e,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_psi,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_q,

    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_psi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,

    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
    const Scalar<DataVector>& tilde_q,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j_drift,
    const double kappa_psi, const double kappa_phi,
    const double parallel_conductivity,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric) {
  get(*non_flux_terms_dt_tilde_q) = 0.0;

  Variables<typename ForceFree::TimeDerivativeTerms::temporary_tags> temp{
      get(lapse).size(), 0.0};

  ForceFree::TimeDerivativeTerms::apply(
      non_flux_terms_dt_tilde_e, non_flux_terms_dt_tilde_b,
      non_flux_terms_dt_tilde_psi, non_flux_terms_dt_tilde_phi,
      non_flux_terms_dt_tilde_q,

      tilde_e_flux, tilde_b_flux, tilde_psi_flux, tilde_phi_flux, tilde_q_flux,

      make_not_null(&get<typename ForceFree::TimeDerivativeTerms::
                             LapseTimesElectricFieldOneForm>(temp)),
      make_not_null(&get<typename ForceFree::TimeDerivativeTerms::
                             LapseTimesMagneticFieldOneForm>(temp)),
      make_not_null(
          &get<typename ForceFree::TimeDerivativeTerms::TildeJDrift>(temp)),

      make_not_null(
          &get<gr::Tags::SpatialChristoffelFirstKind<DataVector, 3>>(temp)),
      make_not_null(
          &get<gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>>(temp)),
      make_not_null(
          &get<gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, 3>>(
              temp)),

      make_not_null(&get<gr::Tags::Lapse<DataVector>>(temp)),
      make_not_null(&get<gr::Tags::Shift<DataVector, 3>>(temp)),
      make_not_null(&get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(temp)),

      tilde_e, tilde_b, tilde_psi, tilde_phi, tilde_q, tilde_j_drift, kappa_psi,
      kappa_phi, parallel_conductivity,

      lapse, shift, sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric,
      extrinsic_curvature, d_lapse, d_shift, d_spatial_metric);

  CHECK(get<gr::Tags::Lapse<DataVector>>(temp) == lapse);
  CHECK(get<gr::Tags::Shift<DataVector, 3>>(temp) == shift);
  CHECK(get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(temp) ==
        inv_spatial_metric);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ForceFree.TimeDerivativeTerms",
                  "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ForceFree"};

  pypp::check_with_random_values<1>(
      &forward_to_time_deriv, "TimeDerivative",
      {"source_tilde_e", "source_tilde_b", "source_tilde_psi",
       "source_tilde_phi", "source_tilde_q", "tilde_e_flux", "tilde_b_flux",
       "tilde_psi_flux", "tilde_phi_flux", "tilde_q_flux"},
      {{{0.0, 1.0}}}, DataVector{5});
}
