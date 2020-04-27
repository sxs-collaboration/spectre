// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TimeDerivativeTerms.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void forward_to_time_deriv(
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_phi,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_tau_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    const gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        tilde_phi_flux,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi, const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const Scalar<DataVector>& sqrt_det_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
    const Scalar<DataVector>& pressure,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& extrinsic_curvature,
    const double constraint_damping_parameter) noexcept {
  get(*non_flux_terms_dt_tilde_d) = 0.0;

  Variables<
      typename grmhd::ValenciaDivClean::TimeDerivativeTerms::temporary_tags>
      temp{get(lapse).size(), 0.0};

  grmhd::ValenciaDivClean::TimeDerivativeTerms::apply(
      non_flux_terms_dt_tilde_d, non_flux_terms_dt_tilde_tau,
      non_flux_terms_dt_tilde_s, non_flux_terms_dt_tilde_b,
      non_flux_terms_dt_tilde_phi,

      tilde_d_flux, tilde_tau_flux, tilde_s_flux, tilde_b_flux, tilde_phi_flux,

      make_not_null(
          &get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3,
                                                   Frame::Inertial>>(temp)),
      make_not_null(
          &get<hydro::Tags::MagneticFieldOneForm<DataVector, 3,
                                                 Frame::Inertial>>(temp)),
      make_not_null(
          &get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(temp)),
      make_not_null(&get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp)),
      make_not_null(&get<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::
                             OneOverLorentzFactorSquared>(temp)),
      make_not_null(&get<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::
                             PressureStar>(temp)),
      make_not_null(&get<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::
                             PressureStarLapseSqrtDetSpatialMetric>(temp)),
      make_not_null(&get<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::
                             TransportVelocity>(temp)),
      make_not_null(&get<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::
                             LapseTimesbOverW>(temp)),

      // Source terms
      make_not_null(
          &get<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::TildeSUp>(
              temp)),
      make_not_null(&get<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::
                             DensitizedStress>(temp)),
      make_not_null(
          &get<gr::Tags::SpatialChristoffelFirstKind<3, Frame::Inertial,
                                                     DataVector>>(temp)),
      make_not_null(
          &get<gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial,
                                                      DataVector>>(temp)),
      make_not_null(
          &get<gr::Tags::TraceSpatialChristoffelSecondKind<3, Frame::Inertial,
                                                           DataVector>>(temp)),
      make_not_null(&get<typename grmhd::ValenciaDivClean::TimeDerivativeTerms::
                             EnthalpyTimesDensityWSquaredPlusBSquared>(temp)),

      tilde_d, tilde_tau, tilde_s, tilde_b, tilde_phi, lapse, shift,
      sqrt_det_spatial_metric, spatial_metric, inv_spatial_metric, d_lapse,
      d_shift, d_spatial_metric, pressure, spatial_velocity, lorentz_factor,
      magnetic_field,

      rest_mass_density, specific_enthalpy, extrinsic_curvature,
      constraint_damping_parameter);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.TimeDerivativeTerms",
                  "[Unit][GrMhd]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  pypp::check_with_random_values<1>(
      &forward_to_time_deriv, "TimeDerivative",
      {"source_tilde_d", "source_tilde_tau", "source_tilde_s", "source_tilde_b",
       "source_tilde_phi", "tilde_d_flux", "tilde_tau_flux", "tilde_s_flux",
       "tilde_b_flux", "tilde_phi_flux"},
      {{{0.0, 1.0}}}, DataVector{5});
}
