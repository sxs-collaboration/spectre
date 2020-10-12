// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/TimeDerivativeTerms.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"

namespace {
void forward_to_time_deriv(
    const gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_e,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        non_flux_terms_dt_tilde_s,

    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    const gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,

    const Scalar<DataVector>& tilde_e,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::II<DataVector, 3, Frame::Inertial>& tilde_p,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,

    const Scalar<DataVector>& source_n,
    const tnsr::i<DataVector, 3, Frame::Inertial>& source_i,
    const tnsr::i<DataVector, 3>& d_lapse,
    const tnsr::iJ<DataVector, 3>& d_shift,
    const tnsr::ijj<DataVector, 3>& d_spatial_metric,
    const tnsr::ii<DataVector, 3>& extrinsic_curvature) {
  Variables<tmpl::list<typename RadiationTransport::M1Grey::TimeDerivativeTerms<
      neutrinos::ElectronNeutrinos<1>>::TildeSUp>>
      temp(get(lapse).size());

  RadiationTransport::M1Grey::
      TimeDerivativeTerms<neutrinos::ElectronNeutrinos<1>>::apply(
          non_flux_terms_dt_tilde_e, non_flux_terms_dt_tilde_s, tilde_e_flux,
          tilde_s_flux,

          make_not_null(
              &get<typename RadiationTransport::M1Grey::TimeDerivativeTerms<
                  neutrinos::ElectronNeutrinos<1>>::TildeSUp>(temp)),

          tilde_e, tilde_s, tilde_p, lapse, shift, spatial_metric,
          inv_spatial_metric,

          source_n, source_i, d_lapse, d_shift, d_spatial_metric,
          extrinsic_curvature);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.TimeDerivativeTerms",
                  "[Unit][M1Grey]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RadiationTransport/M1Grey"};

  pypp::check_with_random_values<1>(
      &forward_to_time_deriv, "TimeDerivative",
      {"non_flux_terms_dt_tilde_e", "non_flux_terms_dt_tilde_s", "tilde_e_flux",
       "tilde_s_flux"},
      {{{0.0, 1.0}}}, DataVector{5});
}
