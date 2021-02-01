// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;
/// \endcond

namespace grmhd::ValenciaDivClean {
/*!
 * \brief Compute the time derivative of the conserved variables for the
 * Valencia formulation of the GRMHD equations with divergence cleaning.
 */
struct TimeDerivativeTerms {
  struct MagneticFieldOneForm : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };
  struct TildeSUp : db::SimpleTag {
    using type = tnsr::I<DataVector, 3, Frame::Inertial>;
  };
  struct DensitizedStress : db::SimpleTag {
    using type = tnsr::II<DataVector, 3, Frame::Inertial>;
  };
  struct OneOverLorentzFactorSquared : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct PressureStarLapseSqrtDetSpatialMetric : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct TransportVelocity : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct PressureStar : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct EnthalpyTimesDensityWSquaredPlusBSquared : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct LapseTimesbOverW : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };

  using temporary_tags = tmpl::list<
      // Flux terms
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
      hydro::Tags::MagneticFieldOneForm<DataVector, 3, Frame::Inertial>,
      hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
      hydro::Tags::MagneticFieldSquared<DataVector>,
      OneOverLorentzFactorSquared, PressureStar,
      PressureStarLapseSqrtDetSpatialMetric, TransportVelocity,
      LapseTimesbOverW,

      // Source terms
      TildeSUp, DensitizedStress,
      gr::Tags::SpatialChristoffelFirstKind<3, Frame::Inertial, DataVector>,
      gr::Tags::SpatialChristoffelSecondKind<3, Frame::Inertial, DataVector>,
      gr::Tags::TraceSpatialChristoffelSecondKind<3, Frame::Inertial,
                                                  DataVector>,
      EnthalpyTimesDensityWSquaredPlusBSquared,

      // Need lapse, shift, and inverse spatial metric to be projected to the
      // boundary for Riemann solvers.
      gr::Tags::Lapse<>, gr::Tags::Shift<3>, gr::Tags::InverseSpatialMetric<3>>;
  using argument_tags = tmpl::list<
      grmhd::ValenciaDivClean::Tags::TildeD,
      grmhd::ValenciaDivClean::Tags::TildeTau,
      grmhd::ValenciaDivClean::Tags::TildeS<>,
      grmhd::ValenciaDivClean::Tags::TildeB<>,
      grmhd::ValenciaDivClean::Tags::TildePhi, gr::Tags::Lapse<>,
      gr::Tags::Shift<3>, gr::Tags::SqrtDetSpatialMetric<>,
      gr::Tags::SpatialMetric<3>, gr::Tags::InverseSpatialMetric<3>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      hydro::Tags::Pressure<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::MagneticField<DataVector, 3>,
      hydro::Tags::RestMassDensity<DataVector>,
      hydro::Tags::SpecificEnthalpy<DataVector>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>,
      grmhd::ValenciaDivClean::Tags::ConstraintDampingParameter>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_tilde_d*/,
      gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          non_flux_terms_dt_tilde_s,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          non_flux_terms_dt_tilde_b,
      gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_phi,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_d_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_tau_flux,
      gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,

      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          spatial_velocity_one_form,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          magnetic_field_one_form,
      gsl::not_null<Scalar<DataVector>*> magnetic_field_dot_spatial_velocity,
      gsl::not_null<Scalar<DataVector>*> magnetic_field_squared,
      gsl::not_null<Scalar<DataVector>*> one_over_w_squared,
      gsl::not_null<Scalar<DataVector>*> pressure_star,
      gsl::not_null<Scalar<DataVector>*>
          pressure_star_lapse_sqrt_det_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> transport_velocity,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> lapse_b_over_w,

      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_s_up,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          densitized_stress,
      gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*>
          spatial_christoffel_first_kind,
      gsl::not_null<tnsr::Ijj<DataVector, 3, Frame::Inertial>*>
          spatial_christoffel_second_kind,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          trace_spatial_christoffel_second,
      gsl::not_null<Scalar<DataVector>*> h_rho_w_squared_plus_b_squared,

      gsl::not_null<Scalar<DataVector>*> temp_lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> temp_shift,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          temp_inverse_spatial_metric,

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
      double constraint_damping_parameter) noexcept;
};
}  // namespace grmhd::ValenciaDivClean
