// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree {

/*!
 * \brief Compute the time derivative of the conserved variables for the GRFFE
 * equations with divergence cleaning.
 *
 */
struct TimeDerivativeTerms {
  struct LapseTimesElectricFieldOneForm : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };
  struct LapseTimesMagneticFieldOneForm : db::SimpleTag {
    using type = tnsr::i<DataVector, 3, Frame::Inertial>;
  };
  struct TildeJDrift : db::SimpleTag {
    using type = tnsr::I<DataVector, 3, Frame::Inertial>;
  };

  using temporary_tags = tmpl::list<
      // Flux terms
      LapseTimesElectricFieldOneForm, LapseTimesMagneticFieldOneForm,

      // Source terms
      TildeJDrift, gr::Tags::SpatialChristoffelFirstKind<DataVector, 3>,
      gr::Tags::SpatialChristoffelSecondKind<DataVector, 3>,
      gr::Tags::TraceSpatialChristoffelSecondKind<DataVector, 3>,

      // Need lapse, shift, and inverse spatial metric to be projected to
      // the boundary for Riemann solvers.
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>>;

  using argument_tags = tmpl::list<
      // EM tags
      Tags::TildeE, Tags::TildeB, Tags::TildePsi, Tags::TildePhi, Tags::TildeQ,
      Tags::TildeJ, Tags::KappaPsi, Tags::KappaPhi, Tags::ParallelConductivity,

      // GR-related tags
      gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>,
      gr::Tags::ExtrinsicCurvature<DataVector, 3>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>, tmpl::size_t<3>,
                    Frame::Inertial>>;

  static void apply(
      // Time derivatives returned by reference.
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          non_flux_terms_dt_tilde_e,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          non_flux_terms_dt_tilde_b,
      gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_psi,
      gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_phi,
      gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_tilde_q*/,

      // Fluxes returned by reference. Listed in the system struct as
      // flux_variables.
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
      gsl::not_null<tnsr::IJ<DataVector, 3, Frame::Inertial>*> tilde_b_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_psi_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_phi_flux,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_q_flux,

      // temporary tags
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          lapse_times_electric_field_one_form,
      gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
          lapse_times_magnetic_field_one_form,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_j_drift,
      gsl::not_null<tnsr::ijj<DataVector, 3, Frame::Inertial>*>
          spatial_christoffel_first_kind,
      gsl::not_null<tnsr::Ijj<DataVector, 3, Frame::Inertial>*>
          spatial_christoffel_second_kind,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          trace_spatial_christoffel_second,

      gsl::not_null<Scalar<DataVector>*> temp_lapse,
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> temp_shift,
      gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          temp_inverse_spatial_metric,

      // argument tags
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const Scalar<DataVector>& tilde_psi, const Scalar<DataVector>& tilde_phi,
      const Scalar<DataVector>& tilde_q,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_j,
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
      const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric);
};

}  // namespace ForceFree
