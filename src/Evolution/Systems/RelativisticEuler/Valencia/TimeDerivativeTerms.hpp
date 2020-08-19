// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/TagsDeclarations.hpp"
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

namespace RelativisticEuler::Valencia {
/*!
 * \brief Compute the time derivative of the conserved variables for the
 * Valencia formulation of the relativistic Euler equations.
 */
template <size_t Dim>
struct TimeDerivativeTerms {
  struct PressureLapseSqrtDetSpatialMetric : db::SimpleTag {
    using type = Scalar<DataVector>;
  };
  struct TransportVelocity : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  struct TildeSUp : db::SimpleTag {
    using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
  };
  struct DensitizedStress : db::SimpleTag {
    using type = tnsr::II<DataVector, Dim, Frame::Inertial>;
  };

  using temporary_tags = tmpl::list<
      // Flux terms
      PressureLapseSqrtDetSpatialMetric, TransportVelocity,

      // Source terms
      TildeSUp, DensitizedStress>;

  using argument_tags = tmpl::list<
      RelativisticEuler::Valencia::Tags::TildeD,
      RelativisticEuler::Valencia::Tags::TildeTau,
      RelativisticEuler::Valencia::Tags::TildeS<Dim>,

      // For fluxes (and maybe sources)
      gr::Tags::Lapse<>, gr::Tags::Shift<Dim>, gr::Tags::SqrtDetSpatialMetric<>,
      hydro::Tags::Pressure<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, Dim>,

      // For sources
      ::Tags::deriv<gr::Tags::Lapse<>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<Dim>, tmpl::size_t<Dim>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<Dim>, tmpl::size_t<Dim>,
                    Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<Dim>, gr::Tags::ExtrinsicCurvature<Dim>>;

  static void apply(
      gsl::not_null<Scalar<DataVector>*> /*non_flux_terms_dt_tilde_d*/,
      gsl::not_null<Scalar<DataVector>*> non_flux_terms_dt_tilde_tau,
      gsl::not_null<tnsr::i<DataVector, Dim, Frame::Inertial>*>
          non_flux_terms_dt_tilde_s,

      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_d_flux,
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_tau_flux,
      gsl::not_null<tnsr::Ij<DataVector, Dim, Frame::Inertial>*> tilde_s_flux,

      // For fluxes
      gsl::not_null<Scalar<DataVector>*> pressure_lapse_sqrt_det_spatial_metric,
      gsl::not_null<Scalar<DataVector>*> transport_velocity,

      // For sources
      gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*> tilde_s_up,
      gsl::not_null<tnsr::II<DataVector, Dim, Frame::Inertial>*>
          densitized_stress,

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
          extrinsic_curvature) noexcept;
};
}  // namespace RelativisticEuler::Valencia
