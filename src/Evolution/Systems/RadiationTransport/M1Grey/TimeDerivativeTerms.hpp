// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/DiscontinuousGalerkin/TimeDerivativeDecisions.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Fluxes.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Sources.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace RadiationTransport::M1Grey {
template <typename... NeutrinoSpecies>
struct TimeDerivativeTerms {
  struct TildeSUp : db::SimpleTag {
    using type = tnsr::I<DataVector, 3, Frame::Inertial>;
  };

  using temporary_tags =
      tmpl::list<TildeSUp, gr::Tags::InverseSpatialMetric<DataVector, 3>>;
  using argument_tags =
      tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeP<Frame::Inertial, NeutrinoSpecies>...,
                 gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>,
                 Tags::GreyEmissivity<NeutrinoSpecies>...,
                 Tags::GreyAbsorptionOpacity<NeutrinoSpecies>...,
                 Tags::GreyScatteringOpacity<NeutrinoSpecies>...,
                 Tags::TildeJ<NeutrinoSpecies>...,
                 Tags::TildeHNormal<NeutrinoSpecies>...,
                 Tags::TildeHSpatial<Frame::Inertial, NeutrinoSpecies>...,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>;

  static ::evolution::dg::TimeDerivativeDecisions<3> apply(
      const gsl::not_null<typename Tags::TildeE<
          Frame::Inertial,
          NeutrinoSpecies>::type*>... non_flux_terms_dt_tilde_e,
      const gsl::not_null<typename Tags::TildeS<
          Frame::Inertial,
          NeutrinoSpecies>::type*>... non_flux_terms_dt_tilde_s,

      const gsl::not_null<typename ::Tags::Flux<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>, tmpl::size_t<3>,
          Frame::Inertial>::type*>... tilde_e_flux,
      const gsl::not_null<typename ::Tags::Flux<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>, tmpl::size_t<3>,
          Frame::Inertial>::type*>... tilde_s_flux,

      const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_s_M,
      const gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*>
          temp_inv_spatial_metric,

      const typename Tags::TildeE<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_e,
      const typename Tags::TildeS<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_s,
      const typename Tags::TildeP<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_p,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const tnsr::i<DataVector, 3>& d_lapse,
      const tnsr::iJ<DataVector, 3>& d_shift,
      const tnsr::ijj<DataVector, 3>& d_spatial_metric,
      const tnsr::ii<DataVector, 3>& extrinsic_curvature,
      const Scalar<DataVector>& emissivity,
      const Scalar<DataVector>& absorption_opacity,
      const Scalar<DataVector>& scattering_opacity,
      const typename Tags::TildeJ<NeutrinoSpecies>::type&... tilde_j,
      const typename Tags::TildeHNormal<
          NeutrinoSpecies>::type&... tilde_h_normal,
      const typename Tags::TildeHSpatial<
          Frame::Inertial, NeutrinoSpecies>::type&... tilde_h_spatial,
      const tnsr::I<DataVector, 3>& spatial_velocity,
      const Scalar<DataVector>& lorentz,
      const Scalar<DataVector>& sqrt_det_spatial_metric) {
    *temp_inv_spatial_metric = inv_spatial_metric;
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_fluxes_impl(
        tilde_e_flux, tilde_s_flux, tilde_s_M, tilde_e, tilde_s, tilde_p, lapse,
        shift, spatial_metric, inv_spatial_metric));
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_sources_impl(
        non_flux_terms_dt_tilde_e, non_flux_terms_dt_tilde_s, tilde_e, tilde_s,
        tilde_p, lapse, d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
        extrinsic_curvature, spatial_metric, emissivity, absorption_opacity,
        scattering_opacity, tilde_j, tilde_h_normal, tilde_h_spatial,
        spatial_velocity, lorentz, sqrt_det_spatial_metric));
    return {true};
  }
};
}  // namespace RadiationTransport::M1Grey
