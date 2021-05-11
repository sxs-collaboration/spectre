// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Fluxes.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Sources.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace RadiationTransport::M1Grey {
template <typename... NeutrinoSpecies>
struct TimeDerivativeTerms {
  struct TildeSUp : db::SimpleTag {
    using type = tnsr::I<DataVector, 3, Frame::Inertial>;
  };

  using temporary_tags =
      tmpl::list<TildeSUp, gr::Tags::InverseSpatialMetric<3>>;
  using argument_tags = tmpl::list<
      Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
      Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
      Tags::TildeP<Frame::Inertial, NeutrinoSpecies>..., gr::Tags::Lapse<>,
      gr::Tags::Shift<3>, gr::Tags::SpatialMetric<3>,
      gr::Tags::InverseSpatialMetric<3>,

      Tags::M1HydroCouplingNormal<NeutrinoSpecies>...,
      Tags::M1HydroCouplingSpatial<Frame::Inertial, NeutrinoSpecies>...,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>;

  static void apply(
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

      const typename Tags::M1HydroCouplingNormal<
          NeutrinoSpecies>::type&... source_n,
      const typename Tags::M1HydroCouplingSpatial<
          Frame::Inertial, NeutrinoSpecies>::type&... source_i,
      const tnsr::i<DataVector, 3>& d_lapse,
      const tnsr::iJ<DataVector, 3>& d_shift,
      const tnsr::ijj<DataVector, 3>& d_spatial_metric,
      const tnsr::ii<DataVector, 3>& extrinsic_curvature) noexcept {
    *temp_inv_spatial_metric = inv_spatial_metric;
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_fluxes_impl(
        tilde_e_flux, tilde_s_flux, tilde_s_M, tilde_e, tilde_s, tilde_p, lapse,
        shift, spatial_metric, inv_spatial_metric));
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_sources_impl(
        non_flux_terms_dt_tilde_e, non_flux_terms_dt_tilde_s, tilde_e, tilde_s,
        tilde_p, source_n, source_i, lapse, d_lapse, d_shift, d_spatial_metric,
        inv_spatial_metric, extrinsic_curvature));
  }
};
}  // namespace RadiationTransport::M1Grey
