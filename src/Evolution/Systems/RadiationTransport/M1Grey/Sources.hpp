// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace RadiationTransport::M1Grey {

// Implementation of the curvature source terms
// for the M1 system, for an individual species.
namespace detail {
void compute_sources_impl(gsl::not_null<Scalar<DataVector>*> source_tilde_e,
                          gsl::not_null<tnsr::i<DataVector, 3>*> source_tilde_s,
                          const Scalar<DataVector>& tilde_e,
                          const tnsr::i<DataVector, 3>& tilde_s,
                          const tnsr::II<DataVector, 3>& tilde_p,
                          const Scalar<DataVector>& lapse,
                          const tnsr::i<DataVector, 3>& d_lapse,
                          const tnsr::iJ<DataVector, 3>& d_shift,
                          const tnsr::ijj<DataVector, 3>& d_spatial_metric,
                          const tnsr::II<DataVector, 3>& inv_spatial_metric,
                          const tnsr::ii<DataVector, 3>& extrinsic_curvature,
                          const tnsr::ii<DataVector, 3>& spatial_metric,
                          const Scalar<DataVector>& emissivity,
                          const Scalar<DataVector>& absorption_opacity,
                          const Scalar<DataVector>& scattering_opacity,
                          const Scalar<DataVector>& tilde_j,
                          const Scalar<DataVector>& tilde_h_normal,
                          const tnsr::i<DataVector, 3>& tilde_h_spatial,
                          const tnsr::I<DataVector, 3>& spatial_velocity,
                          const Scalar<DataVector>& lorentz,
                          const Scalar<DataVector>& sqrt_det_spatial_metric);
}  // namespace detail

/*!
 * \brief Compute the curvature source terms for the flux-balanced
 * grey M1 radiation transport.
 *
 *
 * A flux-balanced system has the generic form:
 * \f[
 * \partial_t U_i + \partial_m F^m(U_i) = S(U_i)
 * \f]
 *
 * where \f$F^a()\f$ denotes the flux of a conserved variable \f$U_i\f$ and
 * \f$S()\f$ denotes the source term for the conserved variable.
 *
 * For the grey M1 formalism (neglecting coupling to the fluid):
 * \f{align*}
 * S({\tilde E}) &= \alpha \tilde P^{ij} K_{ij} - \tilde S^i \partial_i
 * \alpha,\\ S({\tilde S_i}) &= -\tilde E \partial_i \alpha + \tilde S_k
 * \partial_i \beta^k
 * + \frac{1}{2} \alpha \tilde P^{jk} \partial_i \gamma_{jk},
 * \f}
 *
 * where \f${\tilde E}\f$, \f${\tilde S_i}\f$, \f${\tilde P}^{ij}\f$ are the
 * densitized energy, momentum, and pressure tensor of the neutrinos/photons,
 * \f$K_{ij}\f$ is the extrinsic curvature, and \f$\alpha\f$, \f$\beta^i\f$,
 * \f$\gamma_{ij}\f$ are the lapse, shift and 3-metric.
 *
 * In the main function, we loop over all neutrino species, and then call
 * the actual implementation of the curvature source terms.
 */
template <typename... NeutrinoSpecies>
struct ComputeSources {
  using return_tags = tmpl::list<
      ::Tags::Source<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>...,
      ::Tags::Source<Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>...>;

  using argument_tags =
      tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeP<Frame::Inertial, NeutrinoSpecies>...,
                 gr::Tags::Lapse<DataVector>,
                 ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::Shift<DataVector, 3>, tmpl::size_t<3>,
                               Frame::Inertial>,
                 ::Tags::deriv<gr::Tags::SpatialMetric<DataVector, 3>,
                               tmpl::size_t<3>, Frame::Inertial>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 Tags::GreyEmissivity<NeutrinoSpecies>...,
                 Tags::GreyAbsorptionOpacity<NeutrinoSpecies>...,
                 Tags::GreyScatteringOpacity<NeutrinoSpecies>...,
                 Tags::TildeJ<NeutrinoSpecies>...,
                 Tags::TildeHNormal<NeutrinoSpecies>...,
                 Tags::TildeHSpatial<Frame::Inertial, NeutrinoSpecies>...,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>;

  static void apply(
      const gsl::not_null<typename Tags::TildeE<
          Frame::Inertial, NeutrinoSpecies>::type*>... sources_tilde_e,
      const gsl::not_null<typename Tags::TildeS<
          Frame::Inertial, NeutrinoSpecies>::type*>... sources_tilde_s,
      const typename Tags::TildeE<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_e,
      const typename Tags::TildeS<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_s,
      const typename Tags::TildeP<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_p,
      const Scalar<DataVector>& lapse, const tnsr::i<DataVector, 3>& d_lapse,
      const tnsr::iJ<DataVector, 3>& d_shift,
      const tnsr::ijj<DataVector, 3>& d_spatial_metric,
      const tnsr::II<DataVector, 3>& inv_spatial_metric,
      const tnsr::ii<DataVector, 3>& extrinsic_curvature,
      const tnsr::ii<DataVector, 3>& spatial_metric,
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
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_sources_impl(
        sources_tilde_e, sources_tilde_s, tilde_e, tilde_s, tilde_p, lapse,
        d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
        extrinsic_curvature, spatial_metric, emissivity, absorption_opacity,
        scattering_opacity, tilde_j, tilde_h_normal, tilde_h_spatial,
        spatial_velocity, lorentz, sqrt_det_spatial_metric));
  }
};

}  // namespace RadiationTransport::M1Grey
