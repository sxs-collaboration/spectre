// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"  // for item_type
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  //  IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"  // for EXPAND_PACK_LEFT_TO...

// IWYU pragma: no_include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

/// \cond
class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Tags::deriv

namespace RadiationTransport {
namespace M1Grey {

/// Implementation of the curvature source terms
/// for the M1 system, for an individual species.
namespace detail {
void compute_sources_impl(
    gsl::not_null<Scalar<DataVector>*> source_tilde_e,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*> source_tilde_s,
    const Scalar<DataVector>& tilde_e,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::II<DataVector, 3, Frame::Inertial>& tilde_p,
    const Scalar<DataVector>& lapse,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>&
        extrinsic_curvature) noexcept;
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
  using return_tags =
      tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...>;

  using argument_tags = tmpl::list<
      Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
      Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
      Tags::TildeP<Frame::Inertial, NeutrinoSpecies>..., gr::Tags::Lapse<>,
      ::Tags::deriv<gr::Tags::Lapse<DataVector>, tmpl::size_t<3>,
                    Frame::Inertial>,
      ::Tags::deriv<gr::Tags::Shift<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      ::Tags::deriv<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>,
                    tmpl::size_t<3>, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<3>, gr::Tags::SqrtDetSpatialMetric<>,
      gr::Tags::ExtrinsicCurvature<3, Frame::Inertial, DataVector>>;

  static void apply(
      const gsl::not_null<db::item_type<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>*>... sources_tilde_e,
      const gsl::not_null<db::item_type<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>*>... sources_tilde_s,
      const db::item_type<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>&... tilde_e,
      const db::item_type<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>&... tilde_s,
      const db::item_type<
          Tags::TildeP<Frame::Inertial, NeutrinoSpecies>>&... tilde_p,
      const Scalar<DataVector>& lapse,
      const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
      const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
      const tnsr::ijj<DataVector, 3, Frame::Inertial>& d_spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>&
          extrinsic_curvature) noexcept {
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_sources_impl(
        sources_tilde_e, sources_tilde_s, tilde_e, tilde_s, tilde_p, lapse,
        d_lapse, d_shift, d_spatial_metric, inv_spatial_metric,
        extrinsic_curvature));
  }
};

}  // namespace M1Grey
}  // namespace RadiationTransport
