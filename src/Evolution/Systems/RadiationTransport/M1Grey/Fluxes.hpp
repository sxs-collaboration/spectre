// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>

#include "DataStructures/DataBox/DataBoxTag.hpp"  // for item_type
#include "DataStructures/DataBox/Prefixes.hpp"    // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/GrMhd/ValenciaDivClean/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"   // for not_null
#include "Utilities/TMPL.hpp"  // for EXPAND_PACK_LEFT_TO...

// IWYU pragma: no_include "PointwiseFunctions/GeneralRelativity/Tags.hpp"

// IWYU pragma: no_forward_declare Tags::Flux
// IWYU pragma: no_forward_declare Tensor

namespace RadiationTransport {
namespace M1Grey {

// Implementation of the M1 fluxes for individual neutrino species
namespace detail {
void compute_fluxes_impl(
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_e_flux,
    gsl::not_null<tnsr::Ij<DataVector, 3, Frame::Inertial>*> tilde_s_flux,
    gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*> tilde_s_M,
    const Scalar<DataVector>& tilde_e,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::II<DataVector, 3, Frame::Inertial>& tilde_p,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>&
        inv_spatial_metric) noexcept;
}  // namespace detail

/*!
 * \brief The fluxes of the conservative variables in the M1 scheme
 *
 * \f{align*}
 * F^i({\tilde E}) = &~ \alpha \gamma^{ij} {\tilde S}_j - \beta^j {\tilde E} \\
 * F^i({\tilde S}_j) = &~ \alpha {\tilde P}^{ik} \gamma_{kj} - \beta^i {\tilde
 * S}_j \f}
 *
 * where the conserved variables \f${\tilde E}\f$, \f${\tilde S}_i\f$,
 * are a generalized mass-energy density and momentum density.
 * Furthermore, \f${\tilde P^{ij}}\f$ is the pressure tensor density of the
 * radiation field, \f$\alpha\f$ is the lapse, \f$\beta^i\f$ is the shift,
 * \f$\gamma_{ij}\f$ the 3-metric, and \f$\gamma^{ij}\f$ its inverse.
 *
 * In the main function, we loop over all neutrino species, and then call
 * the actual implementation of the fluxes.
 */
template <typename... NeutrinoSpecies>
struct ComputeFluxes {
  using return_tags =
      tmpl::list<::Tags::Flux<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>,
                              tmpl::size_t<3>, Frame::Inertial>...,
                 ::Tags::Flux<Tags::TildeS<Frame::Inertial, NeutrinoSpecies>,
                              tmpl::size_t<3>, Frame::Inertial>...>;

  using argument_tags =
      tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeP<Frame::Inertial, NeutrinoSpecies>...,
                 gr::Tags::Lapse<>, gr::Tags::Shift<3>,
                 gr::Tags::SpatialMetric<3>, gr::Tags::InverseSpatialMetric<3>>;

  static void apply(
      const gsl::not_null<db::item_type<
          ::Tags::Flux<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>,
                       tmpl::size_t<3>, Frame::Inertial>>*>... tilde_e_flux,
      const gsl::not_null<db::item_type<
          ::Tags::Flux<Tags::TildeS<Frame::Inertial, NeutrinoSpecies>,
                       tmpl::size_t<3>, Frame::Inertial>>*>... tilde_s_flux,
      const db::item_type<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>&... tilde_e,
      const db::item_type<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>&... tilde_s,
      const db::item_type<
          Tags::TildeP<Frame::Inertial, NeutrinoSpecies>>&... tilde_p,
      const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>&
          inv_spatial_metric) noexcept {
    // Allocate memory for tildeS^i
    tnsr::I<DataVector, 3, Frame::Inertial> tilde_s_M(get(lapse).size());
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_fluxes_impl(
        tilde_e_flux, tilde_s_flux, &tilde_s_M, tilde_e, tilde_s, tilde_p,
        lapse, shift, spatial_metric, inv_spatial_metric));
  }
};
}  // namespace M1Grey
}  // namespace RadiationTransport
