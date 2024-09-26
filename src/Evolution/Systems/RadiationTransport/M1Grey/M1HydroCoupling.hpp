// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to compute the M1-hydro
/// coupling terms.

#pragma once

#include <tuple>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Imex/Protocols/ImplicitSource.hpp"
#include "Evolution/Imex/Protocols/ImplicitSourceJacobian.hpp"
#include "Evolution/Imex/Tags/Jacobian.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Protocols/StaticReturnApplyable.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
class DataVector;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel

/// \endcond

namespace RadiationTransport {
namespace M1Grey {

// Implementation of the coupling terms for
// individual species
namespace detail {
void compute_m1_hydro_coupling_impl(
    gsl::not_null<Scalar<DataVector>*> source_n,
    gsl::not_null<tnsr::i<DataVector, 3>*> source_i,
    const Scalar<DataVector>& emissivity,
    const Scalar<DataVector>& absorption_opacity,
    const Scalar<DataVector>& scattering_opacity,
    const Scalar<DataVector>& comoving_energy_density,
    const Scalar<DataVector>& comoving_momentum_density_normal,
    const tnsr::i<DataVector, 3>& comoving_momentum_density_spatial,
    const tnsr::I<DataVector, 3>& fluid_velocity,
    const Scalar<DataVector>& fluid_lorentz_factor,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const Scalar<DataVector>& sqrt_det_spatial_metric);

void compute_m1_hydro_coupling_jacobian_impl(
    gsl::not_null<Scalar<DataVector>*> deriv_e_source_e,
    gsl::not_null<tnsr::i<DataVector, 3>*> deriv_e_source_s,
    gsl::not_null<tnsr::I<DataVector, 3>*> deriv_s_source_e,
    gsl::not_null<tnsr::Ij<DataVector, 3>*> deriv_s_source_s,
    const tnsr::i<DataVector, 3>& tilde_s, const Scalar<DataVector>& tilde_e,
    const Scalar<DataVector>& emissivity,
    const Scalar<DataVector>& absorption_opacity,
    const Scalar<DataVector>& scattering_opacity,
    const tnsr::I<DataVector, 3>& fluid_velocity,
    const Scalar<DataVector>& fluid_lorentz_factor,
    const Scalar<DataVector>& closure_factor,
    const Scalar<DataVector>& comoving_energy_density,
    const tnsr::i<DataVector, 3>& comoving_momentum_density_spatial,
    const Scalar<DataVector>& comoving_momentum_density_normal,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3>& spatial_metric,
    const tnsr::II<DataVector, 3>& inverse_spatial_metric);
}  // namespace detail

template <typename NeutrinoSpeciesList>
struct ComputeM1HydroCoupling;
/*!
 * Compute the source terms of the M1 equation due to neutrino-matter
 * interactions.
 * These are:
 *
 * \f{align}{
 * \partial_t \tilde E = \alpha W (\sqrt{\gamma} \eta - \kappa_a \tilde J)
 * + \alpha (\kappa_a + \kappa_s) H_n\\
 * \partial_t \tilde S_i = \alpha u_i (\sqrt{\gamma} \eta - \kappa_a \tilde J)
 * - \alpha  (\kappa_a + \kappa_s) H_i.
 * \f}
 *
 * with \f$W\f$ the Lorentz factor, \f$u_i = W v_i\f$ the spatial components of
 * the fluid 4-velocity, \f$\eta\f$ the emissivity, \f$\kappa_{a,s}\f$
 * the absorption and scattering opacities, \f$J\f$ the comoving energy
 * density, and \f$H_{n,i}\f$ the normal and spatial components of the comoving
 * flux density.
 *
 * The function returns in `source_n` the energy source and in
 * `source_i` the momentum source. We write a separate action for these
 * sources to make it easier to switch between implicit / explicit time
 * stepping, as well as to add the source terms to both the fluid and M1
 * evolutions.
 */
template <typename... NeutrinoSpecies>
struct ComputeM1HydroCoupling<tmpl::list<NeutrinoSpecies...>>
    : tt::ConformsTo<imex::protocols::ImplicitSource>,
      tt::ConformsTo<protocols::StaticReturnApplyable> {
  using return_tags = tmpl::list<
      ::Tags::Source<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>...,
      ::Tags::Source<Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>...>;

  using argument_tags = tmpl::list<
      Tags::GreyEmissivity<NeutrinoSpecies>...,
      Tags::GreyAbsorptionOpacity<NeutrinoSpecies>...,
      Tags::GreyScatteringOpacity<NeutrinoSpecies>...,
      Tags::TildeJ<NeutrinoSpecies>..., Tags::TildeHNormal<NeutrinoSpecies>...,
      Tags::TildeHSpatial<Frame::Inertial, NeutrinoSpecies>...,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::LorentzFactor<DataVector>, gr::Tags::Lapse<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>>;

  static void apply(
      const gsl::not_null<typename Tags::M1HydroCouplingNormal<
          NeutrinoSpecies>::type*>... source_n,
      const gsl::not_null<typename Tags::M1HydroCouplingSpatial<
          Frame::Inertial, NeutrinoSpecies>::type*>... source_i,
      const typename Tags::GreyEmissivity<NeutrinoSpecies>::type&... emissivity,
      const typename Tags::GreyAbsorptionOpacity<
          NeutrinoSpecies>::type&... absorption_opacity,
      const typename Tags::GreyScatteringOpacity<
          NeutrinoSpecies>::type&... scattering_opacity,
      const typename Tags::TildeJ<NeutrinoSpecies>::type&... tilde_j,
      const typename Tags::TildeHNormal<NeutrinoSpecies>::type&... tilde_hn,
      const typename Tags::TildeHSpatial<Frame::Inertial,
                                         NeutrinoSpecies>::type&... tilde_hi,
      const tnsr::I<DataVector, 3>& spatial_velocity,
      const Scalar<DataVector>& lorentz_factor, const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3>& spatial_metric,
      const Scalar<DataVector>& sqrt_det_spatial_metric) {
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_m1_hydro_coupling_impl(
        source_n, source_i, emissivity, absorption_opacity, scattering_opacity,
        tilde_j, tilde_hn, tilde_hi, spatial_velocity, lorentz_factor, lapse,
        spatial_metric, sqrt_det_spatial_metric));
  }
};

template <typename NeutrinoSpecies>
struct ComputeM1HydroCouplingJacobian
    : tt::ConformsTo<imex::protocols::ImplicitSourceJacobian>,
      tt::ConformsTo<protocols::StaticReturnApplyable> {
  using return_tags = tmpl::list<
      imex::Tags::Jacobian<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>,
          ::Tags::Source<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>>,
      imex::Tags::Jacobian<
          Tags::TildeE<Frame::Inertial, NeutrinoSpecies>,
          ::Tags::Source<Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>>,
      imex::Tags::Jacobian<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>,
          ::Tags::Source<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>>>,
      imex::Tags::Jacobian<
          Tags::TildeS<Frame::Inertial, NeutrinoSpecies>,
          ::Tags::Source<Tags::TildeS<Frame::Inertial, NeutrinoSpecies>>>>;

  using argument_tags = tmpl::list<
      Tags::TildeS<Frame::Inertial, NeutrinoSpecies>,
      Tags::TildeE<Frame::Inertial, NeutrinoSpecies>,
      Tags::GreyEmissivity<NeutrinoSpecies>,
      Tags::GreyAbsorptionOpacity<NeutrinoSpecies>,
      Tags::GreyScatteringOpacity<NeutrinoSpecies>,
      hydro::Tags::SpatialVelocity<DataVector, 3>,
      hydro::Tags::LorentzFactor<DataVector>,
      Tags::ClosureFactor<NeutrinoSpecies>, Tags::TildeJ<NeutrinoSpecies>,
      Tags::TildeHSpatial<Frame::Inertial, NeutrinoSpecies>,
      Tags::TildeHNormal<NeutrinoSpecies>, gr::Tags::Lapse<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3>,
      gr::Tags::InverseSpatialMetric<DataVector, 3>>;

  static void apply(
      const gsl::not_null<Scalar<DataVector>*> deriv_e_source_e,
      const gsl::not_null<tnsr::i<DataVector, 3>*> deriv_e_source_s,
      const gsl::not_null<tnsr::I<DataVector, 3>*> deriv_s_source_e,
      const gsl::not_null<tnsr::Ij<DataVector, 3>*> deriv_s_source_s,
      const tnsr::i<DataVector, 3>& tilde_s, const Scalar<DataVector>& tilde_e,
      const Scalar<DataVector>& emissivity,
      const Scalar<DataVector>& absorption_opacity,
      const Scalar<DataVector>& scattering_opacity,
      const tnsr::I<DataVector, 3>& fluid_velocity,
      const Scalar<DataVector>& fluid_lorentz_factor,
      const Scalar<DataVector>& closure_factor,
      const Scalar<DataVector>& comoving_energy_density,
      const tnsr::i<DataVector, 3>& comoving_momentum_density_spatial,
      const Scalar<DataVector>& comoving_momentum_density_normal,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3>& spatial_metric,
      const tnsr::II<DataVector, 3>& inverse_spatial_metric) {
    detail::compute_m1_hydro_coupling_jacobian_impl(
        deriv_e_source_e, deriv_e_source_s, deriv_s_source_e, deriv_s_source_s,
        tilde_s, tilde_e, emissivity, absorption_opacity, scattering_opacity,
        fluid_velocity, fluid_lorentz_factor, closure_factor,
        comoving_energy_density, comoving_momentum_density_spatial,
        comoving_momentum_density_normal, lapse, spatial_metric,
        inverse_spatial_metric);
  }
};
}  // namespace M1Grey
}  // namespace RadiationTransport
