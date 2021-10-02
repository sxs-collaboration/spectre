// Distributed under the MIT License.
// See LICENSE.txt for details.

///\file
/// Defines functions to compute the M1 closure

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"  // IWYU pragma: keep
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"   // for not_...
#include "Utilities/TMPL.hpp"  // for EXPAND_PACK_LEFT_TO...

// IWYU pragma: no_forward_declare Tensor

/// \cond
class DataVector;
/// \endcond

namespace RadiationTransport {
namespace M1Grey {

// Implementation of the M1 closure for an
// individual species
namespace detail {
void compute_closure_impl(
    gsl::not_null<Scalar<DataVector>*> closure_factor,
    gsl::not_null<tnsr::II<DataVector, 3, Frame::Inertial>*> pressure_tensor,
    gsl::not_null<Scalar<DataVector>*> comoving_energy_density,
    gsl::not_null<Scalar<DataVector>*> comoving_momentum_density_normal,
    gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        comoving_momentum_density_spatial,
    const Scalar<DataVector>& energy_density,
    const tnsr::i<DataVector, 3, Frame::Inertial>& momentum_density,
    const tnsr::I<DataVector, 3, Frame::Inertial>& fluid_velocity,
    const Scalar<DataVector>& fluid_lorentz_factor,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric);
}  // namespace detail

template <typename NeutrinoSpeciesList>
struct ComputeM1Closure;
/*!
 * Compute the 2nd moment of the neutrino distribution function
 * in the inertial frame (pressure tensor) from the 0th (energy density)
 * and 1st (momentum density) moments. The M1 closure sets
 * \f{align}{
 * P_{ij} = d_{\rm thin} P_{ij,\rm thin} + d_{\rm thick} P_{ij,\rm thick}
 * \f}
 * with \f$P_{ij}\f$ the pressure tensor in the inertial frame, and
 * \f$P_{ij,thin/thick}\f$ its value assuming an optically thick / optically
 * thin medium.
 *
 * Following the algorithm described in Appendix A of \cite Foucart2015vpa,
 * we choose the Minerbo closure (\cite Minerbo1978) which sets
 * \f{align}{
 * d_{\rm thin} = & 1.5\chi-0.5\\
 * d_{\rm thick} = & 1.5-1.5\chi \\
 * \chi = & \frac{1}{3} + \xi^2 \frac{6-2\xi+6\xi^2}{15} \\
 * \xi = & H^aH_a/J^2
 * \f}
 * with \f$J\f$ the energy density in the
 * frame comoving with the fluid, and \f$H^a\f$ the momentum density
 * in that frame.
 * The optically thick closure is
 * \f{align}{
 * K^{ab,\rm thick} = \frac{J_{\rm thick}}{3} (g^{ab}+u^a u^b)
 * \f}
 * with \f$ K^{ab,\rm thick} \f$ the pressure tensor in the frame comoving with
 * the fluid, and \f$ u^a \f$ the fluid 4-velocity, while the optically thin
 * closure is
 * \f{align}{
 * P^{ij,\rm thin} = \frac{S^i S^j}{S^k S_k} E
 * \f}
 * with \f$ E \f$, \f$ S^{i}
 * \f$ the 0th and 1st moments in the inertial frame (see paper for the
 * computation of \f$P_{ij,thick}\f$ from the other moments).
 *
 * The main step of this function is to solve the equation
 * \f{align}{
 *  \frac{\xi^2 J^2 - H^aH_a}{E^2} = 0
 * \f}
 * for \f$\xi\f$, with \f$J\f$ and \f$H^a\f$ being functions
 * of \f$\xi\f$ and of the input variables (the 0th and 1st moments
 * in the inertial frame). This is done by separating \f$H^a\f$ as
 * \f{align}{
 * H_a = - H_t t_a - H_v v_a - H_f F_a
 * \f}
 * (with \f$v^a\f$ the fluid 3-velocity and \f$t^a\f$ the unit normal
 * to the slice) and then writing each of the
 * variables \f$J\f$, \f$H_n\f$, \f$H_v\f$, \f$H_f\f$ as
 * \f{align}{
 * X = X_0 + d_{\rm thin} X_{\rm thin} + d_{\rm thick} X_{\rm thick}
 * \f}
 * so that evaluating
 * \f{align}{
 * \frac{\xi^2 J^2 - H^aH_a}{E^2} = 0
 * \f}
 * for a given \f$\xi\f$ only requires recomputing \f$d_{\rm thin,thick}\f$
 * and their derivatives with respect to \f$\xi\f$.
 * We perform the root-finding using a Newton-Raphson algorithm, with the
 * accuracy set by the variable root_find_number_of_digits (6 significant digits
 * at the moment).
 *
 * The function returns the closure factors \f$\xi\f$ (to be used as initial
 * guess for this function at the next step), the pressure tensor \f$P_{ij}\f$,
 * and the neutrino moments in the frame comoving with the fluid.
 * The momentum density in the frame comoving with the fluid
 * is decomposed into its normal component \f$ H^a t_a\f$, and its spatial
 * components \f$ \gamma_{ia} H^a\f$.
 */
template <typename... NeutrinoSpecies>
struct ComputeM1Closure<tmpl::list<NeutrinoSpecies...>> {
  using return_tags =
      tmpl::list<Tags::ClosureFactor<NeutrinoSpecies>...,
                 Tags::TildeP<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeJ<NeutrinoSpecies>...,
                 Tags::TildeHNormal<NeutrinoSpecies>...,
                 Tags::TildeHSpatial<Frame::Inertial, NeutrinoSpecies>...>;

  using argument_tags =
      tmpl::list<Tags::TildeE<Frame::Inertial, NeutrinoSpecies>...,
                 Tags::TildeS<Frame::Inertial, NeutrinoSpecies>...,
                 hydro::Tags::SpatialVelocity<DataVector, 3>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 gr::Tags::SpatialMetric<3>, gr::Tags::InverseSpatialMetric<3>>;

  static void apply(
      const gsl::not_null<typename Tags::ClosureFactor<
          NeutrinoSpecies>::type*>... closure_factor,
      const gsl::not_null<typename Tags::TildeP<
          Frame::Inertial, NeutrinoSpecies>::type*>... tilde_p,
      const gsl::not_null<
          typename Tags::TildeJ<NeutrinoSpecies>::type*>... tilde_j,
      const gsl::not_null<
          typename Tags::TildeHNormal<NeutrinoSpecies>::type*>... tilde_hn,
      const gsl::not_null<typename Tags::TildeHSpatial<
          Frame::Inertial, NeutrinoSpecies>::type*>... tilde_hi,
      const typename Tags::TildeE<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_e,
      const typename Tags::TildeS<Frame::Inertial,
                                  NeutrinoSpecies>::type&... tilde_s,
      const tnsr::I<DataVector, 3>& spatial_velocity,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::ii<DataVector, 3>& spatial_metric,
      const tnsr::II<DataVector, 3>& inv_spatial_metric) {
    EXPAND_PACK_LEFT_TO_RIGHT(detail::compute_closure_impl(
        closure_factor, tilde_p, tilde_j, tilde_hn, tilde_hi, tilde_e, tilde_s,
        spatial_velocity, lorentz_factor, spatial_metric, inv_spatial_metric));
  }
};

}  // namespace M1Grey
}  // namespace RadiationTransport
