// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean {
namespace detail {
template <typename FluxTags, typename BoundaryVarsType, typename... ReturnTags,
          typename... ConservativeTags, typename... ArgumentTags,
          typename... FluxArgumentTags>
void compute_fluxes_from_primitives_impl(
    const gsl::not_null<Variables<FluxTags>*> flux_vars,
    const BoundaryVarsType& boundary_vars, tmpl::list<ReturnTags...> /*meta*/,
    tmpl::list<ConservativeTags...> /*meta*/,
    tmpl::list<ArgumentTags...> /*meta*/,
    tmpl::list<FluxArgumentTags...> /*meta*/) {
  Variables<typename tmpl::list<ConservativeTags...>> conserved_vars{
      flux_vars->number_of_grid_points()};
  grmhd::ValenciaDivClean::ConservativeFromPrimitive::apply(
      make_not_null(&get<ConservativeTags>(conserved_vars))...,
      get<ArgumentTags>(boundary_vars)...);
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&get<ReturnTags>(*flux_vars))...,
      get<ConservativeTags>(conserved_vars)...,
      get<FluxArgumentTags>(boundary_vars)...);
}
}  // namespace detail

/*!
 * \brief Helper function that computes fluxes from a given set of primitive
 * variables. Primarily used for filling neighbor data when flux information is
 * needed for boundaries.
 *
 * \warning This computes the conservative variables internally. If you also
 * need those, you shouldn't call this function.
 */
template <typename FluxTags, typename BoundaryVarsType>
void compute_fluxes_from_primitives(
    const gsl::not_null<Variables<FluxTags>*> flux_vars,
    const BoundaryVarsType& boundary_vars) {
  using ConservativeTags =
      typename grmhd::ValenciaDivClean::ConservativeFromPrimitive::return_tags;
  using FluxArgumentTags =
      typename grmhd::ValenciaDivClean::ComputeFluxes::argument_tags;
  using flux_arg_tags = tmpl::back<tmpl::split_at<
      FluxArgumentTags, tmpl::next<tmpl::index_of<
                            FluxArgumentTags, tmpl::back<ConservativeTags>>>>>;
  detail::compute_fluxes_from_primitives_impl(
      flux_vars, boundary_vars,
      typename grmhd::ValenciaDivClean::ComputeFluxes::return_tags{},
      ConservativeTags{},
      typename grmhd::ValenciaDivClean::ConservativeFromPrimitive::
          argument_tags{},
      flux_arg_tags{});
}
}  // namespace grmhd::ValenciaDivClean
