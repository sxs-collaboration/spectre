// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Fluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace grmhd::ValenciaDivClean::subcell {
namespace detail {
template <typename TagsList, typename... ReturnTags, typename... ArgumentTags>
void compute_fluxes_impl(const gsl::not_null<Variables<TagsList>*> vars,
                         tmpl::list<ReturnTags...> /*meta*/,
                         tmpl::list<ArgumentTags...> /*meta*/) noexcept {
  grmhd::ValenciaDivClean::ComputeFluxes::apply(
      make_not_null(&get<ReturnTags>(*vars))..., get<ArgumentTags>(*vars)...);
}
}  // namespace detail

/*!
 * \brief Helper function that calls `ComputeFluxes` by retrieving the return
 * and argument tags from `vars`.
 */
template <typename TagsList>
void compute_fluxes(const gsl::not_null<Variables<TagsList>*> vars) noexcept {
  detail::compute_fluxes_impl(
      vars, typename grmhd::ValenciaDivClean::ComputeFluxes::return_tags{},
      typename grmhd::ValenciaDivClean::ComputeFluxes::argument_tags{});
}
}  // namespace grmhd::ValenciaDivClean::subcell
