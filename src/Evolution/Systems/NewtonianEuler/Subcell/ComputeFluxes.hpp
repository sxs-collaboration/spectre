// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/Fluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::subcell {
namespace detail {
template <size_t Dim, typename TagsList, typename... ReturnTags,
          typename... ArgumentTags>
void compute_fluxes_impl(const gsl::not_null<Variables<TagsList>*> vars,
                         tmpl::list<ReturnTags...> /*meta*/,
                         tmpl::list<ArgumentTags...> /*meta*/) {
  NewtonianEuler::ComputeFluxes<Dim>::apply(
      make_not_null(&get<ReturnTags>(*vars))..., get<ArgumentTags>(*vars)...);
}
}  // namespace detail

/*!
 * \brief Helper function that calls `ComputeFluxes` by retrieving the return
 * and argument tags from `vars`.
 */
template <size_t Dim, typename TagsList>
void compute_fluxes(const gsl::not_null<Variables<TagsList>*> vars) {
  detail::compute_fluxes_impl<Dim>(
      vars, typename NewtonianEuler::ComputeFluxes<Dim>::return_tags{},
      typename NewtonianEuler::ComputeFluxes<Dim>::argument_tags{});
}
}  // namespace NewtonianEuler::subcell
