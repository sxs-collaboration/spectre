// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/ScalarAdvection/Fluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection::subcell {
namespace detail {
/*!
 * \brief Helper function that calls `ScalarAdvection::Fluxes::apply` by
 * retrieving the return and argument tags from the Variables object `vars`.
 */
template <size_t Dim, typename TagsList, typename... ReturnTags,
          typename... ArgumentTags>
void compute_fluxes_impl(const gsl::not_null<Variables<TagsList>*> vars,
                         tmpl::list<ReturnTags...> /*meta*/,
                         tmpl::list<ArgumentTags...> /*meta*/) {
  Fluxes<Dim>::apply(make_not_null(&get<ReturnTags>(*vars))...,
                     get<ArgumentTags>(*vars)...);
}
}  // namespace detail

/*!
 * \brief Compute fluxes for subcell variables.
 */
template <size_t Dim, typename TagsList>
void compute_fluxes(const gsl::not_null<Variables<TagsList>*> vars) {
  detail::compute_fluxes_impl<Dim>(vars, typename Fluxes<Dim>::return_tags{},
                                   typename Fluxes<Dim>::argument_tags{});
}
}  // namespace ScalarAdvection::subcell
