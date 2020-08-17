// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Evolution/Systems/Cce/IntegrandInputSteps.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/PreSwshDerivatives.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
namespace Actions {

/*!
 * \ingroup ActionsGroup
 * \brief Calculates the Bondi quantities that are required for any of the
 * `CalculateScriPlusValue` mutators.
 *
 * \details Internally dispatches to the `PreSwshDerivatives` and
 * `Spectral::Swsh::AngularDerivatives` utilities to perform the radial and
 * angular differentiation that is required to prepare all of the Bondi
 * quantities needed for evaluating the scri+ values. This relies on the
 * typelists `Cce::all_pre_swsh_derivative_tags_for_scri`,
 * `Cce::all_boundary_pre_swsh_derivative_tags_for_scri`,
 * `Cce::all_swsh_derivative_tags_for_scri`, and
 * `Cce::all_boundary_swsh_derivative_tags_for_scri` to determine which
 * calculations to perform.
 */
struct CalculateScriInputs {
  using const_global_cache_tags =
      tmpl::list<Tags::LMax, Tags::NumberOfRadialPoints>;

  template <typename DbTags, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static std::tuple<db::DataBox<DbTags>&&> apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    tmpl::for_each<
        tmpl::append<all_pre_swsh_derivative_tags_for_scri,
                     all_boundary_pre_swsh_derivative_tags_for_scri>>([&box](
        auto pre_swsh_derivative_tag_v) noexcept {
      using pre_swsh_derivative_tag =
          typename decltype(pre_swsh_derivative_tag_v)::type;
      db::mutate_apply<PreSwshDerivatives<pre_swsh_derivative_tag>>(
          make_not_null(&box));
    });

    db::mutate_apply<
        Spectral::Swsh::AngularDerivatives<all_swsh_derivative_tags_for_scri>>(
        make_not_null(&box));
    boundary_derivative_impl(box, db::get<Tags::LMax>(box),
                             all_boundary_swsh_derivative_tags_for_scri{});

    tmpl::for_each<all_swsh_derivative_tags_for_scri>([&box](
        auto derivative_tag_v) noexcept {
      using derivative_tag = typename decltype(derivative_tag_v)::type;
      ::Cce::detail::apply_swsh_jacobian_helper<derivative_tag>(
          make_not_null(&box), typename ApplySwshJacobianInplace<
                                   derivative_tag>::on_demand_argument_tags{});
    });
    return {std::move(box)};
  }

  template <typename DbTags, typename... TagPack>
  static void boundary_derivative_impl(
      db::DataBox<DbTags>& box, const size_t l_max,
      tmpl::list<TagPack...> /*meta*/) noexcept {
    db::mutate<TagPack...>(
        make_not_null(&box),
        [&l_max](const gsl::not_null<typename TagPack::type*>... derivatives,
                 const typename TagPack::derivative_of::
                     type&... arguments) noexcept {
          Spectral::Swsh::angular_derivatives<
              tmpl::list<typename TagPack::derivative_kind...>>(
              l_max, 1, make_not_null(&get(*derivatives))...,
              get(arguments)...);
        },
        db::get<typename TagPack::derivative_of>(box)...);
  }
};
}  // namespace Actions
}  // namespace Cce
