// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <optional>
#include <string>
#include <tuple>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace domain::Tags {
template <size_t VolumeDim>
struct Domain;
}  // namespace domain::Tags
namespace Filters::Tags {
template <typename FilterType>
struct Filter;
}  // namespace Filters::Tags
/// \endcond

namespace dg {
namespace Actions {
namespace Filter_detail {
template <bool SameSize, bool SameList>
struct FilterAllEvolvedVars {
  template <typename EvolvedVarsTagList, typename... FilterTags>
  using f = std::integral_constant<bool, false>;
};

template <>
struct FilterAllEvolvedVars<true, false> {
  template <typename EvolvedVarsTagList, typename... FilterTags>
  using f =
      std::integral_constant<bool, tmpl2::flat_all_v<tmpl::list_contains_v<
                                       EvolvedVarsTagList, FilterTags>...>>;
};

template <>
struct FilterAllEvolvedVars<true, true> {
  template <typename EvolvedVarsTagList, typename... FilterTags>
  using f = std::integral_constant<bool, true>;
};
}  // namespace Filter_detail

/// \cond
template <typename FilterType, typename TagsToFilterList>
struct Filter;
/// \endcond

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Applies a filter to the specified tags.
 *
 * If different Filters are desired for different tags then multiple `Filter`
 * actions must be inserted into the action list with different `FilterType`.
 * Here is an example of an action list with two different exponential filters:
 *
 * \snippet LinearOperators/Test_Filtering.cpp action_list_example
 *
 * Uses:
 * - GlobalCache:
 *   - `Filter`
 * - DataBox:
 *   - `Tags::Mesh`
 * - DataBox changes:
 *   - Adds: nothing
 *   - Removes: nothing
 *   - Modifies:
 *     - `TagsToFilter`
 * - System:
 *   - `volume_dim`
 *   - `variables_tag`
 *
 */
template <typename FilterType, typename... TagsToFilter>
class Filter<FilterType, tmpl::list<TagsToFilter...>> {
 public:
  using const_global_cache_tags =
      tmpl::list<::Filters::Tags::Filter<FilterType>>;

  template <typename DbTags, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTags>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    constexpr size_t volume_dim = Metavariables::system::volume_dim;
    using evolved_vars_tag = typename Metavariables::system::variables_tag;
    using evolved_vars_tags_list = typename evolved_vars_tag::tags_list;
    const FilterType& filter_helper =
        Parallel::get<::Filters::Tags::Filter<FilterType>>(cache);
    const size_t block_id =
        db::get<domain::Tags::Element<volume_dim>>(box).id().block_id();
    const auto& domain = Parallel::get<domain::Tags::Domain<volume_dim>>(cache);
    const auto& block_groups = domain.block_groups();
    const std::string& block_name = domain.blocks()[block_id].name();

    // Technically this whole next block could be done on a single line, but
    // then it would be very dense and hard to understand. This way is easier to
    // read and understand
    bool enable = filter_helper.enable();
    // Only do this check if filtering is enabled. A `nullopt` means all blocks
    // are allowed to do filtering
    if (enable and filter_helper.blocks_to_filter().has_value()) {
      // Enable filtering for this block if it's in any of the listed groups
      enable = alg::any_of(
          filter_helper.blocks_to_filter().value(),
          [&block_name, &block_groups](const std::string& block_to_filter) {
            return domain::block_is_in_group(block_name, block_to_filter,
                                             block_groups);
          });
    }

    if (not enable) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const Mesh<volume_dim> mesh = db::get<domain::Tags::Mesh<volume_dim>>(box);
    const Matrix empty{};
    std::array<std::reference_wrapper<const Matrix>, volume_dim> filter =
        make_array<volume_dim>(std::cref(empty));
    for (size_t d = 0; d < volume_dim; d++) {
      gsl::at(filter, d) =
          std::cref(filter_helper.filter_matrix(mesh.slice_through(d)));
    }

    // In the case that the tags we are filtering are all the evolved variables
    // we filter the entire Variables at once to be more efficient. This case is
    // the first branch of the `if-else`.
    if (Filter_detail::FilterAllEvolvedVars<
            sizeof...(TagsToFilter) ==
                tmpl::size<evolved_vars_tags_list>::value,
            std::is_same_v<evolved_vars_tags_list,
                           tmpl::list<TagsToFilter...>>>::
            template f<evolved_vars_tags_list, TagsToFilter...>::value) {
      db::mutate<typename Metavariables::system::variables_tag>(
          [&filter](const gsl::not_null<
                        typename Metavariables::system::variables_tag::type*>
                        vars,
                    const auto& local_mesh) {
            *vars = apply_matrices(filter, *vars, local_mesh.extents());
          },
          make_not_null(&box), mesh);
    } else {
      db::mutate<TagsToFilter...>(
          [](const gsl::not_null<
                 typename TagsToFilter::type*>... tensors_to_filter,
             const Mesh<volume_dim>& local_mesh,
             const std::array<std::reference_wrapper<const Matrix>, volume_dim>&
                 local_filter) {
            DataVector temp(local_mesh.number_of_grid_points(), 0.0);
            const auto helper = [&local_mesh, &local_filter,
                                 &temp](const auto tensor) {
              for (auto& component : *tensor) {
                temp = 0.0;
                apply_matrices(make_not_null(&temp), local_filter, component,
                               local_mesh.extents());
                component = temp;
              }
            };
            EXPAND_PACK_LEFT_TO_RIGHT(helper(tensors_to_filter));
          },
          make_not_null(&box), mesh, filter);
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace dg
