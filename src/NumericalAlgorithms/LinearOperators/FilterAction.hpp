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
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
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
      const auto& blocks_to_filter = filter_helper.blocks_to_filter().value();

      // If nothing in the loop sets this to true, then our block isn't in the
      // blocks to filter, so we shouldn't filter in this element
      bool block_is_in_blocks_to_filter = false;
      for (const std::string& block_to_filter : blocks_to_filter) {
        // First check if the block to filter is our current block. If it is,
        // then we do want to filter.
        if (block_to_filter == block_name) {
          block_is_in_blocks_to_filter = true;
          break;
        } else if (block_groups.count(block_to_filter) == 1 and
                   block_groups.at(block_to_filter).count(block_name) == 1) {
          // If the block to filter isn't our current block, then we need to
          // check if the block to filter is a block group. If it is, check that
          // our block is in the block group. If it is, then we enable filtering
          block_is_in_blocks_to_filter = true;
          break;
        }
      }

      enable = block_is_in_blocks_to_filter;
    }

    if (not enable) {
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }

    const Mesh<volume_dim> mesh = db::get<domain::Tags::Mesh<volume_dim>>(box);
    const Matrix empty{};
    auto filter = make_array<volume_dim>(std::cref(empty));
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
          [&filter](const gsl::not_null<
                        typename TagsToFilter::type*>... tensors_to_filter,
                    const auto& local_mesh) {
            DataVector temp(local_mesh.number_of_grid_points(), 0.0);
            const auto helper = [&local_mesh, &filter,
                                 &temp](const auto tensor) {
              for (auto& component : *tensor) {
                temp = 0.0;
                apply_matrices(make_not_null(&temp), filter, component,
                               local_mesh.extents());
                component = temp;
              }
            };
            EXPAND_PACK_LEFT_TO_RIGHT(helper(tensors_to_filter));
          },
          make_not_null(&box), mesh);
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};
}  // namespace Actions
}  // namespace dg
