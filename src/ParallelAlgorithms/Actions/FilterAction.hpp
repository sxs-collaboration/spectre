// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <optional>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Structure/BlockGroups.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Tags/Filter.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/CallWithDynamicType.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <size_t Dim>
class Element;
template <size_t Dim>
class ElementId;
template <size_t Dim>
class Mesh;
namespace domain::Tags {
template <size_t VolumeDim>
struct Domain;
}  // namespace domain::Tags
/// \endcond

namespace dg {
namespace Actions {
/// \cond
template <typename FilterListLabel, typename TagsToFilterList>
struct Filter;
/// \endcond

/*!
 * \ingroup DiscontinuousGalerkinGroup
 * \brief Applies a filter to the specified tags.
 *
 * If different filters are desired for different sets of variables then
 * multiple `Filter` actions can be inserted into the action list. The
 * `FilterListLabel` distinguishes between them and determines the name in the
 * input file.
 *
 * \snippet LinearOperators/Test_Filtering.cpp action_list_example
 *
 * Uses:
 * - DataBox:
 *   - `Filters::Tags::FilterList<FilterListLabel>`
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
template <typename FilterListLabel, typename... TagsToFilter>
class Filter<FilterListLabel, tmpl::list<TagsToFilter...>> {
 public:
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
    using filter_types =
        tmpl::at<typename Metavariables::factory_creation::factory_classes,
                 Filters::Filter>;
    const auto& filters =
        db::get<::Filters::Tags::FilterList<FilterListLabel>>(box);
    if (filters.empty()) {
      // Short-circuit if there are no filters to apply
      return {Parallel::AlgorithmExecution::Continue, std::nullopt};
    }
    const size_t block_id =
        db::get<domain::Tags::Element<volume_dim>>(box).id().block_id();
    const auto& domain = Parallel::get<domain::Tags::Domain<volume_dim>>(cache);
    const auto& block_groups = domain.block_groups();
    const std::string& block_name = domain.blocks()[block_id].name();

    for (const auto& filter_ptr : filters) {
      bool apply_filter = true;
      // If the filter specifies blocks to filter, enable the filter only in
      // those blocks. If not, apply the filter in all blocks.
      if (filter_ptr->blocks_to_filter().has_value()) {
        apply_filter = alg::any_of(
            filter_ptr->blocks_to_filter().value(),
            [&block_name, &block_groups](const std::string& block_to_filter) {
              return domain::block_is_in_group(block_name, block_to_filter,
                                               block_groups);
            });
      }
      if (not apply_filter) {
        continue;
      }

      // In the case that the tags we are filtering are all the evolved
      // variables can we filter the entire Variables at once to be more
      // efficient.
      constexpr bool filter_all_vars =
          sizeof...(TagsToFilter) ==
              tmpl::size<evolved_vars_tags_list>::value and
          (std::is_same_v<evolved_vars_tags_list,
                          tmpl::list<TagsToFilter...>> or
           tmpl2::flat_all_v<
               tmpl::list_contains_v<evolved_vars_tags_list, TagsToFilter>...>);
      call_with_dynamic_type<void, filter_types>(
          filter_ptr.get(), [&box](const auto* const filter_helper) {
            using filter_type = std::decay_t<decltype(*filter_helper)>;
            if constexpr (filter_all_vars) {
              db::mutate_apply<tmpl::list<evolved_vars_tag>,
                               typename filter_type::argument_tags>(
                  *filter_helper, make_not_null(&box));
            } else {
              db::mutate_apply<tmpl::list<TagsToFilter...>,
                               typename filter_type::argument_tags>(
                  [filter_helper](const gsl::not_null<
                                      typename TagsToFilter::type*>... tensors,
                                  const auto&... additional_args) {
                    // Wrap in a tuple so type deduction works in the filter (a
                    // variadic pack as first arguments doesn't work)
                    const auto tensor_tuple = std::make_tuple(tensors...);
                    (*filter_helper)(tensor_tuple, additional_args...);
                  },
                  make_not_null(&box));
            }
          });
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

/// \ingroup InitializationGroup
/// \brief Initializes a filter list to the DataBox from input-file options.
template <typename FilterListLabel>
struct InitializeFilters : tt::ConformsTo<amr::protocols::Projector> {
 private:
  using filter_list_tag = ::Filters::Tags::FilterList<FilterListLabel>;
  using filter_list_type = typename filter_list_tag::type;

 public:  // Initializer protocol
  using const_global_cache_tags = tmpl::list<>;
  using mutable_global_cache_tags = tmpl::list<>;
  using simple_tags_from_options = tmpl::list<filter_list_tag>;
  using simple_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;
  using return_tags = tmpl::list<filter_list_tag>;
  using argument_tags = tmpl::list<>;

  static void apply(const gsl::not_null<filter_list_type*> /*filters*/) {
    // Nothing to do. Filters are constructed from options.
  }

 public:  // AMR projector protocol
  template <size_t Dim>
  static void apply(const gsl::not_null<filter_list_type*> /*filters*/,
                    const std::pair<Mesh<Dim>, Element<Dim>>&
                    /*old_mesh_and_element*/) {
    // Nothing to do for p-refinement.
  }

  template <typename... ParentTags>
  static void apply(const gsl::not_null<filter_list_type*> filters,
                    const tuples::TaggedTuple<ParentTags...>& parent_items) {
    *filters = deserialize<filter_list_type>(
        serialize(tuples::get<filter_list_tag>(parent_items)).data());
  }

  template <size_t Dim, typename... ChildrenTags>
  static void apply(const gsl::not_null<filter_list_type*> filters,
                    const std::unordered_map<
                        ElementId<Dim>, tuples::TaggedTuple<ChildrenTags...>>&
                        children_items) {
    const auto& first_child_items = children_items.begin()->second;
    *filters = deserialize<filter_list_type>(
        serialize(tuples::get<filter_list_tag>(first_child_items)).data());
  }
};

}  // namespace Actions
}  // namespace dg
