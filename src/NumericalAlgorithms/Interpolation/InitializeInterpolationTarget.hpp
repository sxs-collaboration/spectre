// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "NumericalAlgorithms/Interpolation/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
// IWYU pragma: no_forward_declare db::DataBox
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
/// \endcond

namespace intrp {

/// Holds Actions for Interpolator and InterpolationTarget.
namespace Actions {

// The purpose of the functions and metafunctions in this
// namespace is to allow InterpolationTarget::compute_target_points
// to omit an initialize function and a initialization_tags
// type alias if it doesn't add anything to the DataBox.
namespace initialize_interpolation_target_detail {

// Sets type to initialization_tags, or
// to empty list if initialization_tags is not defined.
template <typename T, typename = cpp17::void_t<>>
struct initialization_tags {
  using type = tmpl::list<>;
};

template <typename T>
struct initialization_tags<
    T, cpp17::void_t<typename T::compute_target_points::initialization_tags>> {
  using type = typename T::compute_target_points::initialization_tags;
};

// Tests whether T::compute_target_points has a non-empty
// initialization_tags member.
template <typename T>
constexpr bool has_empty_initialization_tags_v =
    tmpl::size<typename initialization_tags<T>::type>::value == 0;

// Calls initialization function only if initialization_tags is defined
// and non-empty; otherwise just moves the box.
template <
    typename InterpolationTargetTag, typename DbTags, typename Metavariables,
    Requires<not has_empty_initialization_tags_v<InterpolationTargetTag>> =
        nullptr>
auto make_initial_box(
    db::DataBox<DbTags>&& box,
    const Parallel::ConstGlobalCache<Metavariables>& cache) noexcept {
  return InterpolationTargetTag::compute_target_points::initialize(
      std::move(box), cache);
}

template <
    typename InterpolationTargetTag, typename DbTags, typename Metavariables,
    Requires<has_empty_initialization_tags_v<InterpolationTargetTag>> = nullptr>
auto make_initial_box(
    db::DataBox<DbTags>&& box,
    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/) noexcept {
  return std::move(box);
}

}  // namespace initialize_interpolation_target_detail

/// \ingroup ActionsGroup
/// \brief Initializes an InterpolationTarget
///
/// Uses: nothing
///
/// DataBox changes:
/// - Adds:
///   - `Tags::IndicesOfFilledInterpPoints<TemporalId>`
///   - `Tags::IndicesOfInvalidInterpPoints<TemporalId>`
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::CompletedTemporalIds<TemporalId>`
///   - `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
/// - Removes: nothing
/// - Modifies: nothing
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
template <typename Metavariables, typename InterpolationTargetTag>
struct InitializeInterpolationTarget {
  using TemporalId = typename Metavariables::temporal_id::type;
  using return_tag_list_initial = tmpl::list<
      Tags::IndicesOfFilledInterpPoints<TemporalId>,
      Tags::IndicesOfInvalidInterpPoints<TemporalId>,
      Tags::TemporalIds<TemporalId>,
      Tags::CompletedTemporalIds<TemporalId>,
      Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>,
      ::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>;
  using return_tag_list =
      tmpl::append<return_tag_list_initial,
                   typename initialize_interpolation_target_detail::
                       initialization_tags<InterpolationTargetTag>::type>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<not tmpl::list_contains_v<
                DbTagsList, Tags::IndicesOfFilledInterpPoints<TemporalId>>> =
                nullptr>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    auto init_box = initialize_interpolation_target_detail::make_initial_box<
        InterpolationTargetTag>(
        db::create_from<db::RemoveTags<>,
                        db::get_items<return_tag_list_initial>>(
            std::move(box),
            std::unordered_map<TemporalId, std::unordered_set<size_t>>{},
            std::unordered_map<TemporalId, std::unordered_set<size_t>>{},
            std::deque<TemporalId>{}, std::deque<TemporalId>{},
            std::unordered_map<TemporalId,
                               Variables<typename InterpolationTargetTag::
                                             vars_to_interpolate_to_target>>{},
            Variables<typename InterpolationTargetTag::
                          vars_to_interpolate_to_target>{}),
        cache);
    // compute_items_on_target will depend on compute items added in
    // make_initial_box, so compute_items_on_target must be added
    // in a separate step.
    return std::make_tuple(
        db::create_from<
            db::RemoveTags<>, db::AddSimpleTags<>,
            db::AddComputeTags<
                typename InterpolationTargetTag::compute_items_on_target>>(
            std::move(init_box)));
  }

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            Requires<tmpl::list_contains_v<
                DbTagsList, Tags::IndicesOfFilledInterpPoints<TemporalId>>> =
                nullptr>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {
    return {std::move(box)};
  }
};

}  // namespace Actions
}  // namespace intrp
