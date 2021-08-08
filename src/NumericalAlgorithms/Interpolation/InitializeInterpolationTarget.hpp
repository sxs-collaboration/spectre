// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <tuple>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
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
class GlobalCache;
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
template <typename T, typename = std::void_t<>>
struct initialization_tags {
  using type = tmpl::list<>;
};

template <typename T>
struct initialization_tags<
    T, std::void_t<typename T::compute_target_points::initialization_tags>> {
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
void mutate_initial_box(
    const gsl::not_null<db::DataBox<DbTags>*> box,
    const Parallel::GlobalCache<Metavariables>& cache) noexcept {
  InterpolationTargetTag::compute_target_points::initialize(box, cache);
}

template <
    typename InterpolationTargetTag, typename DbTags, typename Metavariables,
    Requires<has_empty_initialization_tags_v<InterpolationTargetTag>> = nullptr>
void mutate_initial_box(
    const gsl::not_null<db::DataBox<DbTags>*> /*box*/,
    const Parallel::GlobalCache<Metavariables>& /*cache*/) noexcept {}

template <typename InterpolationTargetTag, typename = std::void_t<>,
          bool = not has_empty_initialization_tags_v<InterpolationTargetTag>>
struct compute_target_points_tags {
  using simple_tags = tmpl::list<>;
  using compute_tags = tmpl::list<>;
};

template <typename InterpolationTargetTag>
struct compute_target_points_tags<
    InterpolationTargetTag,
    std::void_t<typename InterpolationTargetTag::compute_target_points>, true> {
  using simple_tags =
      typename InterpolationTargetTag::compute_target_points::simple_tags;
  using compute_tags =
      typename InterpolationTargetTag::compute_target_points::compute_tags;
};

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
///   - `Tags::PendingTemporalIds<TemporalId>`
///   - `Tags::TemporalIds<TemporalId>`
///   - `Tags::CompletedTemporalIds<TemporalId>`
///   - `Tags::InterpolatedVars<InterpolationTargetTag,TemporalId>`
///   - `::Tags::Variables<typename
///                   InterpolationTargetTag::vars_to_interpolate_to_target>`
/// - Removes: nothing
/// - Modifies: nothing
///
/// For requirements on InterpolationTargetTag, see InterpolationTarget
///
/// \note This action relies on the `SetupDataBox` aggregated initialization
/// mechanism, so `Actions::SetupDataBox` must be present in the
/// `Initialization` phase action list prior to this action.
template <typename Metavariables, typename InterpolationTargetTag>
struct InitializeInterpolationTarget {
  using TemporalId = typename InterpolationTargetTag::temporal_id::type;
  using return_tag_list_initial = tmpl::list<
      Tags::IndicesOfFilledInterpPoints<TemporalId>,
      Tags::IndicesOfInvalidInterpPoints<TemporalId>,
      Tags::PendingTemporalIds<TemporalId>, Tags::TemporalIds<TemporalId>,
      Tags::CompletedTemporalIds<TemporalId>,
      Tags::InterpolatedVars<InterpolationTargetTag, TemporalId>,
      ::Tags::Variables<
          typename InterpolationTargetTag::vars_to_interpolate_to_target>>;
  using return_tag_list =
      tmpl::append<return_tag_list_initial,
                   typename initialize_interpolation_target_detail::
                       initialization_tags<InterpolationTargetTag>::type>;

  using simple_tags = tmpl::append<
      return_tag_list_initial,
      typename initialize_interpolation_target_detail::
          compute_target_points_tags<InterpolationTargetTag>::simple_tags>;
  using compute_tags = tmpl::append<
      typename initialize_interpolation_target_detail::
          compute_target_points_tags<InterpolationTargetTag>::compute_tags,
      typename InterpolationTargetTag::compute_items_on_target>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    initialize_interpolation_target_detail::mutate_initial_box<
        InterpolationTargetTag>(make_not_null(&box), cache);
    return std::make_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace intrp
