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
#include "Utilities/TypeTraits/CreateGetTypeAliasOrDefault.hpp"
#include "Utilities/TypeTraits/CreateIsCallable.hpp"

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

// The purpose of the metafunctions in this namespace is to allow
// InterpolationTarget::compute_target_points to omit an initialize
// function and a compute_tags and simple_tags type alias if it
// doesn't add anything to the DataBox.
namespace initialize_interpolation_target_detail {

CREATE_GET_TYPE_ALIAS_OR_DEFAULT(compute_tags)
CREATE_GET_TYPE_ALIAS_OR_DEFAULT(simple_tags)
CREATE_IS_CALLABLE(initialize)
CREATE_IS_CALLABLE_V(initialize)

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

  using simple_tags = tmpl::append<
      return_tag_list_initial,
      initialize_interpolation_target_detail::get_simple_tags_or_default_t<
          typename InterpolationTargetTag::compute_target_points,
          tmpl::list<>>>;
  using compute_tags = tmpl::append<
      initialize_interpolation_target_detail::get_compute_tags_or_default_t<
          typename InterpolationTargetTag::compute_target_points, tmpl::list<>>,
      typename InterpolationTargetTag::compute_items_on_target>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent>
  static auto apply(db::DataBox<DbTagsList>& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& cache,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    if constexpr (
        initialize_interpolation_target_detail::is_initialize_callable_v<
            typename InterpolationTargetTag::compute_target_points,
            const gsl::not_null<db::DataBox<DbTagsList>*>,
            const Parallel::GlobalCache<Metavariables>&>) {
      InterpolationTargetTag::compute_target_points::initialize(
          make_not_null(&box), cache);
    }
    return std::make_tuple(std::move(box));
  }
};

}  // namespace Actions
}  // namespace intrp
