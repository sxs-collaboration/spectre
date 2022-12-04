// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Parallel/Algorithms/AlgorithmSingleton.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "ParallelAlgorithms/Interpolation/Actions/InterpolationTargetSendPoints.hpp"
#include "ParallelAlgorithms/Interpolation/Protocols/InterpolationTargetTag.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
namespace intrp {
namespace Actions {
template <typename Metavariables, typename InterpolationTargetTag>
struct InitializeInterpolationTarget;
}  // namespace Actions
}  // namespace intrp
/// \endcond

namespace intrp {

/// \brief ParallelComponent representing a set of points to be interpolated
/// to and a function to call upon interpolation to those points.
///
/// Each InterpolationTarget will communicate with the `Interpolator`.
///
/// `InterpolationTargetTag` must conform to the
/// intrp::protocols::InterpolationTargetTag protocol.
///
/// The metavariables must contain the following type aliases:
/// - interpolator_source_vars:
///      A `tmpl::list` of tags that define a `Variables` sent from all
///      `Element`s to the local `Interpolator`.
/// - interpolation_target_tags:
///      A `tmpl::list` of all `InterpolationTargetTag`s.
/// - temporal_id:
///      The type held by ::intrp::Tags::TemporalIds.
///
/// `Metavariables` must contain the following static constexpr members:
/// - size_t volume_dim:
///      The dimension of the Domain.
///
/// ### Interpolation with time-dependent CoordinateMaps
///
/// Each set of points to be interpolated onto is labeled by a
/// `temporal_id`.  If any step of the interpolation procedure ever
/// uses a time-dependent `CoordinateMap`, then it needs to grab
/// `FunctionOfTime`s from the `GlobalCache`.  Before doing so, it
/// must verify that those `FunctionOfTime`s are up-to-date for the
/// given `temporal_id`.
///
/// Note that once the `FunctionOfTime` has been verified to be
/// up-to-date for a particular `temporal_id` at one step in the
/// interpolation procedure, all subsequent steps of the interpolation
/// procedure for that same `temporal_id` need not worry about
/// re-verifying the `FunctionOfTime`.  Therefore, we need only focus
/// on the first step in the interpolation procedure that needs
/// `FunctionOfTime`s: computing the points on which to interpolate.
///
/// Each `InterpolationTarget` has a function
/// `InterpolationTargetTag::compute_target_points` that returns the
/// points to be interpolated onto, expressed in the frame
/// `InterpolationTargetTag::compute_target_points::frame`.  Then the
/// function `block_logical_coordinates` (and eventually
/// `element_logical_coordinates`) is called to convert those points
/// to the element logical frame to do the interpolation.  If
/// `InterpolationTargetTag::compute_target_points::frame` is
/// different from the grid frame, and if the `CoordinateMap` is
/// time-dependent, then `block_logical_coordinates` grabs
/// `FunctionOfTime`s from the `GlobalCache`.  So therefore any Action
/// calling `block_logical_coordinates` must wait until the
/// `FunctionOfTime`s in the `GlobalCache` are up-to-date for the
/// `temporal_id` being passed into `block_logical_coordinates`.
///
/// Here we describe the logic used in all the Actions that call
/// `block_logical_coordinates`.
///
/// #### Interpolation using the Interpolator ParallelComponent
///
/// Recall that `InterpolationTarget` can be used with the
/// `Interpolator` ParallelComponent (as for the horizon finder), or
/// by having the `Element`s interpolate directly (as for most
/// Observers).  Here we discuss the case when the `Interpolator`
/// is used; the other case is discussed below.
///
/// Ensuring the `FunctionOfTime`s are up-to-date is done via two Tags
/// in the DataBox and a helper Action.  When interpolation is
/// requested for a new `temporal_id` (e.g. by
/// `intrp::Events::Interpolate`), the `temporal_id` is added to
/// `Tags::PendingTemporalIds`, which holds a
/// `std::deque<temporal_id>`, and represents `temporal_ids` that we
/// want to interpolate onto, but for which `FunctionOfTime`s are not
/// necessarily up-to-date.  We also keep another list of
/// `temporal_ids`: `Tags::TemporalIds`, for which `FunctionOfTime`s
/// are guaranteed to be up-to-date.
///
/// The action `Actions::VerifyTemporalIdsAndSendPoints` moves
/// `temporal_id`s from `PendingTemporalIds` to `TemporalIds` as
/// appropriate, and if any `temporal_id`s have been so moved, it
/// generates the `block_logical_coordinates` and sends them to the
/// `Interpolator` `ParallelComponent`.  The logic is illustrated in
/// pseudocode below.  Recall that some InterpolationTargets are
/// sequential, (i.e. you cannot interpolate onto one temporal_id until
/// interpolation on previous ones are done, like the apparent horizon
/// finder), and some are non-sequential (i.e. you can interpolate in
/// any order).
///
/// ```
/// if (map is time-independent or frame is grid frame) {
///   move all PendingTemporalIds to TemporalIds
///   if (sequential) {
///     call block_logical_coords and interpolate for first temporal_id.
///     // when interpolation is complete,
///     // Actions::InterpolationTargetReceiveVars will begin interpolation
///     // on the next temporal_id.
///   } else {
///     call block_logical_coords and interpolate for all temporal_ids.
///   }
///   return;
/// }
/// if (FunctionOfTimes are up-to-date for any PendingTemporalIds) {
///   move up-to-date PendingTemporalIds to TemporalIds
///   if (sequential) {
///     call block_logical_coords and interpolate for first temporal_id.
///     // when interpolation is complete,
///     // Actions::InterpolationTargetReceiveVars will begin interpolation
///     // on the next temporal_id.
///   } else {
///     call block_logical_coords and interpolate for all temporal_ids.
///     if (PendingTemporalIds is non-empty) {
///       call myself (i.e. execute VerifyTemporalIdsAndSendPoints)
///     }
///   }
/// } else {
///   set VerifyTemporalIdsAndSendPoints as a callback for the
///   `domain::Tags::FunctionsOfTime` in MutableGlobalCache, so that
///   once `FunctionsOfTime` are updated, VerifyTemporalIdsAndSendPoints
///   is called.
/// }
/// ```
///
/// Note that VerifyTemporalIdsAndSendPoints always exits in one of three
/// ways:
/// - It has set itself up as a callback, so it will be called again.
/// - It started a sequential interpolation that will automatically
///   start another sequential interpolation when finished.
/// - It started non-sequential interpolations on all
///   `TemporalIds`, and there are no `PendingTemporalIds` left.
///
/// We now describe the logic of the Actions that use
/// VerifyTemporalIdsAndSendPoints.
///
/// ##### Actions::AddTemporalIdsToInterpolationTarget
///
/// `Actions::AddTemporalIdsToInterpolationTarget` is called by
/// `intrp::Events::Interpolate` to trigger interpolation for new
/// `temporal_id`s. Its logic is as follows, in pseudocode:
///
/// ```
/// Add passed-in temporal_ids to PendingTemporalIds
/// if (sequential) {
///   if (TemporalIds is empty and
///       PendingTemporalIds was empty before it was appended above) {
///     call VerifyTemporalIdsAndSendPoints
///   } else {
///     Do nothing, because there is already a sequential interpolation in
///     progress, or there is a callback already waiting.
///   }
/// } else { // not sequential
///   if (TemporalIds is non-empty) {
///     call block_logical_coordinates and begin interpolating for
///     all TemporalIds.
///   }
///   if (PendingTemporalIds is non-empty) {
///     call VerifyTemporalIdsAndSendPoints
///   }
///}
///```
///
/// ##### Actions::InterpolationTargetReceiveVars
///
/// `Actions::InterpolationTargetReceiveVars` is called by the
/// `Interpolator` when it is finished interpolating the current
/// `temporal_id`.  For the sequential case, it needs to start
/// interpolating for the next `temporal_id`.  The logic is, in pseudocode:
///
///```
/// if (sequential) {
///   if (TemporalIds is not empty) {
///     call block_logical_coordinates and interpolate for next temporal_id
///   } else if (PendingTemporalIds is not empty) {
///     call VerifyTemporalIdsAndSendPoints
///   }
/// }
///```
///
/// ##### intrp::callbacks::FindApparentHorizon
///
/// `intrp::callbacks::FindApparentHorizon calls`
/// `block_logical_coordinates` when it needs to start a new iteration
/// of the horizon finder at the same `temporal_id`, so one might
/// think you need to worry about up-to-date `FunctionOfTime`s. But
/// since `intrp::callbacks::FindApparentHorizon` always works on the same
/// `temporal_id` for which the `FunctionOfTime`s have already been
/// verified as up-to-date from the last iteration, no special consideration
/// of `FunctionOfTime`s need be done here.
///
/// #### Interpolation without using the Interpolator ParallelComponent
///
/// This case is easier than the case with the `Interpolator`, because
/// the target points are always time-independent in the frame
/// `compute_target_points::frame`.
///
/// ##### Actions::EnsureFunctionOfTimeUpToDate
///
/// `Actions::EnsureFunctionOfTimeUpToDate` verifies that the
///  `FunctionOfTime`s are up-to-date at the `DgElementArray`s current
///  time.
///
/// ###### Current logic:
///
/// > `Actions::EnsureFunctionOfTimeUpToDate` is placed in `DgElementArray`s
/// >  PDAL before any use of interpolation.
///
/// ##### Actions::InterpolationTargetSendTimeIndepPointsToElements
///
/// `Actions::InterpolationTargetSendTimeIndepPointsToElements`
/// is invoked on `InterpolationTarget` during the Registration PDAL,
/// to send time-independent point information to `Element`s.
///
/// ###### Current logic:
///
/// > Send the result of `compute_target_points` to all `Element`s.
///
/// Note that this may need to be revisited because every `Element` has
/// a copy of every target point, which may use a lot of memory.  An
/// alternative is for each Element to invoke an Action on each
/// `InterpolationTarget` (presumably from an `Event`) at each time,
/// and then the InterpolationTarget invokes another Action to send points
/// to only those `Elements` that contain the points; this alternative
/// uses less memory but much more communication. Another alternative would
/// be to place the points in the MutableGlobalCache (so that there is one
/// copy per core, rather than one copy per Element), or even in the
/// GlobalCache (one copy per node) since the points need be computed only
/// once.
///
template <class Metavariables, typename InterpolationTargetTag>
struct InterpolationTarget {
  using interpolation_target_tag = InterpolationTargetTag;
  static_assert(
      tt::assert_conforms_to_v<interpolation_target_tag,
                               intrp::protocols::InterpolationTargetTag>);
  static std::string name() {
    return pretty_type::name<InterpolationTargetTag>();
  }
  using chare_type = ::Parallel::Algorithms::Singleton;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<tmpl::list<
          typename InterpolationTargetTag::compute_target_points,
          typename InterpolationTargetTag::post_interpolation_callback>>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
                         Metavariables, InterpolationTargetTag>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Register,
          tmpl::list<
              tmpl::conditional_t<
                  InterpolationTargetTag::compute_target_points::is_sequential::
                      value,
                  tmpl::list<>,
                  tmpl::list<
                      Actions::InterpolationTargetSendTimeIndepPointsToElements<
                          InterpolationTargetTag>>>,
              Parallel::Actions::TerminatePhase>>>;

  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<
        InterpolationTarget<metavariables, InterpolationTargetTag>>(local_cache)
        .start_phase(next_phase);
  };
};
}  // namespace intrp
