// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "AlgorithmSingleton.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace intrp {
namespace Actions {
template <typename Metavariables, typename InterpolationTargetTag>
struct InitializeInterpolationTarget;
template <typename InterpolationTargetTag>
struct RegisterTargetWithObserver;
}  // namespace Actions
}  // namespace intrp
namespace observers {
namespace Actions {
template <typename InterpolationTargetTag>
struct RegisterSingletonWithObserverWriter;
}  // namespace Actions
}  // namespace observers
/// \endcond

namespace intrp {

/// \brief ParallelComponent representing a set of points to be interpolated
/// to and a function to call upon interpolation to those points.
///
/// Each InterpolationTarget will communicate with the `Interpolator`.
///
/// `InterpolationTargetTag` must contain the following type aliases:
/// - vars_to_interpolate_to_target:
///      A `tmpl::list` of tags describing variables to interpolate.
///      Will be used to construct a `Variables`.
/// - compute_items_on_source:
///      A `tmpl::list` of compute items that uses
///      `Metavariables::interpolator_source_vars` as input and computes the
///      `Variables` defined by `vars_to_interpolate_to_target`.
/// - compute_items_on_target:
///      A `tmpl::list` of compute items that uses
///      `vars_to_interpolate_to_target` as input.
/// - compute_target_points:
///      A `simple_action` of `InterpolationTarget`
///      that computes the target points and sends them to `Interpolators`. It
///      takes a `temporal_id` as an extra argument. `compute_target_points`
///      can (optionally) have an additional function
///```
///   static auto initialize(db::DataBox<DbTags>&&,
///                          const Parallel::GlobalCache<Metavariables>&)
///                          noexcept;
///```
///      that adds arbitrary tags to the `DataBox` when the
///      `InterpolationTarget` is initialized.  If `compute_target_points` has
///      an `initialize` function, it must also have a type alias
///      `initialization_tags` which is a `tmpl::list` of the tags that are
///      added by `initialize`.
/// - post_interpolation_callback:
///      A struct with a type alias `const_global_cache_tags` (listing tags that
///      should be read from option parsing), with a type alias
///      `observation_types` (listing any ObservationTypes that the callback
///      will use in constructing ObserverIds to call
///      observers::ThreadedActions::WriteReductionData), and with a function
///```
///     void apply(const DataBox<DbTags>&,
///                const intrp::GlobalCache<Metavariables>&,
///                const Metavariables::temporal_id&) noexcept;
///```
/// or
///```
///     bool apply(const gsl::not_null<db::DataBox<DbTags>*>,
///                const gsl::not_null<intrp::GlobalCache<Metavariables>*>,
///                const Metavariables::temporal_id&) noexcept;
///```
///      that will be called when interpolation is complete.  `DbTags` includes
///      everything in `vars_to_interpolate_to_target`, plus everything in
///      `compute_items_on_target`.  The second form of the `apply` function
///      should return false only if it calls another `intrp::Action` that still
///      needs the volume data at this temporal_id (such as another iteration of
///      the horizon finder).
///
///      post_interpolation_callback can optionally have a static constexpr
///      double called `fill_invalid_points_with`.  Any points outside the
///      Domain will be filled with this value. If this variable is not defined,
///      then the `apply` function must check for invalid points,
///      and should typically exit with an error message if it finds any.
///
/// `Metavariables` must contain the following type aliases:
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
template <class Metavariables, typename InterpolationTargetTag>
struct InterpolationTarget {
  struct RegistrationHelper {
    template <typename ParallelComponent, typename DbTagsList,
              typename ArrayIndex>
    static std::pair<observers::TypeOfObservation, observers::ObservationId>
    register_info(const db::DataBox<DbTagsList>& /*box*/,
                  const ArrayIndex& /*array_index*/) noexcept {
      observers::ObservationId fake_initial_observation_id{
          0.,
          // Currently this ignores anything in
          // post_interpolation_callback::observation_types except the first
          // element.  This will be changed in an upcoming PR, which
          // will modify RegisterSingletonWithObserverWriter.
          tmpl::front<typename InterpolationTargetTag::
                          post_interpolation_callback::observation_types>{}};
      return {observers::TypeOfObservation::Reduction,
              fake_initial_observation_id};
    }
  };
  using chare_type = ::Parallel::Algorithms::Singleton;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<tmpl::list<
          typename InterpolationTargetTag::compute_target_points,
          typename InterpolationTargetTag::post_interpolation_callback>>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
                         Metavariables, InterpolationTargetTag>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Register,
          tmpl::list<::observers::Actions::RegisterSingletonWithObserverWriter<
                         RegistrationHelper>,
                     Parallel::Actions::TerminatePhase>>>;

  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      typename metavariables::Phase next_phase,
      Parallel::CProxy_GlobalCache<metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<
        InterpolationTarget<metavariables, InterpolationTargetTag>>(local_cache)
        .start_phase(next_phase);
  };
};
}  // namespace intrp
