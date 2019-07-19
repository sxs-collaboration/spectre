// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/DomainCreator.hpp"
#include "Domain/Domain.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace intrp {
namespace Actions {
template <typename InterpolationTargetTag>
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
///                          const Parallel::ConstGlobalCache<Metavariables>&)
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
///                const intrp::ConstGlobalCache<Metavariables>&,
///                const Metavariables::temporal_id&) noexcept;
///```
/// or
///```
///     bool apply(const gsl::not_null<db::DataBox<DbTags>*>,
///                const gsl::not_null<intrp::ConstGlobalCache<Metavariables>*>,
///                const Metavariables::temporal_id&) noexcept;
///```
///      that will be called when interpolation is complete.  `DbTags` includes
///      everything in `vars_to_interpolate_to_target`, plus everything in
///      `compute_items_on_target`.  The second form of the `apply` function
///      should return false only if it calls another `intrp::Action` that still
///      needs the volume data at this temporal_id (such as another iteration of
///      the horizon finder).
///
/// `Metavariables` must contain the following type aliases:
/// - interpolator_source_vars:
///      A `tmpl::list` of tags that define a `Variables` sent from all
///      `Element`s to the local `Interpolator`.
/// - interpolation_target_tags:
///      A `tmpl::list` of all `InterpolationTargetTag`s.
/// - temporal_id:
///      The type held by ::intrp::Tags::TemporalIds.
/// - domain_frame:
///      The `::Frame` of the Domain.
///
/// `Metavariables` must contain the following static constexpr members:
/// - size_t domain_dim:
///      The dimension of the Domain.
template <class Metavariables, typename InterpolationTargetTag>
struct InterpolationTarget {
  using chare_type = ::Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox =
      db::compute_databox_type<typename Actions::InitializeInterpolationTarget<
          InterpolationTargetTag>::template return_tag_list<Metavariables>>;
  using options = tmpl::list<::OptionTags::DomainCreator<
      Metavariables::domain_dim, typename Metavariables::domain_frame>>;
  using const_global_cache_tag_list = Parallel::get_const_global_cache_tags<
      tmpl::list<typename InterpolationTargetTag::compute_target_points,
                 typename InterpolationTargetTag::post_interpolation_callback>>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache,
      std::unique_ptr<DomainCreator<Metavariables::domain_dim,
                                    typename Metavariables::domain_frame>>
          domain_creator) noexcept;
  static void execute_next_phase(
      typename metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache) noexcept {
    try_register_with_observers(next_phase, global_cache);
  }

 private:
  template <typename PhaseType,
            Requires<not observers::has_register_with_observer_v<PhaseType>> =
                nullptr>
  static void try_register_with_observers(
      const PhaseType /*next_phase*/,
      Parallel::CProxy_ConstGlobalCache<
          Metavariables>& /*global_cache*/) noexcept {}
  template <
      typename PhaseType,
      Requires<observers::has_register_with_observer_v<PhaseType>> = nullptr>
  static void try_register_with_observers(
      const PhaseType next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    if (next_phase == Metavariables::Phase::RegisterWithObserver) {
      auto& local_cache = *(global_cache.ckLocalBranch());
      tmpl::for_each<
          typename InterpolationTargetTag::post_interpolation_callback::
              observation_types>([&local_cache](auto type_v) noexcept {
        using type = typename decltype(type_v)::type;
        // The 'time' value of the observation_id doesn't matter here and
        // is currently not used.
        // In the future when we do load balancing, we will need to register
        // and unregister at specific times.
        const observers::ObservationId observation_id(0., type{});
        Parallel::simple_action<
            observers::Actions::RegisterSingletonWithObserverWriter>(
            Parallel::get_parallel_component<
                InterpolationTarget<Metavariables, InterpolationTargetTag>>(
                local_cache),
            observation_id);
      });
    }
  }
};

template <class Metavariables, typename InterpolationTargetTag>
void InterpolationTarget<Metavariables, InterpolationTargetTag>::initialize(
    Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache,
    std::unique_ptr<DomainCreator<Metavariables::domain_dim,
                                  typename Metavariables::domain_frame>>
        domain_creator) noexcept {
  auto& my_proxy = Parallel::get_parallel_component<InterpolationTarget>(
      *(global_cache.ckLocalBranch()));
  Parallel::simple_action<
      Actions::InitializeInterpolationTarget<InterpolationTargetTag>>(
      my_proxy, domain_creator->create_domain());
}

}  // namespace intrp
