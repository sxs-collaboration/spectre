// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/DomainCreator.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Invoke.hpp"
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
/// - vars_to_interpolate_to_target: a `tmpl::list` of tags describing
///                                  variables to interpolate.  Will be used
///                                  to construct a `Variables`.
/// - compute_items_on_source:       a `tmpl::list` of compute items that uses
///                                  `Metavariables::interpolator_source_vars`
///                                  as input and computes the `Variables`
///                                  defined by `vars_to_interpolate_to_target`.
/// - compute_items_on_target:       a `tmpl::list` of compute items that uses
///                                 `vars_to_interpolate_to_target` as input.
/// - compute_target_points:         a `simple_action` of `InterpolationTarget`
///                                  that computes the target points and
///                                  sends them to `Interpolators`.
///                                  It takes a `temporal_id` as an extra
///                                  argument.
/// - post_interpolation_callback:   a struct with a function
///```
///       static void apply(const DataBox<DbTags>&,
///                         const intrp::ConstGlobalCache<Metavariables>&,
///                         const Metavariables::temporal_id&) noexcept;
///```
///                                  that will be called when interpolation
///                                  is complete. `DbTags` includes everything
///                                  in `vars_to_interpolate_to_target`
///                                  and `compute_items_on_target`.
///
/// `Metavariables` must contain the following type aliases:
/// - interpolator_source_vars:   a `tmpl::list` of tags that define a
///                               `Variables` sent from all `Element`s
///                               to the local `Interpolator`.
/// - interpolation_target_tags:  a `tmpl::list` of all
///                               `InterpolationTargetTag`s.
/// - temporal_id:                the type held by ::intrp::Tags::TemporalIds.
template <class Metavariables, typename InterpolationTargetTag,
          size_t VolumeDim, typename Frame>
struct InterpolationTarget {
  using chare_type = ::Parallel::Algorithms::Singleton;
  using metavariables = Metavariables;
  using action_list = tmpl::list<>;
  using initial_databox = db::compute_databox_type<
      typename Actions::InitializeInterpolationTarget<InterpolationTargetTag>::
          template return_tag_list<Metavariables, VolumeDim, Frame>>;
  using options = tmpl::list<::OptionTags::DomainCreator<VolumeDim, Frame>>;
  using const_global_cache_tag_list = tmpl::list<>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache,
      std::unique_ptr<DomainCreator<VolumeDim, Frame>> domain_creator) noexcept;
  static void execute_next_phase(
      typename metavariables::Phase /*next_phase*/,
      const Parallel::CProxy_ConstGlobalCache<metavariables>&
      /*global_cache*/) noexcept {}
};

template <class Metavariables, typename InterpolationTargetTag>
template <size_t VolumeDim, typename Frame>
void InterpolationTarget<Metavariables, InterpolationTargetTag>::initialize(
    Parallel::CProxy_ConstGlobalCache<metavariables>& global_cache,
    std::unique_ptr<DomainCreator<VolumeDim, Frame>> domain_creator) noexcept {
  auto& my_proxy = Parallel::get_parallel_component<InterpolationTarget>(
      *(global_cache.ckLocalBranch()));
  Parallel::simple_action<
      Actions::InitializeInterpolationTarget<InterpolationTargetTag>>(
      my_proxy, domain_creator->create_domain());
}

}  // namespace intrp
