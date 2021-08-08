// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBox.hpp"
#include "NumericalAlgorithms/Interpolation/InitializeInterpolator.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Actions/TerminatePhase.hpp"
#include "Parallel/Algorithms/AlgorithmGroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

namespace intrp {

namespace detail {
template <typename InterpolationTarget>
struct get_interpolation_target_tag {
  using type = typename InterpolationTarget::interpolation_target_tag;
};
template <typename InterpolationTargetTag>
struct get_temporal_id {
  using type = typename InterpolationTargetTag::temporal_id;
};
}  // namespace detail

/// \brief ParallelComponent responsible for collecting data from
/// `Element`s and interpolating it onto `InterpolationTarget`s.
///
/// For requirements on Metavariables, see InterpolationTarget
template <class Metavariables>
struct Interpolator {
  using chare_type = Parallel::Algorithms::Group;
  using metavariables = Metavariables;
  using all_interpolation_target_tags = tmpl::transform<
      tmpl::filter<typename Metavariables::component_list,
                   tt::is_a_lambda<intrp::InterpolationTarget, tmpl::_1>>,
      detail::get_interpolation_target_tag<tmpl::_1>>;
  using all_temporal_ids = tmpl::remove_duplicates<tmpl::transform<
      all_interpolation_target_tags, detail::get_temporal_id<tmpl::_1>>>;
  using phase_dependent_action_list = tmpl::list<Parallel::PhaseActions<
      typename metavariables::Phase, metavariables::Phase::Initialization,
      tmpl::list<
          ::Actions::SetupDataBox,
          Actions::InitializeInterpolator<
              tmpl::transform<all_temporal_ids,
                              tmpl::bind<Tags::VolumeVarsInfo,
                                         tmpl::pin<Metavariables>, tmpl::_1>>,
              Tags::InterpolatedVarsHolders<Metavariables>>,
          Parallel::Actions::TerminatePhase>>>;
  using initialization_tags = Parallel::get_initialization_tags<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;
  static void execute_next_phase(
      typename Metavariables::Phase next_phase,
      const Parallel::CProxy_GlobalCache<Metavariables>&
          global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<Interpolator>(local_cache)
        .start_phase(next_phase);
  };
};
}  // namespace intrp
