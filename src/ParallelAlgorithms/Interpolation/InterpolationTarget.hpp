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
namespace intrp::Actions {
template <typename Metavariables, typename InterpolationTargetTag>
struct InitializeInterpolationTarget;
}  // namespace intrp::Actions
/// \endcond

namespace intrp {

/// \brief ParallelComponent representing a set of points to be interpolated
/// to and a function to call upon interpolation to those points.
///
/// `InterpolationTargetTag` must conform to the
/// intrp::protocols::InterpolationTargetTag protocol.
///
/// The metavariables must contain the following type alias:
/// - interpolation_target_tags:
///      A `tmpl::list` of all `InterpolationTargetTag`s.
///
/// `Metavariables` must contain the following static constexpr members:
/// - size_t volume_dim:
///      The dimension of the Domain.
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
  static constexpr bool checkpoint_data = true;
  using const_global_cache_tags =
      Parallel::get_const_global_cache_tags_from_actions<
          tmpl::flatten<tmpl::list<
              typename InterpolationTargetTag::compute_target_points,
              typename InterpolationTargetTag::post_interpolation_callbacks>>>;
  using metavariables = Metavariables;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<intrp::Actions::InitializeInterpolationTarget<
                         Metavariables, InterpolationTargetTag>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Register,
          tmpl::list<Actions::InterpolationTargetSendTimeIndepPointsToElements<
                         InterpolationTargetTag>,
                     Parallel::Actions::TerminatePhase>>,
      Parallel::PhaseActions<
          Parallel::Phase::Restart,
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
