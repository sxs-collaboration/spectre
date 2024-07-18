// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <memory>

#include "Parallel/Algorithms/AlgorithmNodegroupDeclarations.hpp"
#include "Parallel/ArrayCollection/CreateElementCollection.hpp"
#include "Parallel/ArrayCollection/DgElementArrayMember.hpp"
#include "Parallel/ArrayCollection/DgElementArrayMemberBase.hpp"
#include "Parallel/ArrayCollection/Tags/ElementCollection.hpp"
#include "Parallel/ArrayCollection/TransformPdalForNodegroup.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "ParallelAlgorithms/Actions/TerminatePhase.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Parallel {
/*!
 * \brief A collection of DG elements on a node.
 *
 * The `PhaseDepActionList` is the PDAL that was used for the array
 * approach. Some actions will require updating to support nodegroups if they
 * haven't already been.
 */
template <size_t Dim, class Metavariables, class PhaseDepActionList>
struct DgElementCollection {
  /// \brief The `DgElementCollection` is currently a nodegroup.
  ///
  /// It should be possible to generalize this to work as a group too.
  using chare_type = Parallel::Algorithms::Nodegroup;
  /// \brief The metavariables
  using metavariables = Metavariables;
  /// \brief The simple tags necessary from option parsing.
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<PhaseDepActionList>>;
  /// \brief The type of the `Parallel::DgElementArrayMember` with all
  /// template parameters.
  using dg_element_array_member =
      Parallel::DgElementArrayMember<Dim, Metavariables, PhaseDepActionList,
                                     simple_tags_from_options>;
  /// \brief The tag `Parallel::Tags::ElementCollection` of this
  /// `DgElementCollection`
  using element_collection_tag =
      Parallel::Tags::ElementCollection<Dim, Metavariables, PhaseDepActionList,
                                        simple_tags_from_options>;

  /// \brief The phase dependent action lists.
  ///
  /// These are computed using
  /// `Parallel::TransformPhaseDependentActionListForNodegroup` from the
  /// `PhaseDepActionList` template parameter.
  using phase_dependent_action_list = tmpl::append<
      tmpl::list<Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<Actions::CreateElementCollection<Dim, Metavariables,
                                                      PhaseDepActionList,
                                                      simple_tags_from_options>,
                     Parallel::Actions::TerminatePhase>>>,
      TransformPhaseDependentActionListForNodegroup<PhaseDepActionList>>;

  /// @{
  /// \brief The tags for the global cache.
  ///
  /// These are computed from the `PhaseDepActionList` template parameter.
  using const_global_cache_tags = tmpl::remove_duplicates<
      typename Parallel::detail::get_const_global_cache_tags_from_pdal<
          PhaseDepActionList>::type>;
  using mutable_global_cache_tags =
      typename Parallel::detail::get_mutable_global_cache_tags_from_pdal<
          PhaseDepActionList>::type;
  /// @}

  /// \brief Starts the next phase of execution.
  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    Parallel::printf("%s\n", next_phase);
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<DgElementCollection>(local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace Parallel
