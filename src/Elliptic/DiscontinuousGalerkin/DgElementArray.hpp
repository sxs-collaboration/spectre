// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmArray.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace Frame {
struct Inertial;
}

namespace elliptic {
/*!
 * \brief The parallel component responsible for managing the DG elements that
 * compose the computational domain
 *
 * \note This parallel component is nearly identical to
 * `Evolution/DiscontinuousGalerkin/DgElementArray.hpp` right now, but will
 * likely diverge in the future, for instance to support a multigrid domain.
 *
 * Uses:
 * - Metavariables:
 *   - `domain_creator_tag`
 * - System:
 *   - `volume_dim`
 * - ConstGlobalCache:
 *   - All items required by the actions in `PhaseDepActionList`
 */
template <class Metavariables, class AddOptionsToDataBox,
          class PhaseDepActionList>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementIndex<volume_dim>;

  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags_from_pdal<PhaseDepActionList>;

  using options = tmpl::list<typename Metavariables::domain_creator_tag>;
  using add_options_to_databox = AddOptionsToDataBox;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      const std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
          domain_creator) noexcept {
    auto& dg_element_array = Parallel::get_parallel_component<DgElementArray>(
        *(global_cache.ckLocalBranch()));

    auto domain = domain_creator->create_domain();
    for (const auto& block : domain.blocks()) {
      const auto initial_ref_levs =
          domain_creator->initial_refinement_levels()[block.id()];
      const std::vector<ElementId<volume_dim>> element_ids =
          initial_element_ids(block.id(), initial_ref_levs);
      int which_proc = 0;
      const int number_of_procs = Parallel::number_of_procs();
      for (size_t i = 0; i < element_ids.size(); ++i) {
        dg_element_array(ElementIndex<volume_dim>(element_ids[i]))
            .insert(global_cache,
                    tuples::tagged_tuple_from_typelist<
                        typename add_options_to_databox::simple_tags>(
                        domain_creator->initial_extents(),
                        domain_creator->create_domain()),
                    which_proc);
        which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
      }
    }
    dg_element_array.doneInserting();
  }

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase);
  }
};
}  // namespace elliptic
