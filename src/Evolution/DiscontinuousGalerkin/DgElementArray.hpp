// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <memory>
#include <vector>

#include "AlgorithmArray.hpp"
#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Creators/DomainCreator.hpp"        // IWYU pragma: keep
#include "Domain/ElementId.hpp"                     // IWYU pragma: keep
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "ErrorHandling/Error.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Frame {
struct Inertial;
}  // namespace Frame
/// \endcond

template <typename PhaseDepAction>
struct get_action_list_from_phase_dep_action {
  using type = typename PhaseDepAction::action_list;
};

template <class Metavariables, class PhaseDepActionList,
          class AddOptionsToDataBox>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using phase_dependent_action_list = PhaseDepActionList;
  using array_index = ElementIndex<volume_dim>;

  using const_global_cache_tag_list = tmpl::push_back<
      Parallel::get_const_global_cache_tags_from_pdal<PhaseDepActionList>,
      OptionTags::Domain<volume_dim, Frame::Inertial>>;

  using options = tmpl::flatten<tmpl::list<
      typename Metavariables::domain_creator_tag, OptionTags::InitialTime,
      OptionTags::InitialTimeStep,
      tmpl::conditional_t<tmpl::list_contains_v<const_global_cache_tag_list,
                                                OptionTags::StepController>,
                          OptionTags::InitialSlabSize, tmpl::list<>>>>;

  using add_options_to_databox = AddOptionsToDataBox;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
          domain_creator,
      double initial_time, double initial_dt) noexcept;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
          domain_creator,
      double initial_time, double initial_dt,
      double initial_slab_size) noexcept;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) noexcept {
    auto& local_cache = *(global_cache.ckLocalBranch());
    Parallel::get_parallel_component<DgElementArray>(local_cache)
        .start_phase(next_phase);
  }
};

template <class Metavariables, class PhaseDepActionList,
          class AddOptionsToDataBox>
void DgElementArray<Metavariables, PhaseDepActionList, AddOptionsToDataBox>::
    initialize(Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
               std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
                   domain_creator,
               const double initial_time, const double initial_dt) noexcept {
  initialize(global_cache, std::move(domain_creator), initial_time, initial_dt,
             std::abs(initial_dt));
}

template <class Metavariables, class PhaseDepActionList,
          class AddOptionsToDataBox>
void DgElementArray<Metavariables, PhaseDepActionList, AddOptionsToDataBox>::
    initialize(Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
               const std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
                   domain_creator,
               const double initial_time, const double initial_dt,
               const double initial_slab_size) noexcept {
  auto& cache = *global_cache.ckLocalBranch();
  auto& dg_element_array =
      Parallel::get_parallel_component<DgElementArray>(cache);

  if (not Metavariables::local_time_stepping and
      std::abs(initial_dt) != initial_slab_size) {
    ERROR("Step and slab size must agree for global time-stepping.");
  }

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
                      domain_creator->create_domain(), initial_time, initial_dt,
                      initial_slab_size),
                  which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
  }
  dg_element_array.doneInserting();
}
