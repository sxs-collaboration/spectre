// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmArray.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/InitializeElement.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

template <class Metavariables, class ActionList>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using const_global_cache_tag_list = tmpl::flatten<
      tmpl::list<CacheTags::FinalTime, CacheTags::TimeStepper,
                 typename Metavariables::dg_element_array_add_to_cache>>;
  using metavariables = Metavariables;
  using action_list = ActionList;
  using array_index = ElementIndex<volume_dim>;

  using initial_databox = db::compute_databox_type<
      typename dg::Actions::InitializeElement<volume_dim>::
          template return_tag_list<typename Metavariables::system>>;

  using options = tmpl::list<typename Metavariables::domain_creator_tag,
                             OptionTags::InitialTime, OptionTags::DeltaT>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
          domain_creator,
      double initial_time, double initial_dt) noexcept;

  static void execute_next_global_actions(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    if (next_phase == Metavariables::Phase::Evolve) {
      Parallel::get_parallel_component<DgElementArray>(local_cache)
          .perform_algorithm();
    }
  }
};

template <class Metavariables, class ActionList>
void DgElementArray<Metavariables, ActionList>::initialize(
    Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
    const std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
        domain_creator,
    const double initial_time, const double initial_dt) noexcept {
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
          .insert(global_cache, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
  }
  dg_element_array.doneInserting();

  // Set time and DeltaT allowing for integration backwards in time
  const bool time_reversed = initial_dt < 0;
  const Slab slab =
      time_reversed ? Slab::with_duration_to_end(initial_time, -initial_dt)
                    : Slab::with_duration_from_start(initial_time, initial_dt);
  const Time time = time_reversed ? slab.end() : slab.start();
  const TimeDelta dt = time_reversed ? -slab.duration() : slab.duration();

  Parallel::simple_action<dg::Actions::InitializeElement<volume_dim>>(
      dg_element_array, domain_creator->initial_extents(), std::move(domain),
      time, dt);
}
