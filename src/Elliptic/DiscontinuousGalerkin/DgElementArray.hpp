// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "AlgorithmArray.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeElement.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "IO/Observer/TypeOfObservation.hpp"
#include "Parallel/ConstGlobalCache.hpp"
#include "Parallel/Info.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/ParallelComponentHelpers.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace observers {
namespace Actions {
template <observers::TypeOfObservation TypeOfObservation>
struct RegisterWithObservers;
}  // namespace Actions
}  // namespace observers
/// \endcond

namespace Elliptic {

/*!
 * \brief The parallel component responsible for managing the DG elements that
 * compose the computational domain
 *
 * This parallel component will perform the following phases:
 *
 * - `Phase::Initialize`: Create the domain and execute
 * `Elliptic::dg::Actions::InitializeElement` on all elements.
 * - `Phase::RegisterWithObservers` (optional)
 * - `Phase::Solve`: Execute the actions in `ActionList` on all elements. Repeat
 * until an action terminates the algorithm.
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
 *   - All items required by the actions in `ActionList`
 */
template <class Metavariables, class ActionList>
struct DgElementArray {
  static constexpr size_t volume_dim = Metavariables::system::volume_dim;

  using chare_type = Parallel::Algorithms::Array;
  using metavariables = Metavariables;
  using action_list = ActionList;
  using array_index = ElementIndex<volume_dim>;

  using const_global_cache_tag_list =
      Parallel::get_const_global_cache_tags<action_list>;

  using initial_databox = db::compute_databox_type<
      typename Elliptic::dg::Actions::InitializeElement<
          volume_dim>::template return_tag_list<Metavariables>>;

  using options = tmpl::list<typename Metavariables::domain_creator_tag>;

  static void initialize(
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache,
      std::unique_ptr<DomainCreator<volume_dim, Frame::Inertial>>
          domain_creator) noexcept;

  static void execute_next_phase(
      const typename Metavariables::Phase next_phase,
      Parallel::CProxy_ConstGlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *(global_cache.ckLocalBranch());
    auto& dg_element_array =
        Parallel::get_parallel_component<DgElementArray>(local_cache);
    switch (next_phase) {
      case Metavariables::Phase::Solve:
        dg_element_array.perform_algorithm();
        break;
      default:
        try_register_with_observers(next_phase, global_cache);
        break;
    }
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
      // We currently use an observation_id with a fake time when registering
      // observers but in the future when we start doing load balancing and
      // elements migrate around the system they will need to register and
      // unregister themselves at specific times.
      const observers::ObservationId observation_id_with_fake_time(
          0., typename Metavariables::element_observation_type{});
      Parallel::simple_action<observers::Actions::RegisterWithObservers<
          observers::TypeOfObservation::ReductionAndVolume>>(
          Parallel::get_parallel_component<DgElementArray>(local_cache),
          observation_id_with_fake_time);
    }
  }
};

template <class Metavariables, class ActionList>
void DgElementArray<Metavariables, ActionList>::initialize(
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
          .insert(global_cache, which_proc);
      which_proc = which_proc + 1 == number_of_procs ? 0 : which_proc + 1;
    }
  }
  dg_element_array.doneInserting();

  dg_element_array.template simple_action<
      Elliptic::dg::Actions::InitializeElement<volume_dim>>(
      std::make_tuple(domain_creator->initial_extents(), std::move(domain)));
}
}  // namespace Elliptic
