// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// The parallel component and actions in this file keep track of all element
/// IDs during AMR and update the array sections for the multigrid hierarchy.

#pragma once

#include <algorithm>
#include <charm++.h>
#include <cstddef>
#include <optional>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "Domain/Structure/ElementId.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Parallel/Protocols/ElementRegistrar.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/LinearSolver/Multigrid/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace LinearSolver::multigrid {

namespace Tags {
// All element IDs grouped by multigrid level are stored in this tag on the
// singleton component below. The element IDs are registered and deregistered
// during AMR.
template <size_t Dim>
struct AllElementIds : db::SimpleTag {
  using type = std::unordered_map<size_t, std::unordered_set<ElementId<Dim>>>;
};
}  // namespace Tags

namespace detail {

// The projector for the sections does nothing, as the sections are updated by a
// simple action broadcast below.
template <typename Metavariables>
struct ProjectMultigridSections : tt::ConformsTo<amr::protocols::Projector> {
 private:
  using element_array = typename Metavariables::amr::element_array;

 public:
  using argument_tags = tmpl::list<>;
  using return_tags =
      tmpl::list<Parallel::Tags::Section<element_array, Tags::MultigridLevel>,
                 Parallel::Tags::Section<element_array, Tags::IsFinestGrid>>;

  template <typename AmrData>
  static void apply(
      const gsl::not_null<std::optional<
          Parallel::Section<element_array, Tags::MultigridLevel>>*>
      /*multigrid_level_section*/,
      const gsl::not_null<
          std::optional<Parallel::Section<element_array, Tags::IsFinestGrid>>*>
      /*finest_grid_section*/,
      const AmrData& /*amr_data*/) {}
};

template <size_t Dim>
struct InitializeElementsRegistration {
  using simple_tags = tmpl::list<Tags::AllElementIds<Dim>>;
  using compute_tags = tmpl::list<>;

  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename Metavariables, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& /*box*/,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

struct RegisterOrDeregisterElement {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, typename ArrayIndex, size_t Dim>
  static void apply(db::DataBox<DbTagsList>& box,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ElementId<Dim>& element_id,
                    const bool register_or_deregister) {
    db::mutate<Tags::AllElementIds<Dim>>(
        [&element_id, register_or_deregister](const auto all_element_ids) {
          auto& element_ids = (*all_element_ids)[element_id.grid_index()];
          if (register_or_deregister) {
            element_ids.insert(element_id);
          } else {
            element_ids.erase(element_id);
          }
        },
        make_not_null(&box));
  }
};

struct UpdateSectionsOnElement {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, size_t Dim>
  static void apply(
      db::DataBox<DbTagsList>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id,
      Parallel::Section<ParallelComponent, Tags::MultigridLevel>
          multigrid_level_section,
      std::optional<Parallel::Section<ParallelComponent, Tags::IsFinestGrid>>
          finest_grid_section) {
    if (multigrid_level_section.id() != element_id.grid_index()) {
      // Discard broadcast to elements that are not part of the section. This
      // happens because we broadcast to all elements, not just to the section.
      // Broadcasting to the section fails with a segfault for some reason.
      return;
    }
    db::mutate<Parallel::Tags::Section<ParallelComponent, Tags::MultigridLevel>,
               Parallel::Tags::Section<ParallelComponent, Tags::IsFinestGrid>>(
        [&multigrid_level_section, &finest_grid_section](
            const auto stored_multigrid_level_section,
            const auto stored_finest_grid_section) {
          *stored_multigrid_level_section = std::move(multigrid_level_section);
          *stored_finest_grid_section = std::move(finest_grid_section);
        },
        make_not_null(&box));
  }
};

template <size_t Dim, typename ElementArray, typename OptionsGroup>
struct UpdateSections {
  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    const auto& all_element_ids = db::get<Tags::AllElementIds<Dim>>(box);
    auto& element_array = Parallel::get_parallel_component<ElementArray>(cache);
    for (const auto& [multigrid_level, element_ids] : all_element_ids) {
      Parallel::printf("%s level %zu has %zu elements.\n",
                       pretty_type::name<OptionsGroup>(), multigrid_level,
                       element_ids.size());
      std::vector<CkArrayIndex> array_indices(element_ids.size());
      std::transform(
          element_ids.begin(), element_ids.end(), array_indices.begin(),
          [](const ElementId<Dim>& local_element_id) {
            return Parallel::ArrayIndex<ElementId<Dim>>(local_element_id);
          });
      using MultigridLevelSection =
          Parallel::Section<ElementArray, Tags::MultigridLevel>;
      const MultigridLevelSection multigrid_level_section{
          multigrid_level, MultigridLevelSection::cproxy_section::ckNew(
                               element_array.ckGetArrayID(),
                               array_indices.data(), array_indices.size())};
      using FinestGridSection =
          Parallel::Section<ElementArray, Tags::IsFinestGrid>;
      const std::optional<FinestGridSection> finest_grid_section =
          multigrid_level == 0
              ? std::make_optional(FinestGridSection{
                    true, FinestGridSection::cproxy_section::ckNew(
                              element_array.ckGetArrayID(),
                              array_indices.data(), array_indices.size())})
              : std::nullopt;
      // Send new sections to all elements. Broadcasting to the section fails
      // with a segfault for some reason.
      Parallel::simple_action<UpdateSectionsOnElement>(
          element_array, multigrid_level_section, finest_grid_section);
    }
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

template <typename Metavariables, typename OptionsGroup>
struct ElementsRegistrationComponent {
  using chare_type = Parallel::Algorithms::Singleton;
  using const_global_cache_tags = tmpl::list<>;
  using metavariables = Metavariables;
  static constexpr size_t Dim = metavariables::volume_dim;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<Parallel::Phase::Initialization,
                             tmpl::list<InitializeElementsRegistration<Dim>>>,
      Parallel::PhaseActions<
          // Update sections in the `CheckDomain` phase, which runs after the
          // AMR phase.
          Parallel::Phase::CheckDomain,
          tmpl::list<UpdateSections<
              Dim, typename metavariables::amr::element_array, OptionsGroup>>>>;
  using simple_tags_from_options = Parallel::get_simple_tags_from_options<
      Parallel::get_initialization_actions_list<phase_dependent_action_list>>;

  static void execute_next_phase(
      const Parallel::Phase next_phase,
      Parallel::CProxy_GlobalCache<Metavariables>& global_cache) {
    auto& local_cache = *Parallel::local_branch(global_cache);
    Parallel::get_parallel_component<ElementsRegistrationComponent>(local_cache)
        .start_phase(next_phase);
  }
};

template <size_t Dim, typename OptionsGroup>
struct RegisterElement : tt::ConformsTo<Parallel::protocols::ElementRegistrar> {
 public:  // ElementRegistrar protocol
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void perform_registration(const db::DataBox<DbTagList>& /*box*/,
                                   Parallel::GlobalCache<Metavariables>& cache,
                                   const ElementId<Dim>& element_id) {
    Parallel::simple_action<RegisterOrDeregisterElement>(
        Parallel::get_parallel_component<
            ElementsRegistrationComponent<Metavariables, OptionsGroup>>(cache),
        element_id, true);
  }

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void perform_deregistration(
      const db::DataBox<DbTagList>& /*box*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id) {
    Parallel::simple_action<RegisterOrDeregisterElement>(
        Parallel::get_parallel_component<
            ElementsRegistrationComponent<Metavariables, OptionsGroup>>(cache),
        element_id, false);
  }

 public:  // Iterable action
  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    perform_registration<ParallelComponent>(box, cache, element_id);
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

}  // namespace detail
}  // namespace LinearSolver::multigrid
