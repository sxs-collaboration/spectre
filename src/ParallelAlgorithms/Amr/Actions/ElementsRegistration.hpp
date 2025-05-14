// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// The actions in this file keep track of all element IDs during AMR and update
/// the array sections that represent the grid hierarchy.

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
#include "Parallel/ElementRegistration.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Local.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Parallel/Protocols/ElementRegistrar.hpp"
#include "Parallel/Section.hpp"
#include "Parallel/Tags/Section.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace amr {
template <typename Metavariables>
struct Component;
}  // namespace amr

namespace amr::Actions {

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

struct RegisterElement : tt::ConformsTo<Parallel::protocols::ElementRegistrar> {
 public:  // ElementRegistrar protocol
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, size_t Dim>
  static void perform_registration(const db::DataBox<DbTagList>& /*box*/,
                                   Parallel::GlobalCache<Metavariables>& cache,
                                   const ElementId<Dim>& element_id) {
    Parallel::simple_action<RegisterOrDeregisterElement>(
        Parallel::get_parallel_component<::amr::Component<Metavariables>>(
            cache),
        element_id, true);
  }

  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables, size_t Dim>
  static void perform_deregistration(
      const db::DataBox<DbTagList>& /*box*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ElementId<Dim>& element_id) {
    Parallel::simple_action<RegisterOrDeregisterElement>(
        Parallel::get_parallel_component<::amr::Component<Metavariables>>(
            cache),
        element_id, false);
  }

 public:  // Iterable action
  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent, size_t Dim>
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

struct UpdateSectionsOnElement {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, size_t Dim>
  static void apply(
      db::DataBox<DbTagsList>& box,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ElementId<Dim>& element_id, const size_t grid_index,
      Parallel::Section<ParallelComponent, Tags::GridIndex> grid_index_section,
      std::optional<Parallel::Section<ParallelComponent, Tags::IsFinestGrid>>
          finest_grid_section) {
    if (grid_index != element_id.grid_index()) {
      // Discard broadcast to elements that are not part of the section. This
      // happens because we broadcast to all elements, not just to the section.
      // Broadcasting to the section fails with a segfault for some reason.
      return;
    }
    db::mutate<Parallel::Tags::Section<ParallelComponent, Tags::GridIndex>,
               Parallel::Tags::Section<ParallelComponent, Tags::IsFinestGrid>>(
        [&grid_index_section, &finest_grid_section](
            const auto stored_grid_index_section,
            const auto stored_finest_grid_section) {
          // Only update the grid index section if we don't have one already
          // because the elements in the old grid didn't change. This avoids a
          // bug(?) with Charm++ where section reductions don't work with the
          // new section. It's possible that this issue is with (not) updating
          // the section cookie, but it's not clear how to do that because a
          // multicast message is needed for that and we can't even do a
          // broadcast to the section without a segfault.
          if (not stored_grid_index_section->has_value()) {
            *stored_grid_index_section = std::move(grid_index_section);
          }
          *stored_finest_grid_section = std::move(finest_grid_section);
        },
        make_not_null(&box));
  }
};

struct DestroyGrid {
  template <typename ParallelComponent, typename DbTagsList,
            typename Metavariables, size_t Dim>
  static void apply(db::DataBox<DbTagsList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Dim>& element_id, const size_t grid_index) {
    if (grid_index == element_id.grid_index()) {
      // Destroy the element
      Parallel::deregister_element<ParallelComponent>(box, cache, element_id);
      auto& array_proxy =
          Parallel::get_parallel_component<ParallelComponent>(cache);
      array_proxy[element_id].ckDestroy();
      return;
    }
    // Unregister the parent ID if it was destroyed
    const auto& parent_id = db::get<amr::Tags::ParentId<Dim>>(box);
    if (parent_id.has_value() and parent_id->grid_index() == grid_index) {
      db::mutate<amr::Tags::ParentId<Dim>, amr::Tags::ParentMesh<Dim>>(
          [](const gsl::not_null<std::optional<ElementId<Dim>>*>
                 stored_parent_id,
             const gsl::not_null<std::optional<Mesh<Dim>>*>
                 stored_parent_mesh) {
            *stored_parent_id = std::nullopt;
            *stored_parent_mesh = std::nullopt;
          },
          make_not_null(&box));
    }
  }
};

template <typename ElementArray>
struct UpdateSections {
  using const_global_cache_tags = tmpl::list<amr::Tags::MaxCoarseLevels>;

  template <typename DbTagList, typename... InboxTags, typename Metavariables,
            typename ArrayIndex, typename ActionList,
            typename ParallelComponent>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagList>& box,
      const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& cache,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    static constexpr size_t Dim = Metavariables::volume_dim;
    const auto& all_element_ids = db::get<Tags::AllElementIds<Dim>>(box);
    auto& element_array = Parallel::get_parallel_component<ElementArray>(cache);
    const size_t finest_grid_index = std::prev(all_element_ids.end())->first;
    const std::optional<size_t> max_coarse_levels =
        db::get<amr::Tags::MaxCoarseLevels>(box);
    for (const auto& [grid_index, element_ids] : all_element_ids) {
      if (max_coarse_levels.has_value() and
          finest_grid_index - grid_index > max_coarse_levels.value()) {
        // Delete grids that are coarser than the maximum allowed level
        Parallel::simple_action<DestroyGrid>(element_array, grid_index);
        continue;
      }
      std::vector<CkArrayIndex> array_indices(element_ids.size());
      std::transform(
          element_ids.begin(), element_ids.end(), array_indices.begin(),
          [](const ElementId<Dim>& local_element_id) {
            return Parallel::ArrayIndex<ElementId<Dim>>(local_element_id);
          });
      using GridIndexSection = Parallel::Section<ElementArray, Tags::GridIndex>;
      GridIndexSection grid_index_section{
          grid_index, GridIndexSection::cproxy_section::ckNew(
                          element_array.ckGetArrayID(), array_indices.data(),
                          array_indices.size())};
      using FinestGridSection =
          Parallel::Section<ElementArray, Tags::IsFinestGrid>;
      const std::optional<FinestGridSection> finest_grid_section =
          grid_index == finest_grid_index
              ? std::make_optional(FinestGridSection{
                    true, FinestGridSection::cproxy_section::ckNew(
                              element_array.ckGetArrayID(),
                              array_indices.data(), array_indices.size())})
              : std::nullopt;
      // Send new sections to all elements. Broadcasting to the section fails
      // with a segfault for some reason.
      Parallel::simple_action<UpdateSectionsOnElement>(
          element_array, grid_index, grid_index_section, finest_grid_section);
    }
    return {Parallel::AlgorithmExecution::Pause, std::nullopt};
  }
};

}  // namespace amr::Actions
