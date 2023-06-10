// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Amr/NewNeighborIds.hpp"
#include "Domain/Amr/Tags/Flags.hpp"
#include "Domain/Amr/Tags/NeighborFlags.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateParent.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"

namespace amr {
/// \cond
template <class Metavariables>
struct Component;
/// \endcond
}  // namespace amr

namespace amr::Actions {
/// \brief Adjusts the domain given the refinement criteria
///
/// \details
/// - Checks if an Element wants to split in any dimension; if yes, determines
///   the ElementId%s of the new children Element%s and calls
///   amr::Actions::CreateChild on the amr::Component, then exits the action.
/// - Checks if an Element wants to join in any dimension; if yes, either calls
///   amr::Actions::CreateParent on the amr::Component if this is the child
///   Element that should create the parent Element or does nothing, and then
///   exits the action.
/// - Checks if an Element wants to increase or decrease its resolution, if yes,
///   mutates the Mesh
/// - Updates the Neighbors of the Element
/// - Resets amr::Tags::Flag%s to amr::Flag::Undefined
/// - Resets amr::Tags::NeighborFlags to an empty map
///
/// \warning At the moment, this action only updates the Mesh for a p-refined
/// Element.  It does not update any data related to evolution or elliptic
/// solves.
struct AdjustDomain {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Metavariables::volume_dim>& element_id) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    const auto& my_amr_flags = db::get<amr::Tags::Flags<volume_dim>>(box);
    auto& element_array =
        Parallel::get_parallel_component<ParallelComponent>(cache);

    // Check for h-refinement
    if (alg::any_of(my_amr_flags,
                    [](amr::Flag flag) { return flag == amr::Flag::Split; })) {
      using ::operator<<;
      ASSERT(alg::count(my_amr_flags, amr::Flag::Join) == 0,
             "Element " << element_id
                        << " cannot both split and join, but had AMR flags "
                        << my_amr_flags << "\n");
      auto children_ids = amr::ids_of_children(element_id, my_amr_flags);
      auto& amr_component =
          Parallel::get_parallel_component<amr::Component<Metavariables>>(
              cache);
      Parallel::simple_action<CreateChild>(amr_component, element_array,
                                           element_id, children_ids, 0_st);

    } else if (alg::any_of(my_amr_flags, [](amr::Flag flag) {
                 return flag == amr::Flag::Join;
               })) {
      // Only one element should create the new parent
      if (amr::is_child_that_creates_parent(element_id, my_amr_flags)) {
        auto parent_id = amr::id_of_parent(element_id, my_amr_flags);
        const auto& element = db::get<::domain::Tags::Element<volume_dim>>(box);
        auto ids_to_join = amr::ids_of_joining_neighbors(element, my_amr_flags);
        auto& amr_component =
            Parallel::get_parallel_component<amr::Component<Metavariables>>(
                cache);
        Parallel::simple_action<CreateParent>(amr_component, element_array,
                                              std::move(parent_id), element_id,
                                              std::move(ids_to_join));
      }

    } else {
      // Check for p-refinement
      if (alg::any_of(my_amr_flags, [](amr::Flag flag) {
            return (flag == amr::Flag::IncreaseResolution or
                    flag == amr::Flag::DecreaseResolution);
          })) {
        db::mutate<::domain::Tags::Mesh<volume_dim>>(
            [&my_amr_flags](const gsl::not_null<Mesh<volume_dim>*> mesh) {
              *mesh = amr::projectors::mesh(*mesh, my_amr_flags);
            },
            make_not_null(&box));
      }

      // Need to reset AMR flags and determine new neighbors
      db::mutate<::domain::Tags::Element<volume_dim>,
                 amr::Tags::Flags<volume_dim>,
                 amr::Tags::NeighborFlags<volume_dim>>(
          [&element_id](
              const gsl::not_null<Element<volume_dim>*> element,
              const gsl::not_null<std::array<amr::Flag, volume_dim>*> amr_flags,
              const gsl::not_null<std::unordered_map<
                  ElementId<volume_dim>, std::array<amr::Flag, volume_dim>>*>
                  amr_flags_of_neighbors) {
            auto new_neighbors = element->neighbors();
            for (auto& [direction, neighbors] : new_neighbors) {
              neighbors.set_ids_to(amr::new_neighbor_ids(
                  element_id, direction, neighbors, *amr_flags_of_neighbors));
            }
            *element = Element<volume_dim>(element_id, new_neighbors);
            amr_flags_of_neighbors->clear();
            for (size_t d = 0; d < volume_dim; ++d) {
              (*amr_flags)[d] = amr::Flag::Undefined;
            }
          },
          make_not_null(&box));
    }

    // In the near future, add the capability of updating all data needed for
    // an evolution or elliptic system
  }
};
}  // namespace amr::Actions
