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
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Tags/DistributedObjectTags.hpp"
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

namespace detail {
template <typename ListOfProjectors>
struct GetMutatedTags;

template <typename... Projectors>
struct GetMutatedTags<tmpl::list<Projectors...>> {
  using type = tmpl::remove_duplicates<
      tmpl::flatten<tmpl::append<typename Projectors::return_tags...>>>;
};
}  // namespace detail

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
/// - Mutates all return_tags of Metavariables::amr::projectors
struct AdjustDomain {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Metavariables::volume_dim>& element_id) {
    constexpr size_t volume_dim = Metavariables::volume_dim;

    // To prevent bugs when new mutable items are added to a DataBox, we require
    // that all mutable_item_creation_tags of box are either:
    // - mutated by one of the projectors in Metavariables::amr::projectors
    // - in the list of distributed_object_tags
    // - or in the list of tags mutated by this action
    using distributed_object_tags =
        typename ::Parallel::Tags::distributed_object_tags<
            Metavariables, ElementId<volume_dim>>;
    using tags_mutated_by_this_action = tmpl::list<
        ::domain::Tags::Element<volume_dim>, ::domain::Tags::Mesh<volume_dim>,
        amr::Tags::Flags<volume_dim>, amr::Tags::NeighborFlags<volume_dim>>;
    using mutated_tags =
        tmpl::append<distributed_object_tags, tags_mutated_by_this_action,
                     typename detail::GetMutatedTags<
                         typename Metavariables::amr::projectors>::type>;
    using mutable_tags =
        typename db::DataBox<DbTagList>::mutable_item_creation_tags;
    using mutable_tags_not_mutated =
        tmpl::list_difference<mutable_tags, mutated_tags>;
    static_assert(std::is_same_v<mutable_tags_not_mutated, tmpl::list<>>,
                  "All mutable tags in the DataBox must be explicitly mutated "
                  "by an amr::projector.  Default initialized objects can use "
                  "amr::projector::DefaultInitialize.");

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
      const auto old_mesh_and_element =
          std::make_pair(db::get<::domain::Tags::Mesh<volume_dim>>(box),
                         db::get<::domain::Tags::Element<volume_dim>>(box));
      const auto& old_mesh = old_mesh_and_element.first;

      // Determine new neighbors and update the Element
      {  // avoid shadowing when mutating flags below
        const auto& amr_flags_of_neighbors =
            db::get<amr::Tags::NeighborFlags<volume_dim>>(box);
        db::mutate<::domain::Tags::Element<volume_dim>>(
            [&element_id, &amr_flags_of_neighbors](
                const gsl::not_null<Element<volume_dim>*> element) {
              auto new_neighbors = element->neighbors();
              for (auto& [direction, neighbors] : new_neighbors) {
                neighbors.set_ids_to(amr::new_neighbor_ids(
                    element_id, direction, neighbors, amr_flags_of_neighbors));
              }
              *element = Element<volume_dim>(element_id, new_neighbors);
            },
            make_not_null(&box));
      }

      // Check for p-refinement
      if (alg::any_of(my_amr_flags, [](amr::Flag flag) {
            return (flag == amr::Flag::IncreaseResolution or
                    flag == amr::Flag::DecreaseResolution);
          })) {
        db::mutate<::domain::Tags::Mesh<volume_dim>>(
            [&old_mesh,
             &my_amr_flags](const gsl::not_null<Mesh<volume_dim>*> mesh) {
              *mesh = amr::projectors::mesh(old_mesh, my_amr_flags);
            },
            make_not_null(&box));
      }

      // Run the projectors on all elements, even if they did no h-refinement.
      // This allows projectors to update mutable items that depend upon the
      // neighbors of the element.
      tmpl::for_each<typename Metavariables::amr::projectors>(
          [&box, &old_mesh_and_element](auto projector_v) {
            using projector = typename decltype(projector_v)::type;
            db::mutate_apply<projector>(make_not_null(&box),
                                        old_mesh_and_element);
          });

      // Reset the AMR flags
      db::mutate<amr::Tags::Flags<volume_dim>,
                 amr::Tags::NeighborFlags<volume_dim>>(
          [](const gsl::not_null<std::array<amr::Flag, volume_dim>*> amr_flags,
             const gsl::not_null<std::unordered_map<
                 ElementId<volume_dim>, std::array<amr::Flag, volume_dim>>*>
                 amr_flags_of_neighbors) {
            amr_flags_of_neighbors->clear();
            for (size_t d = 0; d < volume_dim; ++d) {
              (*amr_flags)[d] = amr::Flag::Undefined;
            }
          },
          make_not_null(&box));
    }
  }
};
}  // namespace amr::Actions
