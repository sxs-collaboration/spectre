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
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Domain/Tags/NeighborMesh.hpp"
#include "IO/Logging/Tags.hpp"
#include "IO/Logging/Verbosity.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Invoke.hpp"
#include "Parallel/Phase.hpp"
#include "Parallel/Printf/Printf.hpp"
#include "Parallel/Tags/DistributedObjectTags.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateChild.hpp"
#include "ParallelAlgorithms/Amr/Actions/CreateParent.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "ParallelAlgorithms/Amr/Tags.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

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
/// - Resets amr::Tags::NeighborInfo to an empty map
/// - Mutates all return_tags of Metavariables::amr::projectors
struct AdjustDomain {
  template <typename ParallelComponent, typename DbTagList,
            typename Metavariables>
  static void apply(db::DataBox<DbTagList>& box,
                    Parallel::GlobalCache<Metavariables>& cache,
                    const ElementId<Metavariables::volume_dim>& element_id) {
    constexpr size_t volume_dim = Metavariables::volume_dim;
    using amr_projectors = typename Metavariables::amr::projectors;
    static_assert(
        tmpl::all<
            amr_projectors,
            tt::assert_conforms_to<tmpl::_1, amr::protocols::Projector>>::value,
        "All AMR projectors must conform to 'amr::protocols::Projector'.");

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
        ::domain::Tags::NeighborMesh<volume_dim>, amr::Tags::Info<volume_dim>,
        amr::Tags::NeighborInfo<volume_dim>>;
    using mutated_tags =
        tmpl::append<distributed_object_tags, tags_mutated_by_this_action,
                     typename detail::GetMutatedTags<amr_projectors>::type>;
    using mutable_tags =
        typename db::DataBox<DbTagList>::mutable_item_creation_tags;
    using mutable_tags_not_mutated =
        tmpl::list_difference<mutable_tags, mutated_tags>;
    static_assert(std::is_same_v<mutable_tags_not_mutated, tmpl::list<>>,
                  "All mutable tags in the DataBox must be explicitly mutated "
                  "by an amr::projector.  Default initialized objects can use "
                  "amr::projector::DefaultInitialize.");

    const auto& my_amr_info = db::get<amr::Tags::Info<volume_dim>>(box);
    const auto& my_amr_flags = my_amr_info.flags;
    auto& element_array =
        Parallel::get_parallel_component<ParallelComponent>(cache);
    const auto& phase_bookmarks =
        Parallel::local(element_array[element_id])->phase_bookmarks();
    const auto& verbosity =
        db::get<logging::Tags::Verbosity<amr::OptionTags::AmrGroup>>(box);

    if (alg::all_of(my_amr_flags, [](amr::Flag flag) {
          return flag == amr::Flag::Undefined;
        })) {
      // AMR flags are undefined. This state can be reached when the AMR
      // component broadcasts the `AdjustDomain` simple action to the entire
      // element array, then some of those elements run the simple action and
      // create new child elements, and then the broadcast also arrives at these
      // new elements for some reason (seems like a Charm++ bug). The AMR flags
      // will be undefined in this case, so we just ignore the broadcast and
      // return early here.
      return;
    } else if (alg::any_of(my_amr_flags, [](amr::Flag flag) {
                 return flag == amr::Flag::Split;
               })) {
      // h-refinement
      using ::operator<<;
      ASSERT(alg::count(my_amr_flags, amr::Flag::Join) == 0,
             "Element " << element_id
                        << " cannot both split and join, but had AMR flags "
                        << my_amr_flags << "\n");
      auto children_ids = amr::ids_of_children(element_id, my_amr_flags);
      auto& amr_component =
          Parallel::get_parallel_component<amr::Component<Metavariables>>(
              cache);
      if (verbosity >= Verbosity::Debug) {
        Parallel::printf("Splitting element %s into %zu: %s\n", element_id,
                         children_ids.size(), children_ids);
      }
      Parallel::simple_action<CreateChild>(amr_component, element_array,
                                           element_id, children_ids, 0_st,
                                           phase_bookmarks);

    } else if (alg::any_of(my_amr_flags, [](amr::Flag flag) {
                 return flag == amr::Flag::Join;
               })) {
      // h-coarsening
      // Only one element should create the new parent
      if (amr::is_child_that_creates_parent(element_id, my_amr_flags)) {
        auto parent_id = amr::id_of_parent(element_id, my_amr_flags);
        const auto& element = db::get<::domain::Tags::Element<volume_dim>>(box);
        auto ids_to_join = amr::ids_of_joining_neighbors(element, my_amr_flags);
        auto& amr_component =
            Parallel::get_parallel_component<amr::Component<Metavariables>>(
                cache);
        if (verbosity >= Verbosity::Debug) {
          Parallel::printf("Joining %zu elements: %s -> %s\n",
                           ids_to_join.size(), ids_to_join, parent_id);
        }
        Parallel::simple_action<CreateParent>(
            amr_component, element_array, std::move(parent_id), element_id,
            std::move(ids_to_join), phase_bookmarks);
      }

    } else {
      // Neither h-refinement nor h-coarsening. This element will remain.
      const auto old_mesh_and_element =
          std::make_pair(db::get<::domain::Tags::Mesh<volume_dim>>(box),
                         db::get<::domain::Tags::Element<volume_dim>>(box));
      const auto& old_mesh = old_mesh_and_element.first;

      // Determine new neighbors and update the Element
      {  // avoid shadowing when mutating flags below
        using NeighborMeshType =
            DirectionalIdMap<volume_dim, ::Mesh<volume_dim>>;
        const auto& amr_info_of_neighbors =
            db::get<amr::Tags::NeighborInfo<volume_dim>>(box);
        db::mutate<::domain::Tags::Element<volume_dim>,
                   ::domain::Tags::NeighborMesh<volume_dim>>(
            [&element_id, &amr_info_of_neighbors](
                const gsl::not_null<Element<volume_dim>*> element,
                const gsl::not_null<NeighborMeshType*> neighbor_meshes) {
              auto new_neighbors = element->neighbors();
              neighbor_meshes->clear();
              for (auto& [direction, neighbors] : new_neighbors) {
                const auto new_neighbor_ids_and_meshes = amr::new_neighbor_ids(
                    element_id, direction, neighbors, amr_info_of_neighbors);
                std::unordered_set<ElementId<volume_dim>> new_neighbor_ids;
                for (const auto& [id, mesh] : new_neighbor_ids_and_meshes) {
                  neighbor_meshes->insert({{direction, id}, mesh});
                  new_neighbor_ids.insert(id);
                }
                neighbors.set_ids_to(new_neighbor_ids);
              }
              *element =
                  Element<volume_dim>(element_id, std::move(new_neighbors));
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

        if (verbosity >= Verbosity::Debug) {
          Parallel::printf(
              "Increasing order of element %s: %s -> %s\n", element_id,
              old_mesh.extents(),
              db::get<::domain::Tags::Mesh<volume_dim>>(box).extents());
        }
      }

      // Run the projectors on all elements, even if they did no h-refinement.
      // This allows projectors to update mutable items that depend upon the
      // neighbors of the element.
      tmpl::for_each<amr_projectors>(
          [&box, &old_mesh_and_element](auto projector_v) {
            using projector = typename decltype(projector_v)::type;
            try {
              db::mutate_apply<projector>(make_not_null(&box),
                                          old_mesh_and_element);
            } catch (std::exception& e) {
              ERROR("Error in AMR projector '"
                    << pretty_type::get_name<projector>() << "':\n"
                    << e.what());
            }
          });

      // Reset the AMR flags
      db::mutate<amr::Tags::Info<volume_dim>,
                 amr::Tags::NeighborInfo<volume_dim>>(
          [](const gsl::not_null<amr::Info<volume_dim>*> amr_info,
             const gsl::not_null<std::unordered_map<ElementId<volume_dim>,
                                                    amr::Info<volume_dim>>*>
                 amr_info_of_neighbors) {
            amr_info_of_neighbors->clear();
            for (size_t d = 0; d < volume_dim; ++d) {
              amr_info->flags[d] = amr::Flag::Undefined;
            }
          },
          make_not_null(&box));
    }
  }
};
}  // namespace amr::Actions
