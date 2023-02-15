// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <map>
#include <memory>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg::Tags {
/*!
 * \brief The inbox tag for boundary correction communication and DG-subcell
 * ghost zone cells.
 *
 * The stored data consists of the following (in argument order of the tuple):
 *
 * 1. the mesh of the ghost cell data we received. This allows eliding
 *    projection when all neighboring elements are doing DG.
 * 2. the mesh of the neighboring element's face (not the mortar mesh!)
 * 3. the variables at the ghost zone cells for finite difference/volume
 *    reconstruction
 * 4. the data on the mortar needed for computing the boundary corrections (e.g.
 *    fluxes, characteristic speeds, conserved variables)
 * 5. the TimeStepId beyond which the boundary terms are no longer valid, when
 *    using local time stepping.
 *
 * The TimeStepId is the neighboring element's next time step. When using local
 * time stepping, the neighbor's boundary data is valid up until this time,
 * which may include multiple local time steps. By receiving and storing the
 * neighbor time step, the local element knows whether or not it should remove
 * boundary data and expect new data to be sent from the neighbor.
 *
 * The ghost cell data (second element of the tuple) will be valid whenever a
 * DG-subcell scheme is being used. Whenever a DG-subcell scheme is being used,
 * elements using DG and not FD/FV always send both the ghost cells and boundary
 * correction data together. Elements using FD/FV send the ghost cells first
 * followed by the boundary correction data once the element has received all
 * neighbor ghost cell data. Note that the second send/receive only modifies the
 * flux and the TimeStepId used for the flux validity range.
 *
 * When only a DG scheme (not a DG-subcell scheme) is used the ghost cell data
 * will never be valid.
 *
 * In the DG-subcell scheme this tag is used both for communicating the ghost
 * cell data needed for the FD/FV reconstruction step and the data needed for
 * the boundary corrections.
 * - For an element using DG, both ghost cells and boundary corrections are
 *   sent using a single communication. After receiving all neighbor
 *   boundary corrections the element can finish computing the time step.
 *   The ghost cell data from neighbors is unused.
 * - For an element using FD/FV, first the ghost cells are sent. Once all
 *   neighboring ghost cell data is received, reconstruction is done and the
 *   boundary terms are computed and sent to the neighbors. After receiving all
 *   neighbor boundary corrections the element can finish computing the time
 *   step.
 * - Whether or not an extra communication is needed when an element switches
 *   from DG to FD/FV depends on how exactly the decision to switch is
 *   implemented. If the volume terms are integrated and verified to be
 *   valid before a DG element sends ghost cell and boundary data then no
 *   additional communication is needed when switching from DG to FD/FV. In this
 *   case a second check of the data that includes the boundary correction needs
 *   to be done. If the second check determines a switch from DG to FD/FV is
 *   needed, we can continue to use the DG fluxes since the evolution in the
 *   small was valid, thus avoiding an additional communication. However, to
 *   fully guarantee physical realizability a second communication or evolving
 *   the neighboring ghost cells needs to be done. We have not yet decided how
 *   to deal with the possible need for an additional communication since it
 *   also depends on whether or not we decide to couple to Voronoi instead of
 *   just Neumann neighbors.
 * - The data for the inbox tags is erased after the boundary correction is
 *   complete and the solution has been verified to be valid at the new time
 *   step. The ghost cells could be invalidated immediately after
 *   reconstruction, thus using the ghost cell data after reconstruction is
 *   complete is considered undefined behavior. That is, we make no guarantee as
 *   to whether or not it will work.
 * - The reason for minimizing the number of communications rather than having a
 *   more uniform implementation between DG and FD/FV is that it is the number
 *   of communications that adds the most overhead, not the size of each
 *   communication. Thus, one large communication is cheaper than several small
 *   communications.
 */
template <size_t Dim>
struct BoundaryCorrectionAndGhostCellsInbox {
  using stored_type =
      std::tuple<Mesh<Dim>, Mesh<Dim - 1>, std::optional<DataVector>,
                 std::optional<DataVector>, ::TimeStepId, int>;

 public:
  using temporal_id = TimeStepId;
  using type = std::map<
      TimeStepId,
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>, stored_type,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>>;

  template <typename Inbox, typename ReceiveDataType>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                const temporal_id& time_step_id,
                                ReceiveDataType&& data) {
    auto& current_inbox = (*inbox)[time_step_id];
    auto& [volume_mesh_of_ghost_cell_data, face_mesh, ghost_cell_data,
           boundary_data, boundary_data_validity_range, boundary_tci_status] =
        data.second;
    (void)ghost_cell_data;

    if (auto it = current_inbox.find(data.first); it != current_inbox.end()) {
      auto& [current_volume_mesh_of_ghost_cell_data, current_face_mesh,
             current_ghost_cell_data, current_boundary_data,
             current_boundary_data_validity_range, current_tci_status] =
          it->second;
      (void)current_volume_mesh_of_ghost_cell_data;  // Need to use when
                                                     // optimizing subcell
      // We have already received some data at this time. Receiving data twice
      // at the same time should only occur when receiving fluxes after having
      // previously received ghost cells. We sanity check that the data we
      // already have is the ghost cells and that we have not yet received flux
      // data.
      //
      // This is used if a 2-send implementation is used (which we don't right
      // now!). We generally find that the number of communications is more
      // important than the size of each communication, and so a single
      // communication per time/sub step is preferred.
      ASSERT(current_ghost_cell_data.has_value(),
             "Have not yet received ghost cells at time step "
                 << time_step_id
                 << " but the inbox entry already exists. This is a bug in the "
                    "ordering of the actions.");
      ASSERT(not current_boundary_data.has_value(),
             "The fluxes have already been received at time step "
                 << time_step_id
                 << ". They are either being received for a second time, there "
                    "is a bug in the ordering of the actions (though a "
                    "different ASSERT should've caught that), or the incorrect "
                    "temporal ID is being sent.");

      ASSERT(current_face_mesh == face_mesh,
             "The mesh being received for the fluxes is different than the "
             "mesh received for the ghost cells. Mesh for fluxes: "
                 << face_mesh << " mesh for ghost cells " << current_face_mesh);
      ASSERT(current_volume_mesh_of_ghost_cell_data ==
                 volume_mesh_of_ghost_cell_data,
             "The mesh being received for the ghost cell data is different "
             "than the mesh received previously. Mesh for received when we got "
             "fluxes: "
                 << volume_mesh_of_ghost_cell_data
                 << " mesh received when we got ghost cells "
                 << current_volume_mesh_of_ghost_cell_data);

      // We always move here since we take ownership of the data and moves
      // implicitly decay to copies
      current_boundary_data = std::move(boundary_data);
      current_boundary_data_validity_range = boundary_data_validity_range;
      current_tci_status = boundary_tci_status;
    } else {
      // We have not received ghost cells or fluxes at this time.
      if (not current_inbox.insert(std::forward<ReceiveDataType>(data))
                  .second) {
        ERROR("Failed to insert data to receive at instance '"
              << time_step_id
              << "' with tag 'BoundaryCorrectionAndGhostCellsInbox'.\n");
      }
    }
  }
};

/*!
 * \brief The inbox tag for boundary correction communication and DG-subcell
 * ghost zone cells using a `BoundaryMessage` object
 *
 * To see what is stored within a `BoundaryMessage`, see its documentation.
 *
 * This inbox tag is very similar to `BoundaryCorrectionAndGhostCellsInbox` in
 * that it stores subcell/DG data sent from neighboring elements. To see exactly
 * when data is stored and how it's used, see the docs for
 * `BoundaryCorrectionAndGhostCellsInbox`. This inbox tag is different than
 * `BoundaryCorrectionAndGhostCellsInbox` in that it only takes a pointer to a
 * `BoundaryMessage` as an argument to `insert_into_inbox` and stores a
 * `std::unique_ptr<BoundaryMessage>` inside the inbox.
 *
 * This inbox tag is meant to be used to avoid unnecessary copies between
 * elements on the same node which share a block of memory. If two elements
 * aren't on the same node, then a copy/send is done regardless.
 *
 * \warning The `boundary_message` argument to `insert_into_inbox()` will be
 * invalid after the function is called because a `std::unique_ptr` now controls
 * the memory. Calling a method on the `boundary_message` pointer after the
 * `insert_into_inbox()` function is called can result in undefined behaviour.
 */
template <size_t Dim>
struct BoundaryMessageInbox {
  using stored_type = std::unique_ptr<BoundaryMessage<Dim>>;

 public:
  using temporal_id = TimeStepId;
  using type = std::map<
      TimeStepId,
      FixedHashMap<maximum_number_of_neighbors(Dim),
                   std::pair<Direction<Dim>, ElementId<Dim>>, stored_type,
                   boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>>;
  using message_type = BoundaryMessage<Dim>;

  template <typename Inbox>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                BoundaryMessage<Dim>* boundary_message) {
    const auto& time_step_id = boundary_message->current_time_step_id;
    auto& current_inbox = (*inbox)[time_step_id];

    const auto key = std::pair{boundary_message->neighbor_direction,
                               boundary_message->element_id};

    if (auto it = current_inbox.find(key); it != current_inbox.end()) {
      auto& current_boundary_data = it->second;
      // We have already received some data at this time. Receiving data twice
      // at the same time should only occur when receiving fluxes after having
      // previously received ghost cells. We sanity check that the data we
      // already have is the ghost cells and that we have not yet received flux
      // data.
      //
      // This is used if a 2-send implementation is used (which we don't right
      // now!). We generally find that the number of communications is more
      // important than the size of each communication, and so a single
      // communication per time/sub step is preferred.
      ASSERT(current_boundary_data->subcell_ghost_data != nullptr,
             "Have not yet received ghost cells at time step "
                 << time_step_id
                 << " but the inbox entry already exists. This is a bug in the "
                    "ordering of the actions.");
      ASSERT(current_boundary_data->dg_flux_data == nullptr,
             "The fluxes have already been received at time step "
                 << time_step_id
                 << ". They are either being received for a second time, there "
                    "is a bug in the ordering of the actions (though a "
                    "different ASSERT should've caught that), or the incorrect "
                    "temporal ID is being sent.");
      ASSERT(boundary_message->subcell_ghost_data == nullptr,
             "Have already received ghost cells at time step "
                 << time_step_id
                 << ", but we are being sent ghost cells again. This data will "
                    "not be cleaned up while the dg flux data is being "
                    "inserted into the inbox which will cause a memory leak.");

      ASSERT(current_boundary_data->interface_mesh ==
                 boundary_message->interface_mesh,
             "The mesh being received for the fluxes is different than the "
             "mesh received for the ghost cells. Mesh for fluxes: "
                 << boundary_message->interface_mesh << " mesh for ghost cells "
                 << current_boundary_data->interface_mesh);
      ASSERT(current_boundary_data->volume_or_ghost_mesh ==
                 boundary_message->volume_or_ghost_mesh,
             "The mesh being received for the ghost cell data is different "
             "than the mesh received previously. Mesh for received when we got "
             "fluxes: "
                 << boundary_message->volume_or_ghost_mesh
                 << " mesh received when we got ghost cells "
                 << current_boundary_data->volume_or_ghost_mesh);

      // Just need to set the pointer. No need to worry about the data being
      // deleted upon destruction of boundary_message because the destructor of
      // a BoundaryMessage will only delete the pointer stored inside the
      // BoundaryMessage. It won't delete the actual data it points to. Now the
      // unique_ptr owns the data that was previously pointed to by
      // boundary_message
      current_boundary_data->dg_flux_data_size =
          boundary_message->dg_flux_data_size;
      current_boundary_data->dg_flux_data = boundary_message->dg_flux_data;
      current_boundary_data->next_time_step_id =
          boundary_message->next_time_step_id;
      current_boundary_data->tci_status = boundary_message->tci_status;
    } else {
      // We have not received ghost cells or fluxes at this time.
      // Once we insert boundary_message into the unique_ptr we cannot use
      // boundary_message anymore because it is invalidated. The unique_ptr now
      // owns the memory.
      if (not current_inbox
                  .insert(std::pair{key, std::unique_ptr<BoundaryMessage<Dim>>(
                                             boundary_message)})
                  .second) {
        ERROR("Failed to insert data to receive at instance '"
              << time_step_id << "' with tag 'BoundaryMessageInbox'.\n");
      }
    }
  }
};
}  // namespace evolution::dg::Tags
