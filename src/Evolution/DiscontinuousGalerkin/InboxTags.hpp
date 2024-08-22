// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <iomanip>
#include <map>
#include <memory>
#include <optional>
#include <sstream>
#include <string>
#include <type_traits>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DiscontinuousGalerkin/AtomicInboxBoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/Messages/BoundaryMessage.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/InboxInserters.hpp"
#include "Parallel/StaticSpscQueue.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace evolution::dg::Tags {
/*!
 * \brief The inbox tag for boundary correction communication and DG-subcell
 * ghost zone cells.
 *
 * The stored data consists of the following:
 *
 * 1. the volume mesh of the element.
 * 2. the volume mesh corresponding to the ghost cell data. This allows eliding
 *    projection when all neighboring elements are doing DG.
 * 3. the mesh of the neighboring element's face (not the mortar mesh!)
 * 4. the variables at the ghost zone cells for finite difference/volume
 *    reconstruction
 * 5. the data on the mortar needed for computing the boundary corrections (e.g.
 *    fluxes, characteristic speeds, conserved variables)
 * 6. the TimeStepId beyond which the boundary terms are no longer valid, when
 *    using local time stepping.
 * 7. the troublade cell indicator status using for determining halos around
 *    troubled cells.
 * 8. the integration order of the time-stepper.
 *
 * The TimeStepId is the neighboring element's next time step. When using local
 * time stepping, the neighbor's boundary data is valid up until this time,
 * which may include multiple local time steps. By receiving and storing the
 * neighbor time step, the local element knows whether or not it should remove
 * boundary data and expect new data to be sent from the neighbor.
 *
 * The ghost cell data will be valid whenever a DG-subcell scheme is being used.
 * Whenever a DG-subcell scheme is being used, elements using DG and not FD/FV
 * always send both the ghost cells and boundary correction data together.
 * Elements using FD/FV send the ghost cells first followed by the boundary
 * correction data once the element has received all neighbor ghost cell data.
 * Note that the second send/receive only modifies the flux and the TimeStepId
 * used for the flux validity range.
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
 *
 * #### Return Values:
 * - In the case that the type is `type_map` the `insert_into_inbox` function
 *   returns the size of the inbox.
 * - In the case that the type is `type_spsc` the `insert_into_inbox` function
 *   returns the number of neighbor data contributions made that allow the
 *   element to take its next time step/needs a message called on it. When
 *   this number is equal to the number of neighbors, a Charm++ message must
 *   be sent to the element to have it continue the algorithm. This is so as
 *   to minimize the number of communications made through Charm++ and instead
 *   to move data atomically between neighbors whenever possible.
 *
 * #### DG Element Nodegroup Support
 * If you are using the `DgElementCollection` then you must set
 * `UseNodegroupDgElements` to `true`. The actions that use this tag check
 * that the parallel component and the `UseNodegroupDgElements` is consistent.
 */
template <size_t Dim, bool UseNodegroupDgElements>
struct BoundaryCorrectionAndGhostCellsInbox {
  using stored_type = evolution::dg::BoundaryData<Dim>;

 public:
  using temporal_id = TimeStepId;
  // Used by array implementation
  using type_map = std::map<TimeStepId, DirectionalIdMap<Dim, stored_type>>;

  // Used by nodegroup implementation
  using type_spsc = evolution::dg::AtomicInboxBoundaryData<Dim>;

  // The actual type being used.
  using type = tmpl::conditional_t<UseNodegroupDgElements, type_spsc, type_map>;
  using value_type = type;

  template <typename ReceiveDataType>
  static size_t insert_into_inbox(const gsl::not_null<type_spsc*> inbox,
                                  const temporal_id& time_step_id,
                                  ReceiveDataType&& data) {
    const DirectionalId<Dim>& neighbor_id = data.first;
    // Note: This assumes the neighbor_id is oriented into our (the element
    // whose inbox this is) frame.
    const size_t neighbor_index = inbox->index(neighbor_id);
    if (UNLIKELY(not gsl::at(inbox->boundary_data_in_directions, neighbor_index)
                         .try_emplace(time_step_id, std::move(data.second),
                                      std::move(data.first)))) {
      ERROR(
          "Failed to emplace data into inbox. neighbor_id: ("
          << neighbor_id.direction() << ',' << neighbor_id.id()
          << ") at TimeStepID: " << time_step_id << " the size of the inbox is "
          << gsl::at(inbox->boundary_data_in_directions, neighbor_index).size()
          << " the message count is "
          << gsl::at(inbox->boundary_data_in_directions, neighbor_index)
                 .message_count.load()
          << " and the number of neighbors is "
          << gsl::at(inbox->boundary_data_in_directions, neighbor_index)
                 .number_of_neighbors.load());
    }
    // Notes:
    // 1. fetch_add does a post-increment.
    // 2. We need thread synchronization here, so doing relaxed_order would be a
    //    bug.
    return inbox->message_count.fetch_add(1, std::memory_order_acq_rel) + 1;
  }

  template <typename ReceiveDataType>
  static size_t insert_into_inbox(const gsl::not_null<type_map*> inbox,
                                  const temporal_id& time_step_id,
                                  ReceiveDataType&& data) {
    auto& current_inbox = (*inbox)[time_step_id];
    if (auto it = current_inbox.find(data.first); it != current_inbox.end()) {
      auto& [volume_mesh, volume_mesh_of_ghost_cell_data, face_mesh,
             ghost_cell_data, boundary_data, boundary_data_validity_range,
             boundary_tci_status, integration_order] = data.second;
      (void)ghost_cell_data;
      auto& [current_volume_mesh, current_volume_mesh_of_ghost_cell_data,
             current_face_mesh, current_ghost_cell_data, current_boundary_data,
             current_boundary_data_validity_range, current_tci_status,
             current_integration_order] = it->second;
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
      current_integration_order = integration_order;
    } else {
      // We have not received ghost cells or fluxes at this time.
      if (not current_inbox.insert(std::forward<ReceiveDataType>(data))
                  .second) {
        ERROR("Failed to insert data to receive at instance '"
              << time_step_id
              << "' with tag 'BoundaryCorrectionAndGhostCellsInbox'.\n");
      }
    }
    return current_inbox.size();
  }

  static std::string output_inbox(const type_spsc& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');
    ss << std::scientific << std::setprecision(16);
    ss << pad << "BoundaryCorrectionAndGhostCellInbox:\n";
    ss << pad
       << "Warning: Printing atomic state is not possible in general so data "
          "printed is limited.\n";
    for (size_t i = 0; i < inbox.boundary_data_in_directions.size(); ++i) {
      const auto& data_in_direction =
          gsl::at(inbox.boundary_data_in_directions, i);
      ss << pad << "Id: "
         << "Approximate size: " << data_in_direction.size() << "\n";
    }

    return ss.str();
  }

  static std::string output_inbox(const type_map& inbox,
                                  const size_t padding_size) {
    std::stringstream ss{};
    const std::string pad(padding_size, ' ');
    ss << std::scientific << std::setprecision(16);
    ss << pad << "BoundaryCorrectionAndGhostCellInbox:\n";

    for (const auto& [current_time_step_id, hash_map] : inbox) {
      ss << pad << " Current time: " << current_time_step_id << "\n";
      // We only care about the next time because that's important for deadlock
      // detection. The data itself isn't super important
      for (const auto& [key, boundary_data] : hash_map) {
        ss << pad << "  Key: " << key
           << ", next time: " << boundary_data.validity_range << "\n";
      }
    }

    return ss.str();
  }

  void pup(PUP::er& /*p*/) {}
};

/*!
 * \brief Simple tag used to store inbox data in the DataBox.
 *
 * Since the inbox data can be received asynchronously and lockfree ordered
 * containers are slow and challenging to implement, we instead use an
 * unordered container for the inbox, then transfer the data into an ordered map
 * in the DataBox.
 */
template <size_t Dim>
struct BoundaryData : db::SimpleTag {
  using type =
      typename BoundaryCorrectionAndGhostCellsInbox<Dim, false>::type_map;
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
  using type = std::map<TimeStepId, DirectionalIdMap<Dim, stored_type>>;
  using message_type = BoundaryMessage<Dim>;

  template <typename Inbox>
  static void insert_into_inbox(const gsl::not_null<Inbox*> inbox,
                                BoundaryMessage<Dim>* boundary_message) {
    const auto& time_step_id = boundary_message->current_time_step_id;
    auto& current_inbox = (*inbox)[time_step_id];

    const auto key = DirectionalId<Dim>{boundary_message->neighbor_direction,
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
      current_boundary_data->integration_order =
          boundary_message->integration_order;
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
