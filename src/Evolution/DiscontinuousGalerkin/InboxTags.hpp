// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <map>
#include <optional>
#include <tuple>
#include <type_traits>
#include <utility>

#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
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
 * 1. the mesh of the neighboring element's face (not the mortar mesh!)
 * 2. the variables at the ghost zone cells for finite difference/volume
 *    reconstruction
 * 3. the data on the mortar needed for computing the boundary corrections (e.g.
 *    fluxes, characteristic speeds, conserved variables)
 * 4. the TimeStepId beyond which the boundary terms are no longer valid, when
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
 *   valid before a DG element sends ghost cell and bonudary data then no
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
      std::tuple<Mesh<Dim - 1>, std::optional<std::vector<double>>,
                 std::optional<std::vector<double>>, ::TimeStepId>;

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
    auto& [mesh, ghost_cell_data, boundary_data, boundary_data_validity_range] =
        data.second;
    (void)ghost_cell_data;

    if (auto it = current_inbox.find(data.first); it != current_inbox.end()) {
      auto& [current_mesh, current_ghost_cell_data, current_boundary_data,
             current_boundary_data_validity_range] = it->second;
      // We have already received some data at this time. Receiving data twice
      // at the same time should only occur when receiving fluxes after having
      // previously received ghost cells. We sanity check that the data we
      // already have is the ghost cells and that we have not yet received flux
      // data.
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

      ASSERT(current_mesh == mesh,
             "The mesh being received for the fluxes is different than the "
             "mesh received for the ghost cells. Mesh for fluxes: "
                 << mesh << " mesh for ghost cells " << current_mesh);

      // We always move here since we take ownership of the data and moves
      // implicitly decay to copies
      current_boundary_data = std::move(boundary_data);
      current_boundary_data_validity_range = boundary_data_validity_range;
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
}  // namespace evolution::dg::Tags
