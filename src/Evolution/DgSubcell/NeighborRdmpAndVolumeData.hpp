// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/FixedHashMap.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/MaxNumberOfNeighbors.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
template <size_t Dim>
class Element;
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell {
/*!
 * \brief Check whether `neighbor_subcell_data` is FD or DG, and either insert
 * or copy into `ghost_data_ptr` the FD data (projecting if
 * `neighbor_subcell_data` is DG data).
 *
 * This is intended to be used during a rollback from DG to make sure neighbor
 * data is projected to the FD grid.
 */
template <bool InsertIntoMap, size_t Dim>
void insert_or_update_neighbor_volume_data(
    gsl::not_null<
        FixedHashMap<maximum_number_of_neighbors(Dim),
                     std::pair<Direction<Dim>, ElementId<Dim>>, GhostData,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>*>
        ghost_data_ptr,
    const DataVector& neighbor_subcell_data,
    const size_t number_of_rdmp_vars_in_buffer,
    const std::pair<Direction<Dim>, ElementId<Dim>>& directional_element_id,
    const Mesh<Dim>& neighbor_mesh, const Element<Dim>& element,
    const Mesh<Dim>& subcell_mesh, size_t number_of_ghost_zones);

/*!
 * \brief Check whether the neighbor sent is DG volume or FD ghost data, and
 * orient project DG volume data if necessary.
 *
 * This is intended to be used by the `ReceiveDataForReconstruction` action.
 */
template <size_t Dim>
void insert_neighbor_rdmp_and_volume_data(
    gsl::not_null<RdmpTciData*> rdmp_tci_data_ptr,
    gsl::not_null<
        FixedHashMap<maximum_number_of_neighbors(Dim),
                     std::pair<Direction<Dim>, ElementId<Dim>>, GhostData,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>*>
        ghost_data_ptr,
    const DataVector& received_neighbor_subcell_data,
    size_t number_of_rdmp_vars,
    const std::pair<Direction<Dim>, ElementId<Dim>>& directional_element_id,
    const Mesh<Dim>& neighbor_mesh, const Element<Dim>& element,
    const Mesh<Dim>& subcell_mesh, size_t number_of_ghost_zones);
}  // namespace evolution::dg::subcell
