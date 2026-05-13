// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <cstddef>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/DirectionalId.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/RdmpTciData.hpp"
#include "NumericalAlgorithms/Interpolation/IrregularInterpolant.hpp"
#include "NumericalAlgorithms/Spectral/ParityFromSymmetry.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <size_t Dim>
class Element;
template <size_t Dim>
class Mesh;
/// \endcond

namespace evolution::dg::subcell::detail {
/*!
 * \brief Implementation of insert_or_update_neighbor_volume_data, accepting
 * parity information as runtime arguments so the caller can be a thin
 * template wrapper without needing VolumeFields in the function body.
 *
 * The impl-unique parameters are specifically for a ZernikeB1 basis:
 * - **`parity_list`** Run-length encoded even/odd parity of each component
 * (ignored when Dim != 3 or when the mesh is not ZernikeB1).
 * - **`num_even`** Number of even-parity components (0 when unused).
 * - **`num_odd`** Number of odd-parity components (0 when unused).
 */
template <bool InsertIntoMap, size_t Dim>
void insert_or_update_neighbor_volume_data_impl(
    gsl::not_null<DirectionalIdMap<Dim, GhostData>*> ghost_data_ptr,
    const DataVector& neighbor_subcell_data,
    size_t number_of_rdmp_vars_in_buffer,
    const DirectionalId<Dim>& directional_element_id,
    const Mesh<Dim>& neighbor_mesh, const Element<Dim>& element,
    const Mesh<Dim>& subcell_mesh, size_t number_of_ghost_zones,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        neighbor_dg_to_fd_interpolants,
    gsl::span<const size_t> parity_list, size_t num_even, size_t num_odd);
}  // namespace evolution::dg::subcell::detail

namespace evolution::dg::subcell {
/*!
 * \brief Check whether `neighbor_subcell_data` is FD or DG, and either insert
 * or copy into `ghost_data_ptr` the FD data (projecting if
 * `neighbor_subcell_data` is DG data).
 *
 * This is intended to be used during a rollback from DG to make sure neighbor
 * data is projected to the FD grid.
 *
 * The `VolumeFields` meta parameter is the tag list (a `tmpl::list`) of the
 * packed ghost DataVector so that per-component parity information can be
 * deduced for ZernikeB1 meshes.
 */
template <bool InsertIntoMap, size_t Dim, typename VolumeFields>
void insert_or_update_neighbor_volume_data(
    const gsl::not_null<DirectionalIdMap<Dim, GhostData>*> ghost_data_ptr,
    const DataVector& neighbor_subcell_data,
    const size_t number_of_rdmp_vars_in_buffer,
    const DirectionalId<Dim>& directional_element_id,
    const Mesh<Dim>& neighbor_mesh, const Element<Dim>& element,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_zones,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        neighbor_dg_to_fd_interpolants,
    VolumeFields /*meta*/) {
  if constexpr (Dim == 3) {
    const auto parity_info = Spectral::compute_parity_list<VolumeFields>();
    const auto& parity_list = std::get<0>(parity_info);
    detail::insert_or_update_neighbor_volume_data_impl<InsertIntoMap, Dim>(
        ghost_data_ptr, neighbor_subcell_data, number_of_rdmp_vars_in_buffer,
        directional_element_id, neighbor_mesh, element, subcell_mesh,
        number_of_ghost_zones, neighbor_dg_to_fd_interpolants,
        gsl::span<const size_t>{parity_list.data(), parity_list.size()},
        std::get<1>(parity_info), std::get<2>(parity_info));
  } else {
    detail::insert_or_update_neighbor_volume_data_impl<InsertIntoMap, Dim>(
        ghost_data_ptr, neighbor_subcell_data, number_of_rdmp_vars_in_buffer,
        directional_element_id, neighbor_mesh, element, subcell_mesh,
        number_of_ghost_zones, neighbor_dg_to_fd_interpolants, {}, 0, 0);
  }
}

/*!
 * \brief Check whether the neighbor sent is DG volume or FD ghost data, and
 * orient project DG volume data if necessary.
 *
 * This is intended to be used by the `ReceiveDataForReconstruction` action.
 *
 * The `VolumeFields` parameter is the tag list (a `tmpl::list`) of the
 * packed ghost DataVector so that per-component parity information can be
 * deduced for ZernikeB1 meshes.
 */
template <size_t Dim, typename VolumeFields>
void insert_neighbor_rdmp_and_volume_data(
    const gsl::not_null<RdmpTciData*> rdmp_tci_data_ptr,
    const gsl::not_null<DirectionalIdMap<Dim, GhostData>*> ghost_data_ptr,
    const DataVector& received_neighbor_subcell_data,
    const size_t number_of_rdmp_vars,
    const DirectionalId<Dim>& directional_element_id,
    const Mesh<Dim>& neighbor_mesh, const Element<Dim>& element,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_zones,
    const DirectionalIdMap<Dim, std::optional<intrp::Irregular<Dim>>>&
        neighbor_dg_to_fd_interpolants,
    VolumeFields volume_fields) {
  ASSERT(received_neighbor_subcell_data.size() != 0,
         "received_neighbor_subcell_data must be non-empty");
  // Note: since we determine the starting point of the RDMP vars
  // from how many RDMP vars there are, we don't need to account for
  // the mesh the neighbor sent (DG or FD)
  const size_t max_offset =
      received_neighbor_subcell_data.size() - 2 * number_of_rdmp_vars;
  const size_t min_offset =
      received_neighbor_subcell_data.size() - number_of_rdmp_vars;
  for (size_t var_index = 0; var_index < number_of_rdmp_vars; ++var_index) {
    rdmp_tci_data_ptr->max_variables_values[var_index] =
        std::max(rdmp_tci_data_ptr->max_variables_values[var_index],
                 received_neighbor_subcell_data[max_offset + var_index]);
    rdmp_tci_data_ptr->min_variables_values[var_index] =
        std::min(rdmp_tci_data_ptr->min_variables_values[var_index],
                 received_neighbor_subcell_data[min_offset + var_index]);
  }
  // Note: it would be good to assert that the neighbor is at the same
  // refinement level as us, but such a function does not yet exist.

  insert_or_update_neighbor_volume_data<true>(
      ghost_data_ptr, received_neighbor_subcell_data, number_of_rdmp_vars,
      directional_element_id, neighbor_mesh, element, subcell_mesh,
      number_of_ghost_zones, neighbor_dg_to_fd_interpolants, volume_fields);
}
}  // namespace evolution::dg::subcell
