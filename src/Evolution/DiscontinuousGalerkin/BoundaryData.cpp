// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"

#include <cstddef>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Evolution/DiscontinuousGalerkin/InterpolatedBoundaryData.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"

namespace evolution::dg {
template <size_t Dim>
void BoundaryData<Dim>::pup(PUP::er& p) {
  p | volume_mesh;
  p | volume_mesh_ghost_cell_data;
  p | boundary_correction_mesh;
  p | ghost_cell_data;
  p | boundary_correction_data;
  p | validity_range;
  p | tci_status;
  p | integration_order;
  p | interpolated_boundary_data;
}

template <size_t Dim>
bool operator==(const BoundaryData<Dim>& lhs, const BoundaryData<Dim>& rhs) {
  return lhs.volume_mesh == rhs.volume_mesh and
         lhs.volume_mesh_ghost_cell_data == rhs.volume_mesh_ghost_cell_data and
         lhs.boundary_correction_mesh == rhs.boundary_correction_mesh and
         lhs.ghost_cell_data == rhs.ghost_cell_data and
         lhs.boundary_correction_data == rhs.boundary_correction_data and
         lhs.validity_range == rhs.validity_range and
         lhs.tci_status == rhs.tci_status and
         lhs.integration_order == rhs.integration_order and
         lhs.interpolated_boundary_data == rhs.interpolated_boundary_data;
}

template <size_t Dim>
bool operator!=(const BoundaryData<Dim>& lhs, const BoundaryData<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const BoundaryData<Dim>& value) {
  using ::operator<<;
  return os << "Volume mesh: " << value.volume_mesh << '\n'
            << "Ghost mesh: " << value.volume_mesh_ghost_cell_data << '\n'
            << "Boundary correction mesh: " << value.boundary_correction_mesh
            << '\n'
            << "Ghost cell data: " << value.ghost_cell_data << '\n'
            << "Boundary correction data: " << value.boundary_correction_data
            << '\n'
            << "Validy range: " << value.validity_range << '\n'
            << "TCI status: " << value.tci_status << '\n'
            << "Integration order: " << value.integration_order << '\n'
            << "Interpolated boundary data: "
            << value.interpolated_boundary_data;
}

template <size_t Dim>
void merge_boundary_data(const gsl::not_null<BoundaryData<Dim>*> destination,
                         BoundaryData<Dim> source) {
  auto& [volume_mesh, volume_mesh_ghost_cell_data, boundary_correction_mesh,
         ghost_cell_data, boundary_correction_data, validity_range, tci_status,
         integration_order, interpolated_boundary_data] = source;
  (void)ghost_cell_data;
  auto& [current_volume_mesh, current_volume_mesh_ghost_cell_data,
         current_boundary_correction_mesh, current_ghost_cell_data,
         current_boundary_correction_data, current_validity_range,
         current_tci_status, current_integration_order,
         current_interpolated_boundary_data] = *destination;
  (void)current_volume_mesh_ghost_cell_data;  // Need to use when
                                              // optimizing subcell
  ASSERT(current_ghost_cell_data.has_value(),
         "Have not yet received ghost cells, but the inbox entry already "
         "exists. This is a bug in the ordering of the actions.");
  ASSERT(not current_boundary_correction_data.has_value() and
             not current_boundary_correction_mesh.has_value(),
         "The fluxes have already been received. They are either being "
         "received for a second time, there is a bug in the ordering of the "
         "actions (though a different ASSERT should've caught that), or the "
         "incorrect temporal ID is being sent.");

  ASSERT(current_volume_mesh == volume_mesh,
         "The mesh being received for the fluxes is different than the "
         "mesh received for the ghost cells. Mesh for fluxes: "
             << volume_mesh << " mesh for ghost cells " << current_volume_mesh);
  ASSERT(current_volume_mesh_ghost_cell_data == volume_mesh_ghost_cell_data,
         "The mesh being received for the ghost cell data is different "
         "than the mesh received previously. Mesh for received when we got "
         "fluxes: "
             << volume_mesh_ghost_cell_data
             << " mesh received when we got ghost cells "
             << current_volume_mesh_ghost_cell_data);

  current_boundary_correction_mesh = boundary_correction_mesh;
  current_boundary_correction_data = std::move(boundary_correction_data);
  current_validity_range = validity_range;
  current_tci_status = tci_status;
  current_integration_order = integration_order;
  current_interpolated_boundary_data = std::move(interpolated_boundary_data);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                        \
  template class BoundaryData<DIM(data)>;                             \
  template std::ostream& operator<<(                                  \
      std::ostream& os, const BoundaryData<DIM(data)>& BoundaryData); \
  template bool operator==(const BoundaryData<DIM(data)>& lhs,        \
                           const BoundaryData<DIM(data)>& rhs);       \
  template bool operator!=(const BoundaryData<DIM(data)>& lhs,        \
                           const BoundaryData<DIM(data)>& rhs);       \
  template void merge_boundary_data(                                  \
      gsl::not_null<BoundaryData<DIM(data)>*> destination,            \
      BoundaryData<DIM(data)> source);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
