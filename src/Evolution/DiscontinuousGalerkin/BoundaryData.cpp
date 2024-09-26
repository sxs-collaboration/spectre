// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"

#include <pup.h>
#include <pup_stl.h>

#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Serialization/PupStlCpp17.hpp"
#include "Utilities/StdHelpers.hpp"

namespace evolution::dg {
template <size_t Dim>
void BoundaryData<Dim>::pup(PUP::er& p) {
  p | volume_mesh;
  p | volume_mesh_ghost_cell_data;
  p | interface_mesh;
  p | ghost_cell_data;
  p | boundary_correction_data;
  p | validity_range;
  p | tci_status;
  p | integration_order;
}

template <size_t Dim>
bool operator==(const BoundaryData<Dim>& lhs, const BoundaryData<Dim>& rhs) {
  return lhs.volume_mesh == rhs.volume_mesh and
         lhs.volume_mesh_ghost_cell_data == rhs.volume_mesh_ghost_cell_data and
         lhs.interface_mesh == rhs.interface_mesh and
         lhs.ghost_cell_data == rhs.ghost_cell_data and
         lhs.boundary_correction_data == rhs.boundary_correction_data and
         lhs.validity_range == rhs.validity_range and
         lhs.tci_status == rhs.tci_status and
         lhs.integration_order == rhs.integration_order;
}

template <size_t Dim>
bool operator!=(const BoundaryData<Dim>& lhs, const BoundaryData<Dim>& rhs) {
  return not(lhs == rhs);
}

template <size_t Dim>
std::ostream& operator<<(std::ostream& os, const BoundaryData<Dim>& value) {
  return os << "Volume mesh: " << value.volume_mesh << '\n'
            << "Ghost mesh: " << value.volume_mesh_ghost_cell_data << '\n'
            << "Interface mesh: " << value.interface_mesh << '\n'
            << "Ghost cell data: " << value.ghost_cell_data << '\n'
            << "Boundary correction: " << value.boundary_correction_data << '\n'
            << "Validy range: " << value.validity_range << '\n'
            << "TCI status: " << value.tci_status << '\n'
            << "Integration order: " << value.integration_order;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                        \
  template class BoundaryData<DIM(data)>;                             \
  template std::ostream& operator<<(                                  \
      std::ostream& os, const BoundaryData<DIM(data)>& BoundaryData); \
  template bool operator==(const BoundaryData<DIM(data)>& lhs,        \
                           const BoundaryData<DIM(data)>& rhs);       \
  template bool operator!=(const BoundaryData<DIM(data)>& lhs,        \
                           const BoundaryData<DIM(data)>& rhs);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION
#undef DIM
}  // namespace evolution::dg
