// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/DgSubcell/NeighborRdmpAndVolumeData.hpp"

#include <algorithm>
#include <boost/functional/hash.hpp>
#include <functional>
#include <iterator>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMapHelpers.hpp"
#include "Evolution/DgSubcell/GhostData.hpp"
#include "Evolution/DgSubcell/Matrices.hpp"
#include "Evolution/DgSubcell/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/MakeArray.hpp"

namespace evolution::dg::subcell {
template <bool InsertIntoMap, size_t Dim>
void insert_or_update_neighbor_volume_data(
    const gsl::not_null<
        FixedHashMap<maximum_number_of_neighbors(Dim),
                     std::pair<Direction<Dim>, ElementId<Dim>>, GhostData,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>*>
        ghost_data_ptr,
    const DataVector& neighbor_subcell_data,
    const size_t number_of_rdmp_vars_in_buffer,
    const std::pair<Direction<Dim>, ElementId<Dim>>& directional_element_id,
    const Mesh<Dim>& neighbor_mesh, const Element<Dim>& element,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_zones) {
  ASSERT(neighbor_mesh == Mesh<Dim>(neighbor_mesh.extents(0),
                                    neighbor_mesh.basis(0),
                                    neighbor_mesh.quadrature(0)),
         "The neighbor mesh must be uniform but is " << neighbor_mesh);
  ASSERT(neighbor_subcell_data.size() != 0,
         "neighbor_subcell_data must be non-empty");
  const size_t end_of_volume_data =
      neighbor_subcell_data.size() - 2 * number_of_rdmp_vars_in_buffer;

  if constexpr (InsertIntoMap) {
    (*ghost_data_ptr)[directional_element_id] = GhostData{1};
  }

  DataVector& ghost_data = (*ghost_data_ptr)[directional_element_id]
                               .neighbor_ghost_data_for_reconstruction();
  DataVector computed_ghost_data{};
  if (neighbor_mesh.basis(0) == Spectral::Basis::FiniteDifference) {
    ASSERT(neighbor_mesh == subcell_mesh,
           "Neighbor mesh ("
               << neighbor_mesh << ") and my mesh (" << subcell_mesh
               << ") must be the same if we are both doing subcell.");
    if (not InsertIntoMap and
        neighbor_subcell_data.data() == ghost_data.data()) {
      // Short-circuit if we are already doing FD and we would be
      // self-assigning, so elide copy and move.
      return;
    }
    // Copy over the ghost cell data for subcell reconstruction. In this case
    // the neighbor would have reoriented the data for us.
    computed_ghost_data.destructive_resize(end_of_volume_data);
    std::copy(
        neighbor_subcell_data.begin(),
        std::prev(neighbor_subcell_data.end(),
                  2 * static_cast<typename std::iterator_traits<
                          typename DataVector::iterator>::difference_type>(
                          number_of_rdmp_vars_in_buffer)),
        computed_ghost_data.begin());
  } else {
    ASSERT(evolution::dg::subcell::fd::mesh(neighbor_mesh) == subcell_mesh,
           "Neighbor subcell mesh computed from the neighbor DG mesh ("
               << evolution::dg::subcell::fd::mesh(neighbor_mesh)
               << ") and my mesh (" << subcell_mesh << ") must be the same.");
    const Direction<Dim>& direction = directional_element_id.first;
    const auto& orientation = element.neighbors().at(direction).orientation();
    const auto& neighbor_orientation_map = orientation.inverse_map();
    const size_t total_number_of_ghost_zones =
        number_of_ghost_zones *
        subcell_mesh.extents().slice_away(direction.dimension()).product();
    ASSERT(
        end_of_volume_data % neighbor_mesh.number_of_grid_points() == 0,
        "The number of DG volume grid points times the number of variables "
        "sent for reconstruction ("
            << end_of_volume_data
            << ") must be a multiple of the number of DG volume grid points: "
            << neighbor_mesh.number_of_grid_points() << " number of RDMP vars "
            << number_of_rdmp_vars_in_buffer);
    const size_t number_of_vars =
        end_of_volume_data / neighbor_mesh.number_of_grid_points();
    const auto project_to_ghost_data =
        [&direction, &computed_ghost_data, &number_of_ghost_zones,
         &subcell_mesh](const Mesh<Dim>& neighbor_mesh_for_projection,
                        const DataVector& neighbor_data_for_projection) {
          // Project to ghosts
          Matrix empty{};
          auto ghost_projection_mat = make_array<Dim>(std::cref(empty));
          for (size_t i = 0; i < Dim; ++i) {
            if (i == direction.dimension()) {
              gsl::at(ghost_projection_mat, i) =
                  std::cref(evolution::dg::subcell::fd::projection_matrix(
                      neighbor_mesh_for_projection.slice_through(i),
                      subcell_mesh.extents(i), number_of_ghost_zones,
                      direction.opposite().side()));
            } else {
              gsl::at(ghost_projection_mat, i) =
                  std::cref(evolution::dg::subcell::fd::projection_matrix(
                      neighbor_mesh_for_projection.slice_through(i),
                      subcell_mesh.extents(i),
                      Spectral::Quadrature::CellCentered));
            }
          }
          apply_matrices(make_not_null(&computed_ghost_data),
                         ghost_projection_mat, neighbor_data_for_projection,
                         neighbor_mesh_for_projection.extents());
        };
    // Note: Once we have fully unstructured mesh support we could completely
    // elide projection, instead treating DG neighbors as unstructured meshes.
    // Whether this would actually be cheaper than projecting and using uniform
    // meshes will need to be profiled.
    computed_ghost_data.destructive_resize(total_number_of_ghost_zones *
                                           number_of_vars);
    const DataVector neighbor_data_without_rdmp_vars{
        // NOLINTNEXTLINE(cppcoreguidelines-pro-type-const-cast)
        const_cast<double*>(neighbor_subcell_data.data()), end_of_volume_data};
    if (LIKELY(orientation.is_aligned())) {
      project_to_ghost_data(neighbor_mesh, neighbor_data_without_rdmp_vars);
    } else {
      // The neighbor sent DG data, so we need to project it to the ghost cells.
      //
      // We do not reorient the data on send, since this could add a lot of
      // unnecessary orientations in smooth regions. Instead, we reorient on
      // receive here.
      const Mesh<Dim> reoriented_neighbor_mesh =
          neighbor_orientation_map(neighbor_mesh);
      DataVector temp_oriented_volume_data{end_of_volume_data};
      orient_variables(make_not_null(&temp_oriented_volume_data),
                       neighbor_data_without_rdmp_vars, neighbor_mesh.extents(),
                       neighbor_orientation_map);
      project_to_ghost_data(reoriented_neighbor_mesh,
                            temp_oriented_volume_data);
    }
  }

  ghost_data = std::move(computed_ghost_data);
}

template <size_t Dim>
void insert_neighbor_rdmp_and_volume_data(
    const gsl::not_null<RdmpTciData*> rdmp_tci_data_ptr,
    const gsl::not_null<
        FixedHashMap<maximum_number_of_neighbors(Dim),
                     std::pair<Direction<Dim>, ElementId<Dim>>, GhostData,
                     boost::hash<std::pair<Direction<Dim>, ElementId<Dim>>>>*>
        ghost_data_ptr,
    const DataVector& received_neighbor_subcell_data,
    const size_t number_of_rdmp_vars,
    const std::pair<Direction<Dim>, ElementId<Dim>>& directional_element_id,
    const Mesh<Dim>& neighbor_mesh, const Element<Dim>& element,
    const Mesh<Dim>& subcell_mesh, const size_t number_of_ghost_zones) {
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
      number_of_ghost_zones);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                             \
  template void insert_neighbor_rdmp_and_volume_data(                      \
      gsl::not_null<RdmpTciData*> rdmp_tci_data_ptr,                       \
      gsl::not_null<FixedHashMap<                                          \
          maximum_number_of_neighbors(GET_DIM(data)),                      \
          std::pair<Direction<GET_DIM(data)>, ElementId<GET_DIM(data)>>,   \
          GhostData,                                                       \
          boost::hash<std::pair<Direction<GET_DIM(data)>,                  \
                                ElementId<GET_DIM(data)>>>>*>              \
          ghost_data_ptr,                                                  \
      const DataVector& neighbor_subcell_data, size_t number_of_rdmp_vars, \
      const std::pair<Direction<GET_DIM(data)>, ElementId<GET_DIM(data)>>& \
          directional_element_id,                                          \
      const Mesh<GET_DIM(data)>& neighbor_mesh,                            \
      const Element<GET_DIM(data)>& element,                               \
      const Mesh<GET_DIM(data)>& subcell_mesh, size_t number_of_ghost_zones);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define GET_INSERT(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(r, data)                                             \
  template void insert_or_update_neighbor_volume_data<GET_INSERT(data)>(   \
      gsl::not_null<FixedHashMap<                                          \
          maximum_number_of_neighbors(GET_DIM(data)),                      \
          std::pair<Direction<GET_DIM(data)>, ElementId<GET_DIM(data)>>,   \
          GhostData,                                                       \
          boost::hash<std::pair<Direction<GET_DIM(data)>,                  \
                                ElementId<GET_DIM(data)>>>>*>              \
          ghost_data_ptr,                                                  \
      const DataVector& received_neighbor_subcell_data,                    \
      size_t number_of_rdmp_vars_in_buffer,                                \
      const std::pair<Direction<GET_DIM(data)>, ElementId<GET_DIM(data)>>& \
          directional_element_id,                                          \
      const Mesh<GET_DIM(data)>& neighbor_mesh,                            \
      const Element<GET_DIM(data)>& element,                               \
      const Mesh<GET_DIM(data)>& subcell_mesh, size_t number_of_ghost_zones);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (true, false))

#undef INSTANTIATION
#undef GET_DIM
}  // namespace evolution::dg::subcell
