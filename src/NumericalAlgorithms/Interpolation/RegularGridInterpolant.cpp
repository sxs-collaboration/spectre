// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "RegularGridInterpolant.hpp"

#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace intrp {

template <size_t Dim>
RegularGrid<Dim>::RegularGrid(
    const Mesh<Dim>& source_mesh, const Mesh<Dim>& target_mesh,
    const std::array<DataVector, Dim>&
        override_target_mesh_with_1d_logical_coords) noexcept
    : number_of_target_points_(1), source_extents_(source_mesh.extents()) {
  for (size_t d = 0; d < Dim; ++d) {
    const auto source_mesh_1d = source_mesh.slice_through(d);
    const auto target_mesh_1d = target_mesh.slice_through(d);
    if (gsl::at(override_target_mesh_with_1d_logical_coords, d).size() == 0) {
      // Normal mode: use target_mesh
      // Only make a matrix if source and target meshes differ; when a default-
      // constructed matrix is given instead, apply_matrices does no work.
      if (source_mesh_1d != target_mesh_1d) {
        gsl::at(interpolation_matrices_, d) = Spectral::interpolation_matrix(
            source_mesh_1d, get<0>(logical_coordinates(target_mesh_1d)));
      }
      number_of_target_points_ *= target_mesh_1d.number_of_grid_points();
    } else {
      // Override mode: override target_mesh with given points
      const auto& target_coords_1d =
          gsl::at(override_target_mesh_with_1d_logical_coords, d);
      gsl::at(interpolation_matrices_, d) =
          Spectral::interpolation_matrix(source_mesh_1d, target_coords_1d);
      number_of_target_points_ *= target_coords_1d.size();
    }
  }
}

template <size_t Dim>
RegularGrid<Dim>::RegularGrid() = default;

template <size_t Dim>
void RegularGrid<Dim>::pup(PUP::er& p) noexcept {
  p | number_of_target_points_;
  p | source_extents_;
  p | interpolation_matrices_;
}

template <size_t LocalDim>
bool operator==(const RegularGrid<LocalDim>& lhs,
                const RegularGrid<LocalDim>& rhs) noexcept {
  return lhs.number_of_target_points_ == rhs.number_of_target_points_ and
         lhs.source_extents_ == rhs.source_extents_ and
         lhs.interpolation_matrices_ == rhs.interpolation_matrices_;
}

template <size_t Dim>
bool operator!=(const RegularGrid<Dim>& lhs,
                const RegularGrid<Dim>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                        \
  template class RegularGrid<DIM(data)>;                            \
  template bool operator==(const RegularGrid<DIM(data)>&,           \
                           const RegularGrid<DIM(data)>&) noexcept; \
  template bool operator!=(const RegularGrid<DIM(data)>&,           \
                           const RegularGrid<DIM(data)>&) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef DIM
#undef INSTANTIATE
/// \endcond

}  // namespace intrp
