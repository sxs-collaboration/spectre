// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "RegularGridInterpolant.hpp"

#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace intrp {

template <size_t Dim>
RegularGrid<Dim>::RegularGrid(const Mesh<Dim>& source_mesh,
                              const Mesh<Dim>& target_mesh) noexcept {
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(interpolation_matrices_, d) = Spectral::interpolation_matrix(
        source_mesh.slice_through(d),
        get<0>(logical_coordinates(target_mesh.slice_through(d))));
  }
}

template <size_t Dim>
RegularGrid<Dim>::RegularGrid(
    const Mesh<Dim>& source_mesh,
    const std::array<DataVector, Dim>& target_1d_logical_coords) noexcept {
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(interpolation_matrices_, d) = Spectral::interpolation_matrix(
        source_mesh.slice_through(d), gsl::at(target_1d_logical_coords, d));
  }
}

template <size_t Dim>
RegularGrid<Dim>::RegularGrid() = default;

template <size_t Dim>
void RegularGrid<Dim>::pup(PUP::er& p) noexcept {
  p | interpolation_matrices_;
}

template <size_t LocalDim>
bool operator==(const RegularGrid<LocalDim>& lhs,
                const RegularGrid<LocalDim>& rhs) noexcept {
  return lhs.interpolation_matrices_ == rhs.interpolation_matrices_;
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
