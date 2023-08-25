// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/Persson.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Amr/Flag.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace amr::Criteria {

template <size_t Dim>
double persson_smoothness_indicator(
    const gsl::not_null<DataVector*> filtered_component_buffer,
    const DataVector& tensor_component, const Mesh<Dim>& mesh,
    const size_t dimension, const size_t num_highest_modes) {
  // Zero out the lowest modes in the given dimension
  static const Matrix identity{};
  auto matrices = make_array<Dim>(std::cref(identity));
  gsl::at(matrices, dimension) = Spectral::filtering::zero_lowest_modes(
      mesh.slice_through(dimension),
      mesh.extents(dimension) - num_highest_modes);
  set_number_of_grid_points(filtered_component_buffer,
                            mesh.number_of_grid_points());
  apply_matrices(filtered_component_buffer, matrices, tensor_component,
                 mesh.extents());
  // Take the L2 norm over all grid points
  return blaze::l2Norm(*filtered_component_buffer) /
         sqrt(filtered_component_buffer->size());
}

template <size_t Dim>
std::array<double, Dim> persson_smoothness_indicator(
    const DataVector& tensor_component, const Mesh<Dim>& mesh,
    const size_t num_highest_modes) {
  std::array<double, Dim> result{};
  DataVector buffer{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = persson_smoothness_indicator(
        make_not_null(&buffer), tensor_component, mesh, d, num_highest_modes);
  }
  return result;
}

namespace Persson_detail {

template <size_t Dim>
void max_over_components(const gsl::not_null<std::array<Flag, Dim>*> result,
                         const gsl::not_null<DataVector*> buffer,
                         const DataVector& tensor_component,
                         const Mesh<Dim>& mesh, const size_t num_highest_modes,
                         const double alpha, const double absolute_tolerance,
                         const double coarsening_factor) {
  const double umax = max(abs(tensor_component));
  for (size_t d = 0; d < Dim; ++d) {
    // Skip this dimension if we have already decided to refine it
    if (gsl::at(*result, d) == Flag::Split) {
      continue;
    }
    const double relative_tolerance =
        pow(mesh.extents(d) - num_highest_modes, -alpha);
    const double indicator =
        persson_smoothness_indicator(buffer, tensor_component, mesh, d,
                                     num_highest_modes) /
        (relative_tolerance * umax + absolute_tolerance);
    if (indicator > 1.) {
      gsl::at(*result, d) = Flag::Split;
      continue;
    }
    // Don't check if we want to (allow) joining elements if another
    // tensor has already decided that joining elements is bad.
    if (gsl::at(*result, d) == Flag::DoNothing) {
      continue;
    }
    if (indicator < coarsening_factor) {
      gsl::at(*result, d) = Flag::Join;
    } else {
      gsl::at(*result, d) = Flag::DoNothing;
    }
  }
}

}  // namespace Persson_detail

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template double persson_smoothness_indicator(                              \
      gsl::not_null<DataVector*> buffer, const DataVector& tensor_component, \
      const Mesh<DIM(data)>& mesh, size_t dimension,                         \
      size_t num_highest_modes);                                             \
  template std::array<double, DIM(data)> persson_smoothness_indicator(       \
      const DataVector& tensor_component, const Mesh<DIM(data)>& mesh,       \
      size_t num_highest_modes);                                             \
  template void Persson_detail::max_over_components(                         \
      gsl::not_null<std::array<Flag, DIM(data)>*> result,                    \
      gsl::not_null<DataVector*> buffer, const DataVector& tensor_component, \
      const Mesh<DIM(data)>& mesh, size_t num_highest_modes, double alpha,   \
      double absolute_tolerance, double coarsening_factor);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATION
#undef DIM

}  // namespace amr::Criteria
