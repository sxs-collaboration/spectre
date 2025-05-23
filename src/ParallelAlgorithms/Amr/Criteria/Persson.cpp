// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/Persson.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Amr/Flag.hpp"
#include "NumericalAlgorithms/Spectral/Filtering.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace amr::Criteria {

template <typename VectorType, size_t Dim>
double persson_smoothness_indicator(
    const gsl::not_null<VectorType*> filtered_component_buffer,
    const VectorType& tensor_component, const Mesh<Dim>& mesh,
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
  double norm = 0;
  if constexpr (std::is_same_v<VectorType, ComplexDataVector>) {
    norm = blaze::l2Norm(blaze::abs(*filtered_component_buffer));
  } else {
    norm = blaze::l2Norm(*filtered_component_buffer);
  }
  norm /= sqrt(filtered_component_buffer->size());
  return norm;
}

template <typename VectorType, size_t Dim>
std::array<double, Dim> persson_smoothness_indicator(
    const VectorType& tensor_component, const Mesh<Dim>& mesh,
    const size_t num_highest_modes) {
  std::array<double, Dim> result{};
  VectorType buffer{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = persson_smoothness_indicator(
        make_not_null(&buffer), tensor_component, mesh, d, num_highest_modes);
  }
  return result;
}

namespace Persson_detail {

template <typename VectorType, size_t Dim>
void max_over_components(const gsl::not_null<std::array<Flag, Dim>*> result,
                         const gsl::not_null<VectorType*> buffer,
                         const VectorType& tensor_component,
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

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                             \
  template double persson_smoothness_indicator(                          \
      gsl::not_null<DTYPE(data)*> buffer,                                \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh, \
      size_t dimension, size_t num_highest_modes);                       \
  template std::array<double, DIM(data)> persson_smoothness_indicator(   \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh, \
      size_t num_highest_modes);                                         \
  template void Persson_detail::max_over_components(                     \
      gsl::not_null<std::array<Flag, DIM(data)>*> result,                \
      gsl::not_null<DTYPE(data)*> buffer,                                \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh, \
      size_t num_highest_modes, double alpha, double absolute_tolerance, \
      double coarsening_factor);

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (1, 2, 3))

#undef INSTANTIATION
#undef DIM

}  // namespace amr::Criteria
