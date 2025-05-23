// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ParallelAlgorithms/Amr/Criteria/Loehner.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/Amr/Flag.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/SetNumberOfGridPoints.hpp"

namespace amr::Criteria {

template <typename VectorType, size_t Dim>
double loehner_smoothness_indicator(
    const gsl::not_null<VectorType*> first_deriv_buffer,
    const gsl::not_null<VectorType*> second_deriv_buffer,
    const VectorType& tensor_component, const Mesh<Dim>& mesh,
    const size_t dimension) {
  const size_t num_points = mesh.number_of_grid_points();
  set_number_of_grid_points(first_deriv_buffer, num_points);
  set_number_of_grid_points(second_deriv_buffer, num_points);
  static const Matrix identity{};
  // Compute second logical derivative in this dimension
  // Possible performance optimization: pre-compute the second derivative matrix
  // and store it either statically (like `Spectral::differentiation_matrix`)
  // or pass it in as an argument so it's re-used for each tensor component.
  const auto& logical_diff_matrix =
      Spectral::differentiation_matrix(mesh.slice_through(dimension));
  auto matrices = make_array<Dim>(std::cref(identity));
  gsl::at(matrices, dimension) = logical_diff_matrix;
  apply_matrices(first_deriv_buffer, matrices, tensor_component,
                 mesh.extents());
  apply_matrices(second_deriv_buffer, matrices, *first_deriv_buffer,
                 mesh.extents());
  // Take the L2 norm over all grid points
  double norm = 0;
  if constexpr (std::is_same_v<VectorType, ComplexDataVector>) {
    norm = blaze::l2Norm(blaze::abs(*second_deriv_buffer));
  } else {
    norm = blaze::l2Norm(*second_deriv_buffer);
  }
  norm /= sqrt(second_deriv_buffer->size());
  return norm;
}

template <typename VectorType, size_t Dim>
std::array<double, Dim> loehner_smoothness_indicator(
    const VectorType& tensor_component, const Mesh<Dim>& mesh) {
  std::array<double, Dim> result{};
  std::array<VectorType, 2> deriv_buffers{};
  for (size_t d = 0; d < Dim; ++d) {
    gsl::at(result, d) = loehner_smoothness_indicator(
        make_not_null(&deriv_buffers[0]), make_not_null(&deriv_buffers[1]),
        tensor_component, mesh, d);
  }
  return result;
}

namespace Loehner_detail {

template <typename VectorType, size_t Dim>
void max_over_components(
    const gsl::not_null<std::array<Flag, Dim>*> result,
    const gsl::not_null<std::array<VectorType, 2>*> deriv_buffers,
    const VectorType& tensor_component, const Mesh<Dim>& mesh,
    const double relative_tolerance, const double absolute_tolerance,
    const double coarsening_factor) {
  const double umax = max(abs(tensor_component));
  for (size_t d = 0; d < Dim; ++d) {
    // Skip this dimension if we have already decided to refine it
    if (gsl::at(*result, d) == Flag::Split) {
      continue;
    }
    const double indicator =
        loehner_smoothness_indicator(make_not_null(&(*deriv_buffers)[0]),
                                     make_not_null(&(*deriv_buffers)[1]),
                                     tensor_component, mesh, d) /
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

}  // namespace Loehner_detail

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                              \
  template double loehner_smoothness_indicator(                           \
      gsl::not_null<DTYPE(data)*> first_deriv_buffer,                     \
      gsl::not_null<DTYPE(data)*> second_deriv_buffer,                    \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh,  \
      size_t dimension);                                                  \
  template std::array<double, DIM(data)> loehner_smoothness_indicator(    \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh); \
  template void Loehner_detail::max_over_components(                      \
      gsl::not_null<std::array<Flag, DIM(data)>*> result,                 \
      gsl::not_null<std::array<DTYPE(data), 2>*> deriv_buffers,           \
      const DTYPE(data) & tensor_component, const Mesh<DIM(data)>& mesh,  \
      double relative_tolerance, double absolute_tolerance,               \
      double coarsening_factor);

GENERATE_INSTANTIATIONS(INSTANTIATE, (DataVector, ComplexDataVector), (1, 2, 3))

#undef INSTANTIATE
#undef DIM

}  // namespace amr::Criteria
