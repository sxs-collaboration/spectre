// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"

#include <array>  // Used for mesh.slices
#include <ostream>

#include "DataStructures/DataVector.hpp"
#include "Domain/Mesh.hpp"
#include "ErrorHandling/Assert.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Blas.hpp"

/// \cond
// The 2D and 3D definite integrals have been optimized and are up to 2x faster
// than the previous implementation. The main differences are
//
// - No memory allocations are required
// - Manual loop unrolling (this will be somewhat hardware specific on the
//   ~decade time scale).
// - Pointers to avoid any potential overhead from indexing
//
// Note: The inner loop is over x because the memory layout used by SpECTRE is
//       x-varies-fastest.

template <>
double definite_integral<1>(const DataVector& integrand,
                            const Mesh<1>& mesh) noexcept {
  const size_t num_grid_points = mesh.number_of_grid_points();
  ASSERT(integrand.size() == num_grid_points,
         "num_grid_points = " << num_grid_points
                              << ", integrand size = " << integrand.size());
  const DataVector& weights = Spectral::quadrature_weights(mesh);
  return ddot_(num_grid_points, weights.data(), 1, integrand.data(), 1);
}

template <>
double definite_integral<2>(const DataVector& integrand,
                            const Mesh<2>& mesh) noexcept {
  ASSERT(integrand.size() == mesh.number_of_grid_points(),
         "num_grid_points = " << mesh.number_of_grid_points()
                              << ", integrand size = " << integrand.size());
  const auto sliced_meshes = mesh.slices();
  const size_t x_size = sliced_meshes[0].number_of_grid_points();
  const size_t x_last_unrolled = x_size - x_size % 2;
  const size_t y_size = sliced_meshes[1].number_of_grid_points();
  const double* const w_x =
      Spectral::quadrature_weights(sliced_meshes[0]).data();
  const double* const w_y =
      Spectral::quadrature_weights(sliced_meshes[1]).data();

  double result = 0.0;
  for (size_t j = 0; j < y_size; ++j) {
    const size_t offset = j * x_size;
    for (size_t i = 0; i < x_last_unrolled; i += 2) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      result += w_y[j] * w_x[i] * integrand[i + offset] +
                w_y[j] * w_x[i + 1] * integrand[i + 1 + offset];  // NOLINT
    }
    for (size_t i = x_last_unrolled; i < x_size; ++i) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      result += w_y[j] * w_x[i] * integrand[i + offset];
    }
  }
  return result;
}

template <>
double definite_integral<3>(const DataVector& integrand,
                            const Mesh<3>& mesh) noexcept {
  ASSERT(integrand.size() == mesh.number_of_grid_points(),
         "num_grid_points = " << mesh.number_of_grid_points()
                              << ", integrand size = " << integrand.size());
  const auto sliced_meshes = mesh.slices();
  const size_t x_size = sliced_meshes[0].number_of_grid_points();
  const size_t x_last_unrolled = x_size - x_size % 2;
  const size_t y_size = sliced_meshes[1].number_of_grid_points();
  const size_t z_size = sliced_meshes[2].number_of_grid_points();
  const double* const w_x =
      Spectral::quadrature_weights(sliced_meshes[0]).data();
  const double* const w_y =
      Spectral::quadrature_weights(sliced_meshes[1]).data();
  const double* const w_z =
      Spectral::quadrature_weights(sliced_meshes[2]).data();

  double result = 0.0;
  for (size_t k = 0; k < z_size; ++k) {
    for (size_t j = 0; j < y_size; ++j) {
      // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
      const double prod = w_z[k] * w_y[j];
      const size_t offset = x_size * (j + y_size * k);
      // Unrolling at 2 gives better performance on AVX machines (the most
      // common in 2018). The stride will probably need to be updated as
      // hardware changes. Note: using a single loop is ~15% faster when
      // x_size == 3, and both loop styles are comparable for x_size == 2.
      for (size_t i = 0; i < x_last_unrolled; i += 2) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        result += prod * w_x[i] * integrand[i + offset] +
                  prod * w_x[i + 1] * integrand[i + 1 + offset];  // NOLINT
      }
      for (size_t i = x_last_unrolled; i < x_size; ++i) {
        // NOLINTNEXTLINE(cppcoreguidelines-pro-bounds-pointer-arithmetic)
        result += prod * w_x[i] * integrand[i + offset];
      }
    }
  }
  return result;
}
/// \endcond
