// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <limits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "ErrorHandling/Error.hpp"

namespace Spectral {
template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::FiniteDifference, Quadrature::CellCentered>(
    const size_t num_points) noexcept {
  DataVector x{num_points};
  DataVector w{num_points, std::numeric_limits<double>::signaling_NaN()};
  // The finite difference grid cells cover the interval [-1, 1]
  constexpr double lower_bound = -1.0, upper_bound = 1.0;
  const double delta_x = (upper_bound - lower_bound) / num_points;
  for (size_t i = 0; i < num_points; ++i) {
    x[i] = lower_bound + 0.5 * delta_x + i * delta_x;
  }
  return std::make_pair(std::move(x), std::move(w));
}

// The below definitions are necessary to successfully link with some compilers.
template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points) noexcept;

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <>
Matrix spectral_indefinite_integral_matrix<Basis::FiniteDifference>(
    const size_t /*num_points*/) noexcept {
  ERROR("No indefinite integration matrix exists.\n");
}

template <>
DataVector compute_basis_function_value<Basis::FiniteDifference>(
    const size_t /*k*/, const DataVector& /*x*/) noexcept {
  ERROR("No basis functions to compute.\n");
}

template <>
double compute_basis_function_value<Basis::FiniteDifference>(
    const size_t /*k*/, const double& /*x*/) noexcept {
  ERROR("No basis functions to compute.\n");
}

template <>
DataVector compute_inverse_weight_function_values<Basis::FiniteDifference>(
    const DataVector& /*x*/) noexcept {
  ERROR("No no inverse weight function to compute.\n");
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)
}  // namespace Spectral
