// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/Spectral.hpp"

#include <limits>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Spectral {
template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::FiniteDifference, Quadrature::CellCentered>(
    const size_t num_points) {
  DataVector x{num_points};
  // The finite difference grid cells cover the interval [-1, 1]
  constexpr double lower_bound = -1.0;
  constexpr double upper_bound = 1.0;
  const double delta_x = (upper_bound - lower_bound) / num_points;
  // Weights for integration using midpoint method
  DataVector w{num_points, delta_x};
  for (size_t i = 0; i < num_points; ++i) {
    x[i] = lower_bound + 0.5 * delta_x + i * delta_x;
  }
  return std::make_pair(std::move(x), std::move(w));
}

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::FiniteDifference, Quadrature::FaceCentered>(
    const size_t num_points) {
  DataVector x{num_points};
  // The finite difference grid cells cover the interval [-1, 1]
  constexpr double lower_bound = -1.0;
  constexpr double upper_bound = 1.0;
  const double delta_x = (upper_bound - lower_bound) / (num_points - 1.0);
  // Weights for integration using midpoint method
  DataVector w{num_points, delta_x};
  for (size_t i = 0; i < num_points; ++i) {
    x[i] = lower_bound + i * delta_x;
  }
  return std::make_pair(std::move(x), std::move(w));
}

template <>
DataVector compute_inverse_weight_function_values<Basis::FiniteDifference>(
    const DataVector& x) {
  DataVector iw{x.size(),1.0};
  return iw;
}

// The below definitions are necessary to successfully link with some compilers.
template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points);

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif  // defined(__GNUC__) && !defined(__clang__)
template <>
Matrix spectral_indefinite_integral_matrix<Basis::FiniteDifference>(
    const size_t /*num_points*/) {
  ERROR("No indefinite integration matrix exists.\n");
}

template <>
DataVector compute_basis_function_value<Basis::FiniteDifference>(
    const size_t /*k*/, const DataVector& /*x*/) {
  ERROR("No basis functions to compute.\n");
}

template <>
double compute_basis_function_value<Basis::FiniteDifference>(
    const size_t /*k*/, const double& /*x*/) {
  ERROR("No basis functions to compute.\n");
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif  // defined(__GNUC__) && !defined(__clang__)
}  // namespace Spectral
