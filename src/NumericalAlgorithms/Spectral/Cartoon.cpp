// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <cstddef>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/InverseWeightFunctionValues.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

namespace Spectral {

// There should be no calls to Cartoon basis functions, will error
// These functions specialize the templates declared in `Spectral.hpp`.

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif

template <>
DataVector compute_basis_function_value<Basis::Cartoon>(
    const size_t /*k*/, const DataVector& /*x*/) {
  ERROR("Invalid to compute a basis on a Cartoon basis.");
}

template <>
double compute_basis_function_value<Basis::Cartoon>(const size_t /*k*/,
                                                    const double& /*x*/) {
  ERROR("Invalid to compute a basis on a Cartoon basis.");
}

template <>
DataVector compute_inverse_weight_function_values<Basis::Cartoon>(
    const DataVector& /*x*/) {
  ERROR("Invalid to compute weights on a Cartoon basis.");
}

template <>
double compute_basis_function_normalization_square<Basis::Cartoon>(
    const size_t /*k*/) {
  ERROR("Invalid to compute weights on a Cartoon basis.");
}

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::Cartoon, Quadrature::AxialSymmetry>(const size_t /*num_points*/) {
  ERROR(
      "Invalid to compute collocation points and weights for a Cartoon "
      "basis.");
}

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::Cartoon, Quadrature::SphericalSymmetry>(
    const size_t /*num_points*/) {
  ERROR(
      "Invalid to compute collocation points and weights for a Cartoon "
      "basis.");
}

template <Basis BasisTYpe>
Matrix spectral_indefinite_integral_matrix(size_t num_points);

template <>
Matrix spectral_indefinite_integral_matrix<Basis::Cartoon>(
    const size_t /*num_points*/) {
  ERROR(
      "Invalid to compute a spectral indefinite integral matrix for  a"
      "Cartoon basis.");
}
#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif
}  // namespace Spectral
