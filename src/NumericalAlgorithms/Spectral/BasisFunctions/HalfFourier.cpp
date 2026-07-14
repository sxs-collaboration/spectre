// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BasisFunctions/HalfFourier.hpp"

#include <cmath>
#include <cstddef>
#include <numbers>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace Spectral {

DataVector HalfFourier::collocation_points(const size_t num_points) {
  const double pi_over_n = std::numbers::pi / static_cast<double>(num_points);
  DataVector result(num_points);
  for (size_t j = 0; j < num_points; ++j) {
    result[j] = pi_over_n * (static_cast<double>(j) + 0.5);
  }
  return result;
}

DataVector HalfFourier::quadrature_weights(const size_t num_points) {
  return DataVector{num_points,
                    std::numbers::pi / static_cast<double>(num_points)};
}

Matrix HalfFourier::even_differentiation_matrix(const size_t num_points) {
  const double pi_over_n = std::numbers::pi / static_cast<double>(num_points);
  const double two_over_n = 2.0 / static_cast<double>(num_points);
  Matrix result(num_points, num_points, 0.0);
  for (size_t i = 0; i < num_points; ++i) {
    const double phi_i = pi_over_n * (static_cast<double>(i) + 0.5);
    for (size_t j = 0; j < num_points; ++j) {
      const double phi_j = pi_over_n * (static_cast<double>(j) + 0.5);
      double val = 0.0;
      for (size_t n = 1; n < num_points; ++n) {
        val += static_cast<double>(n) * sin(static_cast<double>(n) * phi_i) *
               cos(static_cast<double>(n) * phi_j);
      }
      result(i, j) = -two_over_n * val;
    }
  }
  return result;
}

Matrix HalfFourier::odd_differentiation_matrix(const size_t num_points) {
  const double pi_over_n = std::numbers::pi / static_cast<double>(num_points);
  const double two_over_n = 2.0 / static_cast<double>(num_points);
  Matrix result(num_points, num_points, 0.0);
  for (size_t i = 0; i < num_points; ++i) {
    const double phi_i = pi_over_n * (static_cast<double>(i) + 0.5);
    for (size_t j = 0; j < num_points; ++j) {
      const double phi_j = pi_over_n * (static_cast<double>(j) + 0.5);
      double val = 0.0;
      // Note: the n=num_points term vanishes because cos(num_points * phi_i) =
      // cos(pi*(i+0.5)) = 0, so only n=1,...,num_points-1 contribute.
      for (size_t n = 1; n < num_points; ++n) {
        val += static_cast<double>(n) * cos(static_cast<double>(n) * phi_i) *
               sin(static_cast<double>(n) * phi_j);
      }
      result(i, j) = two_over_n * val;
    }
  }
  return result;
}

template <typename T>
Matrix HalfFourier::even_interpolation_matrix(const size_t num_points,
                                              const T& target_points) {
  const double pi_over_n = std::numbers::pi / static_cast<double>(num_points);
  const double inv_n = 1.0 / static_cast<double>(num_points);
  const size_t num_target_points = get_size(target_points);
  Matrix result(num_target_points, num_points);
  for (size_t i = 0; i < num_target_points; ++i) {
    const double x = get_element(target_points, i);
    for (size_t j = 0; j < num_points; ++j) {
      const double phi_j = pi_over_n * (static_cast<double>(j) + 0.5);
      double val = 1.0;
      for (size_t n = 1; n < num_points; ++n) {
        val += 2.0 * cos(static_cast<double>(n) * x) *
               cos(static_cast<double>(n) * phi_j);
      }
      result(i, j) = inv_n * val;
    }
  }
  return result;
}

template <typename T>
Matrix HalfFourier::odd_interpolation_matrix(const size_t num_points,
                                             const T& target_points) {
  const double pi_over_n = std::numbers::pi / static_cast<double>(num_points);
  const double two_over_n = 2.0 / static_cast<double>(num_points);
  const double inv_n = 1.0 / static_cast<double>(num_points);
  const size_t num_target_points = get_size(target_points);
  Matrix result(num_target_points, num_points);
  for (size_t i = 0; i < num_target_points; ++i) {
    const double x = get_element(target_points, i);
    for (size_t j = 0; j < num_points; ++j) {
      const double phi_j = pi_over_n * (static_cast<double>(j) + 0.5);
      double val = 0.0;
      // Modes n=1,...,N-1 have discrete norm N/2 → coefficient 2/N.
      // Mode n=N (Nyquist) has discrete norm N → coefficient 1/N.
      for (size_t n = 1; n < num_points; ++n) {
        val += sin(static_cast<double>(n) * x) *
               sin(static_cast<double>(n) * phi_j);
      }
      result(i, j) =
          two_over_n * val + inv_n * sin(static_cast<double>(num_points) * x) *
                                 sin(static_cast<double>(num_points) * phi_j);
    }
  }
  return result;
}

template <typename T>
Matrix HalfFourier::interpolation_matrix(const size_t num_points,
                                         const T& target_points,
                                         const Parity parity) {
  ASSERT(parity != Parity::Uninitialized,
         "Parity must be set to either Even or Odd");
  if (parity == Parity::Even) {
    return even_interpolation_matrix(num_points, target_points);
  } else {
    return odd_interpolation_matrix(num_points, target_points);
  }
}

// Specializations of function templates defined in the Spectral directory

template <>
std::pair<DataVector, DataVector> compute_collocation_points_and_weights<
    Basis::HalfFourier, Quadrature::Equiangular>(const size_t num_points) {
  return std::make_pair(HalfFourier::collocation_points(num_points),
                        HalfFourier::quadrature_weights(num_points));
}

template <Basis BasisType>
Matrix spectral_indefinite_integral_matrix(size_t num_points);

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wsuggest-attribute=noreturn"
#endif

template <>
Matrix spectral_indefinite_integral_matrix<Basis::HalfFourier>(
    size_t /*num_points*/) {
  ERROR("Indefinite integral matrix is not implemented for HalfFourier basis");
}

template <>
DataVector compute_basis_function_value<Basis::HalfFourier>(
    const size_t /*k*/, const DataVector& /*x*/) {
  ERROR("HalfFourier basis function value requires a parity argument");
}

template <>
double compute_basis_function_value<Basis::HalfFourier>(const size_t /*k*/,
                                                        const double& /*x*/) {
  ERROR("HalfFourier basis function value requires a parity argument");
}

template <>
double compute_basis_function_normalization_square<Basis::HalfFourier>(
    const size_t /*k*/) {
  ERROR(
      "HalfFourier normalization square requires a parity argument; "
      "norms are pi (even k=0), pi/2 (even k>=1 or odd k=1,...,N).");
}

#if defined(__GNUC__) && !defined(__clang__)
#pragma GCC diagnostic pop
#endif

#define GET_TYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_INTERPOLATION(r, data)                \
  template Matrix HalfFourier::even_interpolation_matrix( \
      size_t num_points, const GET_TYPE(data)&);          \
  template Matrix HalfFourier::odd_interpolation_matrix(  \
      size_t num_points, const GET_TYPE(data)&);          \
  template Matrix HalfFourier::interpolation_matrix(      \
      size_t, const GET_TYPE(data)&, Parity);

GENERATE_INSTANTIATIONS(INSTANTIATE_INTERPOLATION,
                        (double, DataVector, std::vector<double>))

#undef INSTANTIATE_INTERP
#undef GET_TYPE
}  // namespace Spectral
