// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/BasisFunctions/Fourier.hpp"

#include <cmath>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionNormalizationSquare.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPointsAndWeights.hpp"
#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace Spectral {
template <typename T>
T Fourier::basis_function_value(const int k, const T& x) {
  if (UNLIKELY(k == 0)) {
    return make_with_value<T>(x, 1.);
  }
  if (k > 0) {
    return cos(k * x);
  }
  return sin(-k * x);
}

double Fourier::basis_function_normalization_square(const int k) {
  if (UNLIKELY(k == 0)) {
    return 2.0 * M_PI;
  }
  return M_PI;
}

DataVector Fourier::collocation_points(const size_t num_points) {
  DataVector x{num_points, 2.0 * M_PI / static_cast<double>(num_points)};
  for (size_t i = 0; i < num_points; ++i) {
    x[i] *= static_cast<double>(i);
  }
  return x;
}

DataVector Fourier::quadrature_weights(const size_t num_points) {
  return DataVector{num_points, 2.0 * M_PI / static_cast<double>(num_points)};
}

size_t Fourier::modal_storage_index(const int k) {
  if (UNLIKELY(k == 0)) {
    return size_t{0};
  }
  if (k > 0) {
    return static_cast<size_t>(2 * k - 1);
  }
  return 2 * static_cast<size_t>(-k);
}

int Fourier::mode_at_storage_index(const size_t storage_index) {
  if (UNLIKELY(storage_index == 0)) {
    return 0;
  }
  if (storage_index % 2 == 1) {
    return 1 + static_cast<int>(storage_index) / 2;
  }
  return -static_cast<int>(storage_index) / 2;
}

// As the collocation points are evenly spaced and this is a periodic
// domain, each row is identical to the previous row right-shifted by
// one element.  Furthermore the matrix is anti-symmetric.  Thus there
// are only N/2 unique elements that need to be computed; the rest can
// be determined from them
Matrix Fourier::differentiation_matrix(const size_t num_points) {
  const bool n_is_even = num_points % 2 == 0;
  Matrix result{num_points, num_points, 0.0};
  const double half_dx = M_PI / static_cast<double>(num_points);
  double coef = -0.5;
  for (size_t j = 1; j < (num_points + 1) / 2; ++j) {
    coef = -coef;
    const double y = static_cast<double>(j) * half_dx;
    result(0, j) = n_is_even ? coef * cos(y) / sin(y) : coef / sin(y);
    result(0, num_points - j) = -result(0, j);
  }
  for (size_t i = 1; i < num_points; ++i) {
    result(i, 0) = result(i - 1, num_points - 1);
    for (size_t j = 1; j < num_points; ++j) {
      result(i, j) = result(i - 1, j - 1);
    }
  }
  return result;
}

template <typename T>
Matrix Fourier::interpolation_matrix(const size_t num_points,
                                     const T& target_points) {
  const size_t num_target_points = get_size(target_points);
  Matrix result(num_target_points, num_points,
                1.0 / static_cast<double>(num_points));
  const DataVector x_source = collocation_points(num_points);
  const bool n_is_even = num_points % 2 == 0;
  for (size_t i = 0; i < num_target_points; ++i) {
    const double x_target = get_element(target_points, i);
    // Check where no interpolation is necessary since a target point
    // matches the original collocation points
    bool row_has_match = false;
    for (size_t j = 0; j < num_points; j++) {
      if (equal_within_roundoff(x_target, x_source[j])) {
        result(i, j) = 1.0;
        for (size_t m = 0; m < j; ++m) {
          result(i, m) = 0.0;
        }
        for (size_t m = j + 1; m < num_points; ++m) {
          result(i, m) = 0.0;
        }
        row_has_match = true;
        break;
      }
    }
    // Perform interpolation for non-matching points
    if (not row_has_match) {
      for (size_t j = 0; j < num_points; ++j) {
        const double half_dx = 0.5 * (x_target - x_source[j]);
        if (n_is_even) {
          result(i, j) *= sin(static_cast<double>(num_points) * half_dx) *
                          cos(half_dx) / sin(half_dx);
        } else {
          result(i, j) *=
              sin(static_cast<double>(num_points) * half_dx) / sin(half_dx);
        }
      }
    }
  }
  return result;
}

template <>
DataVector compute_basis_function_value<Basis::Fourier>(const size_t k,
                                                        const DataVector& x) {
  return Fourier::basis_function_value(Fourier::mode_at_storage_index(k), x);
}

template <>
double compute_basis_function_value<Basis::Fourier>(const size_t k,
                                                    const double& x) {
  return Fourier::basis_function_value(Fourier::mode_at_storage_index(k), x);
}

template <>
double compute_basis_function_normalization_square<Basis::Fourier>(
    const size_t k) {
  return Fourier::basis_function_normalization_square(
      Fourier::mode_at_storage_index(k));
}

template <>
std::pair<DataVector, DataVector>
compute_collocation_points_and_weights<Basis::Fourier, Quadrature::Equiangular>(
    const size_t num_points) {
  return std::make_pair(Fourier::collocation_points(num_points),
                        Fourier::quadrature_weights(num_points));
}

template <>
Matrix interpolation_matrix<Basis::Fourier, Quadrature::Equiangular>(
    const size_t num_points, const double& target_points) {
  return Fourier::interpolation_matrix(num_points, target_points);
}
}  // namespace Spectral
