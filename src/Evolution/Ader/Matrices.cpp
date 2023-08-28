// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Ader/Matrices.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Basis.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Quadrature.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/StaticCache.hpp"

namespace ader::dg {
template <SpatialDiscretization::Basis Basis,
          SpatialDiscretization::Quadrature Quadrature,
          typename SpectralQuantityGenerator>
const auto& precomputed_spectral_quantity(const size_t num_points) {
  constexpr size_t max_num_points = Spectral::maximum_number_of_points<Basis>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<Basis, Quadrature>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  // We compute the quantity for all possible `num_point`s the first time this
  // function is called and keep the data around for the lifetime of the
  // program. The computation is handled by the call operator of the
  // `SpectralQuantityType` instance.
  static const auto precomputed_data =
      make_static_cache<CacheRange<min_num_points, max_num_points + 1>>(
          SpectralQuantityGenerator{});
  return precomputed_data(num_points);
}

template <SpatialDiscretization::Basis Basis,
          SpatialDiscretization::Quadrature Quadrature>
struct PredictorInverseTemporalMatrix;

template <SpatialDiscretization::Basis Basis>
struct PredictorInverseTemporalMatrix<
    Basis, SpatialDiscretization::Quadrature::GaussLobatto> {
  Matrix operator()(const size_t num_points) const {
    const DataVector& collocation_pts = Spectral::collocation_points<
        Basis, SpatialDiscretization::Quadrature::GaussLobatto>(num_points);
    Matrix differences(num_points, num_points);
    for (size_t i = 0; i < num_points; ++i) {
      for (size_t j = 0; j < num_points; ++j) {
        differences(i, j) = collocation_pts[i] - collocation_pts[j];
      }
    }

    // computes d (ell_j(x))/dx at x = x_k
    const auto lagrange_deriv = [num_points, &differences](const size_t j,
                                                           const size_t k) {
      double result = 0.0;
      for (size_t i = 0; i < num_points; ++i) {
        if (i == j) {
          continue;
        }
        double temp = 1.0 / differences(j, i);
        for (size_t m = 0; m < num_points; ++m) {
          if (m == i or m == j) {
            continue;
          }
          temp *= differences(k, m) / differences(j, m);
        }
        result += temp;
      }
      return result;
    };

    const DataVector& weights = Spectral::quadrature_weights<
        Basis, SpatialDiscretization::Quadrature::GaussLobatto>(num_points);
    Matrix result(num_points, num_points);
    Matrix mass(num_points, num_points, 0.0);
    for (size_t i = 0; i < num_points; ++i) {
      mass(i, i) = weights[i];
      for (size_t j = 0; j < num_points; ++j) {
        result(i, j) = -weights[j] * lagrange_deriv(i, j);
      }
    }
    // The 1.0 here assumes Gauss-Lobatto points since it hard codes ell_i(1)
    result(num_points - 1, num_points - 1) += 1.0;
    return inv(result) * mass;
  }
};

template <SpatialDiscretization::Basis Basis,
          SpatialDiscretization::Quadrature Quadrature>
const Matrix& predictor_inverse_temporal_matrix(const size_t num_points) {
  return precomputed_spectral_quantity<
      Basis, Quadrature, PredictorInverseTemporalMatrix<Basis, Quadrature>>(
      num_points);
}

template const Matrix& predictor_inverse_temporal_matrix<
    SpatialDiscretization::Basis::Legendre,
    SpatialDiscretization::Quadrature::GaussLobatto>(const size_t num_points);
}  // namespace ader::dg
