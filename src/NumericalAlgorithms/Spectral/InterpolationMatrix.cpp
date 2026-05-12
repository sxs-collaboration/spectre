// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/InterpolationMatrix.hpp"

#include <cstddef>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/BarycentricWeights.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Fourier.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctions/Zernike.hpp"
#include "NumericalAlgorithms/Spectral/CollocationPoints.hpp"
#include "NumericalAlgorithms/Spectral/GetSpectralQuantityForMesh.hpp"
#include "NumericalAlgorithms/Spectral/MaximumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"

namespace Spectral {
template <Basis BasisType, Quadrature QuadratureType, typename T>
Matrix interpolation_matrix(const size_t num_points, const T& target_points) {
  if constexpr (BasisType == Spectral::Basis::FiniteDifference) {
    ERROR(
        "Cannot do barycentric interpolation with Basis::FiniteDifference.  "
        "Use an IrregularInterpolant");
  }
  if constexpr (BasisType == Spectral::Basis::Cartoon) {
    ASSERT(num_points == 1, "A Cartoon basis can only have one point.");
    const size_t num_target_points = get_size(target_points);
    Matrix interp_matrix(num_target_points, num_points);
    for (size_t k = 0; k < num_target_points; ++k) {
      interp_matrix(k, 0) = 1.0;
    }
    return interp_matrix;
  } else if constexpr (BasisType == Spectral::Basis::Fourier) {
    return Fourier::interpolation_matrix(num_points, target_points);
  }
  constexpr size_t max_num_points =
      Spectral::maximum_number_of_points<BasisType>;
  constexpr size_t min_num_points =
      Spectral::minimum_number_of_points<BasisType, QuadratureType>;
  ASSERT(num_points >= min_num_points,
         "Tried to work with less than the minimum number of collocation "
         "points for this quadrature.");
  ASSERT(num_points <= max_num_points,
         "Exceeded maximum number of collocation points.");
  const DataVector& collocation_pts =
      collocation_points<BasisType, QuadratureType>(num_points);
  const DataVector& bary_weights =
      detail::barycentric_weights<BasisType, QuadratureType>(num_points);
  const size_t num_target_points = get_size(target_points);
  Matrix interp_matrix(num_target_points, num_points);
  // Algorithm 32 in Kopriva, p. 76
  // This is valid for any collocation points.
  for (size_t k = 0; k < num_target_points; k++) {
    // Check where no interpolation is necessary since a target point
    // matches the original collocation points
    bool row_has_match = false;
    for (size_t j = 0; j < num_points; j++) {
      interp_matrix(k, j) = 0.0;
      if (equal_within_roundoff(get_element(target_points, k),
                                collocation_pts[j])) {
        interp_matrix(k, j) = 1.0;
        row_has_match = true;
      }
    }
    // Perform interpolation for non-matching points
    if (not row_has_match) {
      double sum = 0.0;
      for (size_t j = 0; j < num_points; j++) {
        interp_matrix(k, j) = bary_weights[j] / (get_element(target_points, k) -
                                                 collocation_pts[j]);
        sum += interp_matrix(k, j);
      }
      for (size_t j = 0; j < num_points; j++) {
        interp_matrix(k, j) /= sum;
      }
    }
  }
  return interp_matrix;
}

template <typename T>
Matrix interpolation_matrix(const Mesh<1>& mesh, const T& target_points) {
  return detail::get_spectral_quantity_for_mesh(
      [target_points](const auto basis, const auto quadrature,
                      const size_t num_points) -> Matrix {
        return interpolation_matrix<decltype(basis)::value,
                                    decltype(quadrature)::value>(num_points,
                                                                 target_points);
      },
      mesh);
}

template <Basis BasisType, Quadrature QuadratureType, typename T>
Matrix interpolation_matrix(size_t num_points, const T& target_points,
                            const Spectral::Parity parity) {
  static_assert(BasisType == Basis::ZernikeB1,
                "Only Basis::ZernikeB1 is supported by this overload; for "
                "ZernikeB2/ZernikeB3 use Irregular or Cardinal interpolants.");
  static_assert(
      QuadratureType == Quadrature::GaussRadauUpper,
      "Zernike bases are only instantiated with GaussRadauUpper quadrature.");
  ASSERT(parity != Parity::Uninitialized,
         "Parity must be set to either Even or Odd");
  return Zernike<1>::interpolation_matrix(num_points, target_points, parity);
}

template Matrix interpolation_matrix<Basis::Chebyshev, Quadrature::Gauss>(
    size_t, const std::vector<double>&);
template Matrix
interpolation_matrix<Basis::Chebyshev, Quadrature::GaussLobatto>(
    size_t, const std::vector<double>&);
template Matrix interpolation_matrix<Basis::Fourier, Quadrature::Equiangular>(
    size_t, const std::vector<double>&);
template Matrix interpolation_matrix<Basis::Legendre, Quadrature::Gauss>(
    size_t, const std::vector<double>&);
template Matrix interpolation_matrix<Basis::Legendre, Quadrature::GaussLobatto>(
    size_t, const std::vector<double>&);
template Matrix interpolation_matrix<Basis::Chebyshev, Quadrature::Gauss>(
    size_t, const DataVector&);
template Matrix interpolation_matrix<
    Basis::Chebyshev, Quadrature::GaussLobatto>(size_t, const DataVector&);
template Matrix interpolation_matrix<Basis::Fourier, Quadrature::Equiangular>(
    size_t, const DataVector&);
template Matrix interpolation_matrix<Basis::Legendre, Quadrature::Gauss>(
    size_t, const DataVector&);
template Matrix interpolation_matrix<Basis::Legendre, Quadrature::GaussLobatto>(
    size_t, const DataVector&);
template Matrix interpolation_matrix<Basis::Fourier, Quadrature::Equiangular>(
    size_t, const double&);
template Matrix Spectral::interpolation_matrix(const Mesh<1>&,
                                               const DataVector&);
template Matrix Spectral::interpolation_matrix(const Mesh<1>&,
                                               const std::vector<double>&);
template Matrix Spectral::interpolation_matrix(const Mesh<1>&, const double&);

template Matrix
interpolation_matrix<Basis::ZernikeB1, Quadrature::GaussRadauUpper>(
    size_t, const std::vector<double>&, Spectral::Parity);
template Matrix
interpolation_matrix<Basis::ZernikeB1, Quadrature::GaussRadauUpper>(
    size_t, const DataVector&, Spectral::Parity);
template Matrix
interpolation_matrix<Basis::ZernikeB1, Quadrature::GaussRadauUpper>(
    size_t, const double&, Spectral::Parity);
}  // namespace Spectral
