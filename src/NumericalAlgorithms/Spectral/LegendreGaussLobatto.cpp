// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "NumericalAlgorithms/Spectral/LegendreGaussLobatto.hpp"

#include "DataStructures/Matrix.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

/// \cond
namespace Basis {
namespace lgl {

const DataVector& collocation_points(size_t number_of_pts) {
  return Spectral::collocation_points<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto>(
      number_of_pts);
}

const DataVector& quadrature_weights(size_t number_of_pts) {
  return Spectral::quadrature_weights<Spectral::Basis::Legendre,
                                      Spectral::Quadrature::GaussLobatto>(
      number_of_pts);
}

const Matrix& differentiation_matrix(size_t number_of_pts) {
  return Spectral::differentiation_matrix<Spectral::Basis::Legendre,
                                          Spectral::Quadrature::GaussLobatto>(
      number_of_pts);
}

const Matrix& grid_points_to_spectral_matrix(size_t number_of_pts) {
  return Spectral::grid_points_to_spectral_matrix<
      Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>(
      number_of_pts);
}

const Matrix& spectral_to_grid_points_matrix(size_t number_of_pts) {
  return Spectral::spectral_to_grid_points_matrix<
      Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto>(
      number_of_pts);
}

const Matrix& linear_filter_matrix(size_t number_of_pts) {
  return Spectral::linear_filter_matrix<Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto>(
      number_of_pts);
}

template <typename T>
Matrix interpolation_matrix(size_t number_of_pts, const T& target_points) {
  return Spectral::interpolation_matrix<Spectral::Basis::Legendre,
                                        Spectral::Quadrature::GaussLobatto>(
      number_of_pts, target_points);
}

}  // namespace lgl
}  // namespace Basis

template Matrix Basis::lgl::interpolation_matrix(
    const size_t num_collocation_points, const DataVector& target_points);
template Matrix Basis::lgl::interpolation_matrix(
    const size_t num_collocation_points,
    const std::vector<double>& target_points);
/// \endcond
