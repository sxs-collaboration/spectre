// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Declares functions for retrieving Legendre-Gauss-Lobatto related matrices

#pragma once

#include <cstddef>

class DataVector;
class Matrix;

/// \ingroup NumericalAlgorithmsGroup
/// Basis functions, derivative matrices, etc. for spectral-type methods
namespace Basis {
/// \ingroup NumericalAlgorithmsGroup
/// Functions for using Legendre-Gauss-Lobatto basis
namespace lgl {
/// Collocation points.
/// \param number_of_pts number of collocation points
const DataVector& collocation_points(size_t number_of_pts);

/// Quadrature weights.
/// \param number_of_pts number of collocation points
const DataVector& quadrature_weights(size_t number_of_pts);

/// Matrix used to compute the first derivative of a function.
/// \param number_of_pts number of collocation points
const Matrix& differentiation_matrix(size_t number_of_pts);

/// Matrix used to compute spectral coefficients of a function.
/// \param number_of_pts number of collocation points
const Matrix& grid_points_to_spectral_matrix(size_t number_of_pts);

/// Matrix used to compute function at the collocation points from its
/// spectral coefficients.
/// \param number_of_pts number of collocation points
const Matrix& spectral_to_grid_points_matrix(size_t number_of_pts);

/// Matrix used to linearize a function.
///
/// Filters out all except the lowest two modes.
/// \param number_of_pts number of collocation points
const Matrix& linear_filter_matrix(size_t number_of_pts);

/// Matrix to interpolate to a set of target points
///
/// \warning It is expected but not checked that the target points are inside
/// the domain.
/// \param number_of_pts number of collocation points
/// \param target_points the points to interpolate to
template <typename T>
Matrix interpolation_matrix(size_t number_of_pts, const T& target_points);

/// Maximum number of allowed collocation points for LGL basis
static const size_t maximum_number_of_pts = 12;
}  // namespace lgl
}  // namespace Basis
