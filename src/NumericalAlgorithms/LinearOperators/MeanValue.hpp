// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function mean_value and mean_value_on_boundary.

#pragma once

#include <cstddef>
#include <utility>

#include "Domain/Side.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <size_t>
class Mesh;
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the mean value of a grid-function over a manifold.
 * \f$mean value = \int f dV / \int dV\f$
 *
 * \remarks The mean value is computed on the reference element(s).
 * \note The mean w.r.t. a different set of coordinates x can be computed
 * by pre-multiplying the argument f by the Jacobian J = dx/dxi of the mapping
 * from the reference coordinates xi to the coordinates x.
 *
 * \returns the mean value of `f` on the manifold
 * \param f the grid function of which to find the mean.
 * \param mesh the Mesh of the manifold on which f is located.
 */
template <size_t Dim>
double mean_value(const DataVector& f, const Mesh<Dim>& mesh) noexcept {
  return definite_integral(f, mesh) / two_to_the(Dim);
}

// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * Compute the mean value of a grid-function on a boundary of a manifold.
 * \f$mean value = \int f dV / \int dV\f$
 *
 * \remarks The mean value is computed on the reference element(s).
 *
 * \returns the mean value of `f` on the boundary of the manifold
 *
 * - `f` the grid function of which to find the mean.
 * - `mesh` the Mesh of the manifold on which f is located.
 * - `d` the dimension which is sliced away to get the boundary.
 * - `side` whether it is the lower or upper boundary in the d-th dimension.
 * - `boundary_buffer` is a pointer to a DataVector of size
 *   `mesh.slice_away(d).number_of_grid_points()` used as a temporary buffer
 *   when slicing the data to the boundary.
 * - `volume_and_slice_indices` a pair of `(volume_index_for_point,
 *   slice_index_for_point)` computed using the `SliceIterator`. Because
 *   `SliceIterator` is somewhat expensive, if computing the mean value on the
 *   same boundary for many different tensor components, prefer computing the
 *   slice indices once.
 */
template <size_t Dim>
double mean_value_on_boundary(const DataVector& f, const Mesh<Dim>& mesh,
                              size_t d, Side side) noexcept;

template <size_t Dim>
double mean_value_on_boundary(gsl::not_null<DataVector*> boundary_buffer,
                              const DataVector& f, const Mesh<Dim>& mesh,
                              size_t d, Side side) noexcept;

double mean_value_on_boundary(gsl::not_null<DataVector*> /*boundary_buffer*/,
                              const DataVector& f, const Mesh<1>& mesh,
                              size_t d, Side side) noexcept;

template <size_t Dim>
double mean_value_on_boundary(
    gsl::not_null<DataVector*> boundary_buffer,
    gsl::span<std::pair<size_t, size_t>> volume_and_slice_indices,
    const DataVector& f, const Mesh<Dim>& mesh, size_t d,
    Side /*side*/) noexcept;

double mean_value_on_boundary(
    gsl::not_null<DataVector*> /*boundary_buffer*/,
    gsl::span<std::pair<size_t, size_t>> /*volume_and_slice_indices*/,
    const DataVector& f, const Mesh<1>& mesh, size_t d, Side side) noexcept;
// @}
