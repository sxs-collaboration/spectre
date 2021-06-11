// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions mean_value and mean_value_on_boundary.

#pragma once

#include <cstddef>
#include <utility>

#include "Domain/Structure/Side.hpp"
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
 * \brief Compute the mean value of a function over a manifold.
 *
 * Given a function \f$f\f$, compute its mean value \f$\bar{f}\f$ with respect
 * to the logical coordinates \f$\boldsymbol{\xi} = (\xi, \eta, \zeta)\f$. E.g.,
 * in 1 dimension, \f$\bar{f} = \int_{-1}^1 f d\xi \Big/ \int_{-1}^1 d\xi\f$.
 *
 * \note
 * The mean w.r.t. a different set of coordinates
 * \f$\boldsymbol{x} = \boldsymbol{x}(\boldsymbol{\xi})\f$ can't be directly
 * computed using this function. Before calling `mean_value`, \f$f\f$ must be
 * pre-multiplied by the Jacobian determinant
 * \f$J = \det d\boldsymbol{x}/d\boldsymbol{\xi}\f$ of the mapping
 * \f$\boldsymbol{x}(\boldsymbol{\xi})\f$. Additionally, the output of
 * `mean_value` must be multiplied by a factor
 * \f$2^{\text{d}} / \int J d^{\text{d}}\xi\f$ (in \f$d\f$ dimensions), to
 * account for the different volume of the manifold in the \f$\boldsymbol{x}\f$
 * coordinates.
 *
 * \param f the function to average.
 * \param mesh the Mesh defining the grid points on the manifold.
 */
template <size_t Dim>
double mean_value(const DataVector& f, const Mesh<Dim>& mesh) noexcept {
  return definite_integral(f, mesh) / two_to_the(Dim);
}

/// @{
/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the mean value of a function over a boundary of a manifold.
 *
 * Given a function \f$f\f$, compute its mean value \f$\bar{f}\f$, over a
 * boundary, with respect to the logical coordinates
 * \f$\boldsymbol{\xi} = (\xi, \eta, \zeta)\f$.
 *
 * \see `mean_value` for notes about means w.r.t. other coordinates.
 *
 * - `f` the function to average.
 * - `mesh` the Mesh defining the grid points on the manifold.
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
/// @}
