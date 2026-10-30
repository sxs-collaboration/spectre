// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines function definite_integral.

#pragma once

#include <cstddef>

/// \cond
class DataVector;
template <size_t>
class Mesh;
/// \endcond

/*!
 * \ingroup NumericalAlgorithmsGroup
 * \brief Compute the definite integral of a function over a manifold.
 *
 * Given a function \f$f\f$, compute its integral \f$I\f$ with respect to
 * the logical coordinates \f$\boldsymbol{\xi} = (\xi, \eta, \zeta)\f$.
 * E.g., in 1 dimension, \f$I = \int_{-1}^1 f d\xi\f$.
 *
 * The integral w.r.t. a different set of coordinates
 * \f$\boldsymbol{x} = \boldsymbol{x}(\boldsymbol{\xi})\f$ can be computed
 * by pre-multiplying \f$f\f$ by the Jacobian determinant
 * \f$J = \det d\boldsymbol{x}/d\boldsymbol{\xi}\f$ of the mapping
 * \f$\boldsymbol{x}(\boldsymbol{\xi})\f$. Note that, in the
 * \f$\boldsymbol{x}\f$ coordinates, the domain of integration is the image of
 * the logical cube (square in 2D, interval in 1D) under the mapping.
 *
 * The integral is computed by quadrature, using the quadrature rule for the
 * basis associated with the collocation points.
 *
 * If the mesh uses a Cartoon basis, the passed integrand must already be
 * multiplied by the appropriate Jacobian determinants (i.e. both the
 * $\boldsymbol{x}(\boldsymbol{\xi})$ and the cartesian to spherical/polar
 * coordinates mappings, the latter being either $x^2$ or $x$, respectively). If
 * the Mesh uses a Zernike basis, the passed integrand must already be
 * multiplied by the appropriate cartesian to spherical/polar Jacobain
 * determinant ($r$ for ZernikeB2 and $r^2$ for ZernikeB3).
 *
 * \note For Zernike bases the radial integral is over
 * \f$r = (\xi + 1) / 2 \in [0, 1]\f$ rather than the logical coordinate
 * \f$\xi \in [-1, 1]\f$, requiring a factor of \f$d\xi/dr = 2\f$. This term
 * is applied internally by `definite_integral`, whereas the integrand still
 * must be multiplied by the Jacobian determinant of a coordinate map that
 * includes the polar or spherical coordinates (e.g. `PolarToCartesian` or
 * `SphericalToCartesianPfaffian`).
 *
 * \param integrand the function to integrate.
 * \param mesh the Mesh defining the grid points on the manifold.
 */
template <size_t Dim>
double definite_integral(const DataVector& integrand, const Mesh<Dim>& mesh);
