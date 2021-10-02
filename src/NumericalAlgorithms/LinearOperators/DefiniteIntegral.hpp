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
 * \param integrand the function to integrate.
 * \param mesh the Mesh defining the grid points on the manifold.
 */
template <size_t Dim>
double definite_integral(const DataVector& integrand, const Mesh<Dim>& mesh);
