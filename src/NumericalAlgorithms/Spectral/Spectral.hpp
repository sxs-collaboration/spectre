// Distributed under the MIT License.
// See LICENSE.txt for details.

/*!
 * \file
 * Declares functionality to retrieve spectral quantities associated with
 * a particular choice of basis functions and quadrature.
 */

#pragma once

/*!
 * \ingroup SpectralGroup
 * \brief Functionality associated with a particular choice of basis functions
 * and quadrature for spectral operations
 */
namespace Spectral {

/*!
 * \ingroup SpectralGroup
 * \brief The choice of basis functions for computing collocation points and
 * weights.
 */
enum class Basis { Legendre };

/*!
 * \ingroup SpectralGroup
 * \brief The choice of quadrature method to compute integration weights.
 */
enum class Quadrature { Gauss, GaussLobatto };

}  // namespace Spectral
