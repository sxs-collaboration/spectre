// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

/// @{
/*!
 * \ingroup SpectralGroup
 *
 * \brief Evaluates a real spherical harmonic of order l at the requested
 * angles \f$\theta\f$ and \f$\phi\f$.
 *
 * The real spherical harmonics are defined as:
 * \begin{equation}
 * Y_{lm}(\theta, \phi) =
 * \begin{cases}
 * \sqrt{2} (-1)^m \mathcal{I}[Y_l^{|m|}] & \text{if } m < 0 \\
 *  Y_l^{0} & \text{if } m = 0 \\
 * \sqrt{2} (-1)^m \mathcal{R}[Y_l^{m}] & \text{if } m > 0
 * \end{cases}
 * \end{equation}
 *
 * where \f$Y_l^m\f$ are the complex spherical harmonics and \f$\mathcal{R}\f$
 * denotes the real part and \f$\mathcal{I}\f$ denotes the imaginary part.
 *
 * \note This implementation uses the boost implementation of spherical
 * harmonics. The calculation is not vectorized and may be slow.
 * Stability has not been tested for large l.
 */
void real_spherical_harmonic(gsl::not_null<DataVector*> spherical_harmonic,
                             const DataVector& theta, const DataVector& phi,
                             size_t l, int m);

DataVector real_spherical_harmonic(const DataVector& theta,
                                   const DataVector& phi, size_t l, int m);
/// @}
