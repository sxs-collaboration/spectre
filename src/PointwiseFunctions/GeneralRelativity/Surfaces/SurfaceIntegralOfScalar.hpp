// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
template <typename Frame>
class Strahlkorper;
/// \endcond

namespace gr::surfaces {
/*!
 * \ingroup SurfacesGroup
 * \brief Surface integral of a scalar on a 2D `Strahlkorper`
 *
 * \details Computes the surface integral \f$\oint dA f\f$ for a scalar \f$f\f$
 * on a `Strahlkorper` with area element \f$dA\f$. The area element can be
 * computed via `gr::surfaces::area_element()`.
 */
template <typename Frame>
double surface_integral_of_scalar(const Scalar<DataVector>& area_element,
                                  const Scalar<DataVector>& scalar,
                                  const Strahlkorper<Frame>& strahlkorper);
}  // namespace gr::surfaces
