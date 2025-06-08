// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace ylm {
template <typename Frame>
class Strahlkorper;
}  // namespace ylm
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
                                  const ylm::Strahlkorper<Frame>& strahlkorper);

/*!
 * \ingroup SurfacesGroup
 * \brief Surface average of a scalar on a 2D `Strahlkorper`
 *
 */
template <typename Frame>
double surface_average_of_scalar(const Scalar<DataVector>& area_element,
                                 const Scalar<DataVector>& scalar,
                                 const ylm::Strahlkorper<Frame>& strahlkorper);
}  // namespace gr::surfaces
