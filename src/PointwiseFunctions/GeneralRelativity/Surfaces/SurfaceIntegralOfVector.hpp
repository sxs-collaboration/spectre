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
 * \brief Euclidean surface integral of a vector on a 2D `Strahlkorper`
 *
 * \details Computes the surface integral
 * \f$\oint V^i s_i (s_j s_k \delta^{jk})^{-1/2} d^2S\f$ for a
 * vector \f$V^i\f$ on a `Strahlkorper` with area element \f$d^2S\f$ and
 * normal one-form \f$s_i\f$.  Here \f$\delta^{ij}\f$ is the Euclidean
 * metric (i.e. the Kronecker delta). Note that the input `normal_one_form`
 * is not assumed to be normalized; the denominator of the integrand
 * effectively normalizes it using the Euclidean metric.
 * The area element can be computed via
 * `gr::surfaces::euclidean_area_element()`.
 */
template <typename Frame>
double euclidean_surface_integral_of_vector(
    const Scalar<DataVector>& area_element,
    const tnsr::I<DataVector, 3, Frame>& vector,
    const tnsr::i<DataVector, 3, Frame>& normal_one_form,
    const ylm::Strahlkorper<Frame>& strahlkorper);
}  // namespace gr::surfaces
