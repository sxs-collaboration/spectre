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
namespace gsl {
template <typename>
struct not_null;
}  // namespace gsl
/// \endcond

namespace gr::surfaces {
/*!
 * \ingroup SurfacesGroup
 * \brief Radial distance between two `Strahlkorper`s.
 *
 * \details Computes the pointwise radial distance \f$r_a-r_b\f$ between two
 * Strahlkorpers `strahlkorper_a` and `strahlkorper_b` that have the same
 * center, first (if the Strahlkorpers' resolutions are unequal) prolonging the
 * lower-resolution Strahlkorper to the same resolution as the higher-resolution
 * Strahlkorper.
 */
template <typename Frame>
void radial_distance(gsl::not_null<Scalar<DataVector>*> radial_distance,
                     const ylm::Strahlkorper<Frame>& strahlkorper_a,
                     const ylm::Strahlkorper<Frame>& strahlkorper_b);
}  // namespace gr::surfaces
