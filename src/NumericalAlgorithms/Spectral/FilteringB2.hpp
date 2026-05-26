// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/Gsl.hpp"

/// \cond
class Matrix;
template <size_t>
class Mesh;
template <typename TagsList>
class Variables;
/// \endcond

namespace Spectral::filtering {
/*!
 * \brief Filters the tensors stored within a `Variables` being represented by
 * ZernikeB2 basis functions.
 *
 * \details Representing functions on a filled disk requires special basis
 * functions, namely ZernikeB2. These are inherently two-dimensional, meaning
 * the radial and angular spectral spaces are intertwined. This function goes
 * to that combined modal space, applies the exponential filter, and
 * transforms back.
 *
 * \see exponential_filter()
 */
template <typename TagsList>
void zernike_b2_disk_exponential_filter(gsl::not_null<Variables<TagsList>*> u,
                                       const Mesh<2>& mesh, double alpha,
                                       unsigned half_power);

/*!
 * \brief Filters the tensors stored within a `Variables` being represented by
 * ZernikeB2 \f$\times\f$ Legendre basis functions.
 *
 * \details Representing functions on a filled cylinder requires special basis
 * functions, namely a filled disk with ZernikeB2 cross Legendre. This
 * requires inherently two-dimensional basis functions, meaning the radial and
 * angular spectral spaces are intertwined. This function goes to that combined
 * modal space, applies the exponential filter, transforms back, and then
 * filters the third I1 dimension.
 *
 * \see exponential_filter()
 */
template <typename TagsList>
void zernike_b2_cylinder_exponential_filter(
    gsl::not_null<Variables<TagsList>*> u, const Mesh<3>& mesh, double alpha,
    unsigned half_power);
}  // namespace Spectral::filtering
