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
 * ZernikeB1 basis functions in their first dimension.
 *
 * \details The Cartoon method with spectral bases can be unstable for small
 * \f$x\f$, so we use ZernikeB1 bases with GaussRadauUpper quadrature to push
 * collocation points to higher inertial coordinates. This basis has
 * parity-dependent spectral space, so we must go to the proper modal space,
 * apply the exponential filter, and transform back.
 *
 * \see exponential_filter()
 */
template <typename VariablesTags>
void zernike_b1_exponential_filter(gsl::not_null<Variables<VariablesTags>*> u,
                                   const Mesh<3>& mesh, double alpha,
                                   unsigned half_power);
}  // namespace Spectral::filtering
