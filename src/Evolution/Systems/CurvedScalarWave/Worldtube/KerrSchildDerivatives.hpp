// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {
/*!
 * \brief The spatial derivative of the zero spin inverse Kerr Schild metric,
 * $\partial_i g^{\mu \nu}$, assuming a black hole at the coordinate center with
 * mass M = 1.
 */
tnsr::iAA<double, 3> spatial_derivative_inverse_ks_metric(
    const tnsr::I<double, 3>& pos);

/*!
 * \brief The spatial derivative of the spacetime metric,
 * $\partial_i g_{\mu \nu}$.
 */
tnsr::iaa<double, 3> spatial_derivative_ks_metric(
    const tnsr::aa<double, 3>& metric,
    const tnsr::iAA<double, 3>& di_inverse_metric);

/*!
 * \brief The second spatial derivative of the zero spin inverse Kerr Schild
 * metric, $\partial_i \partial_j g^{\mu \nu}$, assuming a black hole at the
 * coordinate center with mass M = 1.
 */
tnsr::iiAA<double, 3> second_spatial_derivative_inverse_ks_metric(
    const tnsr::I<double, 3>& pos);
}  // namespace CurvedScalarWave::Worldtube
