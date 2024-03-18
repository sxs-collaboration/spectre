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
}  // namespace CurvedScalarWave::Worldtube
