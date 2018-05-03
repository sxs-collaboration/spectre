// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace NewtonianEuler {

/*!
 * \brief Compute the characteristic speeds of NewtonianEuler system
 *
 * The principal symbol of the system is diagonalized so that the elements of
 * the diagonal matrix are the characteristic speeds
 *
 * \f{align*}
 * \lambda_1 &= v_n - c_s,\\
 * \lambda_{i + 1} &= v_n,\\
 * \lambda_{\text{Dim} + 2} &= v_n + c_s,
 * \f}
 *
 * where \f$i = 1,...,\text{Dim}\f$,
 * \f$v_n = n_i v^i\f$ is the velocity projected onto the normal,
 * and \f$c_s\f$ is the sound speed.
 */
template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept;

}  // namespace NewtonianEuler
