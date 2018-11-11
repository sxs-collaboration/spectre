// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
class DataVector;
namespace gsl {
template <class>
class not_null;
}  // namespace gsl
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace RelativisticEuler {
namespace Valencia {

/*!
 * \brief Compute the characteristic speeds for the Valencia formulation
 * of the relativistic Euler system.
 *
 * The principal symbol of the system is diagonalized so that the elements of
 * the diagonal matrix are the \f$(\text{Dim} + 2)\f$ characteristic speeds
 *
 * \f{align*}
 * \lambda_1 &= \alpha \Lambda^- - \beta_n,\\
 * \lambda_{i + 1} &= \alpha v_n - \beta_n,\quad i = 1,...,\text{Dim}\\
 * \lambda_{\text{Dim} + 2} &= \alpha \Lambda^+ - \beta_n,
 * \f}
 *
 * where \f$\alpha\f$ is the lapse, \f$\beta_n = n_i \beta^i\f$ and
 * \f$v_n = n_i v^i\f$ are the projections of the shift \f$\beta^i\f$ and the
 * spatial velocity \f$v^i\f$ onto the normal one-form \f$n_i\f$, respectively,
 * and
 *
 * \f{align*}
 * \Lambda^{\pm} &= \dfrac{1}{1 - v^2 c_s^2}\left[ v_n (1- c_s^2) \pm
 * c_s\sqrt{\left(1 - v^2\right)\left[1 - v^2 c_s^2 - v_n^2(1 - c_s^2)\right]}
 * \right],
 * \f}
 *
 * where \f$v^2 = \gamma_{ij}v^iv^j\f$ is the magnitude squared of the spatial
 * velocity, and \f$c_s\f$ is the sound speed.
 */
template <size_t Dim>
void characteristic_speeds(
    gsl::not_null<std::array<DataVector, Dim + 2>*> char_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept;

template <size_t Dim>
std::array<DataVector, Dim + 2> characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept;

}  // namespace Valencia
}  // namespace RelativisticEuler
