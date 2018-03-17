// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace NewtonianEuler {
// @{
/*!
 * \brief Compute the velocity from the conservative variables.
 */
template <size_t Dim, typename DataType>
void velocity(gsl::not_null<tnsr::I<DataType, Dim>*> velocity,
              const Scalar<DataType>& mass_density,
              const tnsr::I<DataType, Dim>& momentum_density) noexcept;

template <size_t Dim, typename DataType>
tnsr::I<DataType, Dim> velocity(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& momentum_density) noexcept;
// @}

/*!
 * \brief Compute the primitive variables from the conservative variables.
 *
 * \f{align*}
 * v^i &= \frac{S^i}{\rho} \\
 * \epsilon &= \frac{e}{\rho} - \frac{1}{2}\frac{S^2}{\rho^2}
 * \f}
 *
 * where \f$v^i\f$ is the velocity, \f$\epsilon\f$ is the specific
 * internal energy, \f$e\f$ is the energy density, \f$\rho\f$
 * is the mass density, \f$S^i\f$ is the momentum density, and
 * \f$S^2\f$ is the momentum density squared.
 */
template <size_t Dim, typename DataType>
void primitive_from_conservative(
    gsl::not_null<tnsr::I<DataType, Dim>*> velocity,
    gsl::not_null<Scalar<DataType>*> specific_internal_energy,
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& momentum_density,
    const Scalar<DataType>& energy_density) noexcept;

}  // namespace NewtonianEuler
