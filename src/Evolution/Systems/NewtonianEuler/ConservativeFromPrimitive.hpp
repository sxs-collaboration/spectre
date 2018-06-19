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

/*!
 * \brief Compute the conservative variables from the primitive variables.
 *
 * \f{align*}
 * S^i &= \rho v^i \\
 * e &= \dfrac{1}{2}\rho v^2 + \rho\epsilon
 * \f}
 *
 * where \f$S^i\f$ is the momentum density, \f$e\f$ is the energy density,
 * \f$\rho\f$ is the mass density, \f$v^i\f$ is the velocity, \f$v^2\f$ is its
 * magnitude squared, and \f$\epsilon\f$ is the specific internal energy.
 */
template <size_t Dim, typename DataType>
void conservative_from_primitive(
    gsl::not_null<tnsr::I<DataType, Dim>*> momentum_density,
    gsl::not_null<Scalar<DataType>*> energy_density,
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim>& velocity,
    const Scalar<DataType>& specific_internal_energy) noexcept;

}  // namespace NewtonianEuler
