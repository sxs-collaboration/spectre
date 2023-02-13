// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro {

/*!
 * \brief The total mass-energy density measured by a normal observer, $E = n_a
 * n_b T^{ab}$
 *
 * This quantity sources the gravitational field equations in the 3+1
 * decomposition (see Eq. (2.138) in \cite BaumgarteShapiro).
 *
 * Perfect fluid contribution (Eq. (5.33) in \cite BaumgarteShapiro):
 *
 * \begin{equation}
 *   E_\mathrm{fluid} = \rho h W^2 - p
 * \end{equation}
 *
 * Magnetic field contribution (Eq. (5.152) in \cite BaumgarteShapiro):
 *
 * \begin{equation}
 *   E_\mathrm{em} = b^2 \left(W^2 - \frac{1}{2}\right) - (\alpha b^t)^2
 * \end{equation}
 *
 * where $\alpha b^t = W B^k v_k$.
 *
 * \param result Output buffer. Will be resized if needed.
 * \param rest_mass_density $\rho$
 * \param specific_enthalpy $h$
 * \param pressure $p$
 * \param lorentz_factor $W$
 * \param magnetic_field_dot_spatial_velocity $B^k v_k$
 * \param comoving_magnetic_field_squared $b^2$
 *
 * \see gr::Tags::EnergyDensity
 */
template <typename DataType>
void energy_density(gsl::not_null<Scalar<DataType>*> result,
                    const Scalar<DataType>& rest_mass_density,
                    const Scalar<DataType>& specific_enthalpy,
                    const Scalar<DataType>& pressure,
                    const Scalar<DataType>& lorentz_factor,
                    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
                    const Scalar<DataType>& comoving_magnetic_field_squared);

/*!
 * \brief The spatial momentum density $S^i = -\gamma^{ij} n^a T_{aj}$
 *
 * This quantity sources the gravitational field equations in the 3+1
 * decomposition (see Eq. (2.138) in \cite BaumgarteShapiro).
 *
 * Perfect fluid contribution (Eq. (5.34) in \cite BaumgarteShapiro):
 *
 * \begin{equation}
 *   S^i_\mathrm{fluid} = \rho h W^2 v^i
 * \end{equation}
 *
 * Magnetic field contribution (Eq. (5.153) in \cite BaumgarteShapiro):
 *
 * \begin{equation}
 *   S^i_\mathrm{em} = b^2 W^2 v^i - \alpha b^t \gamma^{ij} b_j
 * \end{equation}
 *
 * where $\alpha b^t \gamma^{ij} b_j = B^k v_k B^i + (B^k v_k)^2 W^2 v^i$.
 *
 * \param result Output buffer. Will be resized if needed.
 * \param rest_mass_density $\rho$
 * \param specific_enthalpy $h$
 * \param spatial_velocity $v^i$
 * \param lorentz_factor $W$
 * \param magnetic_field $B^i$
 * \param magnetic_field_dot_spatial_velocity $B^k v_k$
 * \param comoving_magnetic_field_squared $b^2$
 *
 * \see gr::Tags::MomentumDensity
 */
template <typename DataType>
void momentum_density(
    gsl::not_null<tnsr::I<DataType, 3>*> result,
    const Scalar<DataType>& rest_mass_density,
    const Scalar<DataType>& specific_enthalpy,
    const tnsr::I<DataType, 3>& spatial_velocity,
    const Scalar<DataType>& lorentz_factor,
    const tnsr::I<DataType, 3>& magnetic_field,
    const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
    const Scalar<DataType>& comoving_magnetic_field_squared);

/*!
 * \brief The trace of the spatial stress tensor, $S =
 * \gamma^{ij}\gamma_{ia}\gamma_{jb}T^{ab}$
 *
 * This quantity sources the gravitational field equations in the 3+1
 * decomposition (see Eq. (2.138) in \cite BaumgarteShapiro).
 *
 * Perfect fluid contribution (Eq. (5.36) in \cite BaumgarteShapiro):
 *
 * \begin{equation}
 *   S_\mathrm{fluid} = 3 p + \rho h (W^2 - 1)
 * \end{equation}
 *
 * Magnetic field contribution (Eq. (5.155) in \cite BaumgarteShapiro):
 *
 * \begin{equation}
 *   S_\mathrm{em} = b^2 (W^2 v^2 + \frac{3}{2}) - \gamma^{ij} b_i b_j
 * \end{equation}
 *
 * where $\gamma^{ij} b_i b_j = b^2 + (B^k v_k)^2 (W^2 v^2 + 1)$.
 *
 * \param result Output buffer. Will be resized if needed.
 * \param rest_mass_density $\rho$
 * \param specific_enthalpy $h$
 * \param pressure $p$
 * \param spatial_velocity_squared $v^2 = \gamma_{ij} v^i v^j$
 * \param lorentz_factor $W$
 * \param magnetic_field_dot_spatial_velocity $B^k v_k$
 * \param comoving_magnetic_field_squared $b^2$
 *
 * \see gr::Tags::StressTrace
 */
template <typename DataType>
void stress_trace(gsl::not_null<Scalar<DataType>*> result,
                  const Scalar<DataType>& rest_mass_density,
                  const Scalar<DataType>& specific_enthalpy,
                  const Scalar<DataType>& pressure,
                  const Scalar<DataType>& spatial_velocity_squared,
                  const Scalar<DataType>& lorentz_factor,
                  const Scalar<DataType>& magnetic_field_dot_spatial_velocity,
                  const Scalar<DataType>& comoving_magnetic_field_squared);

}  // namespace hydro
