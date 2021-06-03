// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler {
/// @{
/*!
 * Compute the kinetic energy density, \f$\rho v^2/2\f$,
 * where \f$\rho\f$ is the mass density, and \f$v\f$ is the
 * magnitude of the velocity.
 */
template <typename DataType, size_t Dim, typename Fr>
void kinetic_energy_density(
    gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept;

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> kinetic_energy_density(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept;
/// @}

namespace Tags {
/// Compute item for the kinetic energy density, \f$\rho v^2/2\f$.
/// \see NewtonianEuler::kinetic_energy_density
///
/// Can be retrieved using `NewtonianEuler::Tags::KineticEnergyDensity`
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct KineticEnergyDensityCompute : KineticEnergyDensity<DataType>,
                                     db::ComputeTag {
  using base = KineticEnergyDensity<DataType>;

  using argument_tags =
      tmpl::list<MassDensity<DataType>, Velocity<DataType, Dim, Fr>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function = static_cast<void (*)(
      const gsl::not_null<Scalar<DataType>*>, const Scalar<DataType>&,
      const tnsr::I<DataType, Dim, Fr>&)>(
      &kinetic_energy_density<DataType, Dim, Fr>);
};
}  // namespace Tags
}  // namespace NewtonianEuler
