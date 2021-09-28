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
 * Compute the specific kinetic energy, \f$v^2/2\f$,
 * where \f$v\f$ is the magnitude of the velocity.
 */
template <typename DataType, size_t Dim, typename Fr>
void specific_kinetic_energy(gsl::not_null<Scalar<DataType>*> result,
                             const tnsr::I<DataType, Dim, Fr>& velocity);

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> specific_kinetic_energy(
    const tnsr::I<DataType, Dim, Fr>& velocity);
/// @}

namespace Tags {
/// Compute item for the specific kinetic energy, \f$v^2/2\f$.
/// \see NewtonianEuler::specific_kinetic_energy
///
/// Can be retrieved using `NewtonianEuler::Tags::SpecificKineticEnergy`
template <typename DataType, size_t Dim, typename Fr = Frame::Inertial>
struct SpecificKineticEnergyCompute : SpecificKineticEnergy<DataType>,
                                      db::ComputeTag {
  using base = SpecificKineticEnergy<DataType>;

  using argument_tags = tmpl::list<Velocity<DataType, Dim, Fr>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<Scalar<DataType>*>,
                           const tnsr::I<DataType, Dim, Fr>&)>(
          &specific_kinetic_energy<DataType, Dim, Fr>);
};
}  // namespace Tags
}  // namespace NewtonianEuler
