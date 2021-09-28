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
 * Compute the internal energy density, \f$\rho \epsilon\f$,
 * where \f$\rho\f$ is the mass density, and \f$\epsilon\f$ is the
 * specific internal energy.
 */
template <typename DataType>
void internal_energy_density(gsl::not_null<Scalar<DataType>*> result,
                             const Scalar<DataType>& mass_density,
                             const Scalar<DataType>& specific_internal_energy);

template <typename DataType>
Scalar<DataType> internal_energy_density(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy);
/// @}

namespace Tags {
/// Compute item for the internal energy density, \f$\rho \epsilon\f$.
/// \see NewtonianEuler::internal_energy_density
///
/// Can be retrieved using `NewtonianEuler::Tags::InternalEnergyDensity`
template <typename DataType>
struct InternalEnergyDensityCompute : InternalEnergyDensity<DataType>,
                                      db::ComputeTag {
  using base = InternalEnergyDensity<DataType>;

  using argument_tags =
      tmpl::list<MassDensity<DataType>, SpecificInternalEnergy<DataType>>;

  using return_type = Scalar<DataType>;

  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<Scalar<DataType>*>,
                           const Scalar<DataType>&, const Scalar<DataType>&)>(
          &internal_energy_density<DataType>);
};
}  // namespace Tags
}  // namespace NewtonianEuler
