// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/IsentropicVortexSource.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace NewtonianEuler {
namespace Sources {

void IsentropicVortexSource::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density_source,
    const gsl::not_null<tnsr::I<DataVector, 3>*> momentum_density_source,
    const gsl::not_null<Scalar<DataVector>*> energy_density_source,
    const Scalar<DataVector>& vortex_mass_density,
    const tnsr::I<DataVector, 3>& vortex_momentum_density,
    const Scalar<DataVector>& vortex_energy_density,
    const Scalar<DataVector>& vortex_pressure,
    const Scalar<DataVector>& vortex_velocity_z,
    const Scalar<DataVector>& dz_vortex_velocity_z) noexcept {
  get(*mass_density_source) =
      get(vortex_mass_density) * get(dz_vortex_velocity_z);

  for (size_t i = 0; i < 3; ++i) {
    momentum_density_source->get(i) =
        vortex_momentum_density.get(i) * get(dz_vortex_velocity_z);
  }
  momentum_density_source->get(2) *= 2.0;

  get(*energy_density_source) =
      (get(vortex_energy_density) + get(vortex_pressure) +
       vortex_momentum_density.get(2) * get(vortex_velocity_z)) *
      get(dz_vortex_velocity_z);
}

}  // namespace Sources
}  // namespace NewtonianEuler

/// \endcond
