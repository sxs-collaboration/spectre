// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/LaneEmdenGravitationalField.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/LaneEmdenStar.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare Tensor

namespace NewtonianEuler {
namespace Sources {

void LaneEmdenGravitationalField::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, 3>& momentum_density,
    const NewtonianEuler::Solutions::LaneEmdenStar& star,
    const tnsr::I<DataVector, 3>& x) noexcept {
  const auto gravitational_field = star.gravitational_field(x);
  get(*source_energy_density) = 0.0;
  for (size_t i = 0; i < 3; ++i) {
    source_momentum_density->get(i) =
        get(mass_density_cons) * gravitational_field.get(i);
    get(*source_energy_density) +=
        momentum_density.get(i) * gravitational_field.get(i);
  }
}

}  // Namespace Sources
}  // namespace NewtonianEuler
