// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/VortexPerturbation.hpp"

#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/NewtonianEuler/ConservativeFromPrimitive.hpp"
#include "Evolution/Systems/NewtonianEuler/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/NewtonianEuler/IsentropicVortex.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace NewtonianEuler::Sources {
VortexPerturbation::VortexPerturbation(const double perturbation_amplitude)
    : perturbation_amplitude_(perturbation_amplitude) {}

VortexPerturbation::VortexPerturbation(CkMigrateMessage* msg) : Source{msg} {}

void VortexPerturbation::pup(PUP::er& p) {
  Source::pup(p);
  p | perturbation_amplitude_;
}

auto VortexPerturbation::get_clone() const -> std::unique_ptr<Source<3>> {
  return std::make_unique<VortexPerturbation>(*this);
}

void VortexPerturbation::operator()(
    const gsl::not_null<Scalar<DataVector>*> source_mass_density_cons,
    const gsl::not_null<tnsr::I<DataVector, 3>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, 3>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const tnsr::I<DataVector, 3>& velocity, const Scalar<DataVector>& pressure,
    const Scalar<DataVector>& /*specific_internal_energy*/,
    const EquationsOfState::EquationOfState<false, 2>& /*eos*/,
    const tnsr::I<DataVector, 3>& coords, const double /*time*/) const {
  const Scalar<DataVector> dvz_by_dz{perturbation_amplitude_ *
                                     cos(get<2>(coords))};

  get(*source_mass_density_cons) += get(mass_density_cons) * get(dvz_by_dz);
  for (size_t i = 0; i < 3; ++i) {
    source_momentum_density->get(i) += momentum_density.get(i) * get(dvz_by_dz);
  }
  get(*source_energy_density) += (get(energy_density) + get(pressure) +
                                  get<2>(velocity) * get<2>(momentum_density)) *
                                 get(dvz_by_dz);
}

// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID VortexPerturbation::my_PUP_ID = 0;
}  // namespace NewtonianEuler::Sources
