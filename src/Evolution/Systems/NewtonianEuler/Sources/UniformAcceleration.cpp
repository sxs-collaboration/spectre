// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/Sources/UniformAcceleration.hpp"

#include <array>
#include <cstddef>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace NewtonianEuler::Sources {
template <size_t Dim>
UniformAcceleration<Dim>::UniformAcceleration(
    const std::array<double, Dim>& acceleration_field)
    : acceleration_field_(acceleration_field) {}

template <size_t Dim>
UniformAcceleration<Dim>::UniformAcceleration(CkMigrateMessage* msg)
    : Source<Dim>{msg} {}

template <size_t Dim>
void UniformAcceleration<Dim>::pup(PUP::er& p) {
  Source<Dim>::pup(p);
  p | acceleration_field_;
}

template <size_t Dim>
auto UniformAcceleration<Dim>::get_clone() const
    -> std::unique_ptr<Source<Dim>> {
  return std::make_unique<UniformAcceleration>(*this);
}

template <size_t Dim>
void UniformAcceleration<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> /*source_mass_density_cons*/,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> source_momentum_density,
    const gsl::not_null<Scalar<DataVector>*> source_energy_density,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& /*energy_density*/,
    const tnsr::I<DataVector, Dim>& /*velocity*/,
    const Scalar<DataVector>& /*pressure*/,
    const Scalar<DataVector>& /*specific_internal_energy*/,
    const EquationsOfState::EquationOfState<false, 2>& /*eos*/,
    const tnsr::I<DataVector, Dim>& /*coords*/, const double /*time*/) const {
  for (size_t i = 0; i < Dim; ++i) {
    source_momentum_density->get(i) +=
        get(mass_density_cons) * gsl::at(acceleration_field_, i);
    get(*source_energy_density) +=
        gsl::at(acceleration_field_, i) * momentum_density.get(i);
  }
}

template <size_t Dim>
// NOLINTNEXTLINE(cppcoreguidelines-avoid-non-const-global-variables)
PUP::able::PUP_ID UniformAcceleration<Dim>::my_PUP_ID = 0;

template <size_t Dim>
bool operator==(const UniformAcceleration<Dim>& lhs,
                const UniformAcceleration<Dim>& rhs) {
  return lhs.acceleration_field_ == rhs.acceleration_field_;
}

template <size_t Dim>
bool operator!=(const UniformAcceleration<Dim>& lhs,
                const UniformAcceleration<Dim>& rhs) {
  return not(lhs == rhs);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                       \
  template struct UniformAcceleration<DIM(data)>;                  \
  template bool operator==(const UniformAcceleration<DIM(data)>&,  \
                           const UniformAcceleration<DIM(data)>&); \
  template bool operator!=(const UniformAcceleration<DIM(data)>&,  \
                           const UniformAcceleration<DIM(data)>&);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace NewtonianEuler::Sources
