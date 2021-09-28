// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/PrimitiveFromConservative.hpp"

#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

namespace NewtonianEuler {

template <size_t Dim>
template <size_t ThermodynamicDim>
void PrimitiveFromConservative<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) {
  get(*mass_density) = get(mass_density_cons);

  // store inverse mass density here in order to save allocation
  get(*specific_internal_energy) = 1.0 / get(mass_density_cons);
  for (size_t i = 0; i < Dim; ++i) {
    velocity->get(i) = momentum_density.get(i) * get(*specific_internal_energy);
  }

  get(*specific_internal_energy) *= get(energy_density);
  get(*specific_internal_energy) -=
      (0.5 * get(dot_product(*velocity, *velocity)));

  if constexpr (ThermodynamicDim == 1) {
    *pressure = equation_of_state.pressure_from_density(mass_density_cons);
  } else if constexpr (ThermodynamicDim == 2) {
    *pressure = equation_of_state.pressure_from_density_and_energy(
        mass_density_cons, *specific_internal_energy);
  }
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(_, data) \
  template struct NewtonianEuler::PrimitiveFromConservative<DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATION(_, data)                                               \
  template void NewtonianEuler::PrimitiveFromConservative<DIM(data)>::apply< \
      THERMO_DIM(data)>(                                                     \
      const gsl::not_null<Scalar<DataVector>*> mass_density,                 \
      const gsl::not_null<tnsr::I<DataVector, DIM(data)>*> velocity,         \
      const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,     \
      const gsl::not_null<Scalar<DataVector>*> pressure,                     \
      const Scalar<DataVector>& mass_density_cons,                           \
      const tnsr::I<DataVector, DIM(data)>& momentum_density,                \
      const Scalar<DataVector>& energy_density,                              \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>&      \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2))

#undef INSTANTIATION
#undef THERMO_DIM
#undef DIM
