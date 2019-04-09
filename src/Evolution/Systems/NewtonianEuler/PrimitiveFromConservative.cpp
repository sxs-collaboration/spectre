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
#include "Utilities/Overloader.hpp"

// IWYU pragma: no_include <array>

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

/// \cond
namespace NewtonianEuler {

template <size_t Dim, size_t ThermodynamicDim>
void PrimitiveFromConservative<Dim, ThermodynamicDim>::apply(
    const gsl::not_null<Scalar<DataVector>*> mass_density,
    const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const Scalar<DataVector>& mass_density_cons,
    const tnsr::I<DataVector, Dim>& momentum_density,
    const Scalar<DataVector>& energy_density,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept {
  get(*mass_density) = get(mass_density_cons);
  const DataVector one_over_mass_density = 1.0 / get(mass_density_cons);

  for (size_t i = 0; i < Dim; ++i) {
    velocity->get(i) = momentum_density.get(i) * one_over_mass_density;
  }

  get(*specific_internal_energy) = one_over_mass_density * get(energy_density) -
                                   0.5 * get(dot_product(*velocity, *velocity));

  *pressure = make_overloader(
      [&mass_density_cons](const EquationsOfState::EquationOfState<false, 1>&
                               the_equation_of_state) noexcept {
        return the_equation_of_state.pressure_from_density(mass_density_cons);
      },
      [&mass_density_cons, &specific_internal_energy ](
          const EquationsOfState::EquationOfState<false, 2>&
              the_equation_of_state) noexcept {
        return the_equation_of_state.pressure_from_density_and_energy(
            mass_density_cons, *specific_internal_energy);
      })(equation_of_state);
}

}  // namespace NewtonianEuler

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                           \
  template struct NewtonianEuler::PrimitiveFromConservative<DIM(data), \
                                                            THERMO_DIM(data)>;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (1, 2))

#undef DIM
#undef THERMO_DIM
#undef INSTANTIATE
/// \endcond
