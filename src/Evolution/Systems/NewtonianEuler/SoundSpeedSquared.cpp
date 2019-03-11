// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"  // IWYU pragma: keep
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Overloader.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState

/// \cond
namespace NewtonianEuler {

template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> sound_speed_squared(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept {
  DataType sound_speed_squared{};
  make_overloader(
      [&mass_density,
       &sound_speed_squared ](const EquationsOfState::EquationOfState<false, 1>&
                                  the_equation_of_state) noexcept {
        sound_speed_squared =
            get(the_equation_of_state.chi_from_density(mass_density)) +
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density(mass_density));
      },
      [&mass_density, &specific_internal_energy,
       &sound_speed_squared ](const EquationsOfState::EquationOfState<false, 2>&
                                  the_equation_of_state) noexcept {
        sound_speed_squared =
            get(the_equation_of_state.chi_from_density_and_energy(
                mass_density, specific_internal_energy)) +
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density_and_energy(
                        mass_density, specific_internal_energy));
      })(equation_of_state);
  return Scalar<DataType>{sound_speed_squared};
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)
#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                                            \
  template Scalar<DTYPE(data)> sound_speed_squared(                     \
      const Scalar<DTYPE(data)>& mass_density,                          \
      const Scalar<DTYPE(data)>& specific_internal_energy,              \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& \
          equation_of_state) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (double, DataVector), (1, 2))

#undef DTYPE
#undef THERMO_DIM
#undef INSTANTIATE

}  // namespace NewtonianEuler
/// \endcond
