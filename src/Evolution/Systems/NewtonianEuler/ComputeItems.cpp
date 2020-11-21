// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/NewtonianEuler/ComputeItems.hpp"

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

/// \cond
namespace NewtonianEuler {

template <typename DataType>
void internal_energy_density(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy) noexcept {
  get(*result) = get(mass_density) * get(specific_internal_energy);
}

template <typename DataType>
Scalar<DataType> internal_energy_density(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy) noexcept {
  Scalar<DataType> result{};
  internal_energy_density(make_not_null(&result), mass_density,
                          specific_internal_energy);
  return result;
}

template <typename DataType, size_t Dim, typename Fr>
void kinetic_energy_density(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept {
  get(*result) = 0.5 * get(mass_density) * get(dot_product(velocity, velocity));
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> kinetic_energy_density(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept {
  Scalar<DataType> result{};
  kinetic_energy_density(make_not_null(&result), mass_density, velocity);
  return result;
}

template <typename DataType, size_t Dim, typename Fr>
void mach_number(const gsl::not_null<Scalar<DataType>*> result,
                 const tnsr::I<DataType, Dim, Fr>& velocity,
                 const Scalar<DataType>& sound_speed) noexcept {
  get(*result) = get(magnitude(velocity)) / get(sound_speed);
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> mach_number(const tnsr::I<DataType, Dim, Fr>& velocity,
                             const Scalar<DataType>& sound_speed) noexcept {
  Scalar<DataType> result{};
  mach_number(make_not_null(&result), velocity, sound_speed);
  return result;
}

template <typename DataType, size_t Dim, typename Fr>
void ram_pressure(const gsl::not_null<tnsr::II<DataType, Dim, Fr>*> result,
                  const Scalar<DataType>& mass_density,
                  const tnsr::I<DataType, Dim, Fr>& velocity) noexcept {
  for (size_t i = 0; i < Dim; ++i) {
    for (size_t j = i; j < Dim; ++j) {
      result->get(i, j) = get(mass_density) * velocity.get(i) * velocity.get(j);
    }
  }
}

template <typename DataType, size_t Dim, typename Fr>
tnsr::II<DataType, Dim, Fr> ram_pressure(
    const Scalar<DataType>& mass_density,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept {
  tnsr::II<DataType, Dim, Fr> result{};
  ram_pressure(make_not_null(&result), mass_density, velocity);
  return result;
}

template <typename DataType, size_t ThermodynamicDim>
void sound_speed_squared(
    const gsl::not_null<Scalar<DataType>*> result,
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept {
  destructive_resize_components(result, get_size(get(mass_density)));
  if constexpr (ThermodynamicDim == 1) {
    get(*result) =
        get(equation_of_state.chi_from_density(mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            mass_density));
  } else if constexpr (ThermodynamicDim == 2) {
    get(*result) =
        get(equation_of_state.chi_from_density_and_energy(
            mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    mass_density, specific_internal_energy));
  }
}

template <typename DataType, size_t ThermodynamicDim>
Scalar<DataType> sound_speed_squared(
    const Scalar<DataType>& mass_density,
    const Scalar<DataType>& specific_internal_energy,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept {
  Scalar<DataType> result{};
  sound_speed_squared(make_not_null(&result), mass_density,
                      specific_internal_energy, equation_of_state);
  return result;
}

template <typename DataType, size_t Dim, typename Fr>
void specific_kinetic_energy(
    const gsl::not_null<Scalar<DataType>*> result,
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept {
  get(*result) = 0.5 * get(dot_product(velocity, velocity));
}

template <typename DataType, size_t Dim, typename Fr>
Scalar<DataType> specific_kinetic_energy(
    const tnsr::I<DataType, Dim, Fr>& velocity) noexcept {
  Scalar<DataType> result{};
  specific_kinetic_energy(make_not_null(&result), velocity);
  return result;
}

#define DTYPE(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE_SCALAR_ARGS(_, data)                             \
  template void internal_energy_density(                             \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,              \
      const Scalar<DTYPE(data)>& mass_density,                       \
      const Scalar<DTYPE(data)>& specific_internal_energy) noexcept; \
  template Scalar<DTYPE(data)> internal_energy_density(              \
      const Scalar<DTYPE(data)>& mass_density,                       \
      const Scalar<DTYPE(data)>& specific_internal_energy) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SCALAR_ARGS, (double, DataVector))

#define DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_TNSR_ARGS(_, data)                                        \
  template void kinetic_energy_density(                                       \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,                       \
      const Scalar<DTYPE(data)>& mass_density,                                \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity) noexcept;              \
  template Scalar<DTYPE(data)> kinetic_energy_density(                        \
      const Scalar<DTYPE(data)>& mass_density,                                \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity) noexcept;              \
  template void mach_number(const gsl::not_null<Scalar<DTYPE(data)>*> result, \
                            const tnsr::I<DTYPE(data), DIM(data)>& velocity,  \
                            const Scalar<DTYPE(data)>& sound_speed) noexcept; \
  template Scalar<DTYPE(data)> mach_number(                                   \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity,                        \
      const Scalar<DTYPE(data)>& sound_speed) noexcept;                       \
  template void ram_pressure(                                                 \
      const gsl::not_null<tnsr::II<DTYPE(data), DIM(data)>*> result,          \
      const Scalar<DTYPE(data)>& mass_density,                                \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity) noexcept;              \
  template tnsr::II<DTYPE(data), DIM(data)> ram_pressure(                     \
      const Scalar<DTYPE(data)>& mass_density,                                \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity) noexcept;              \
  template void specific_kinetic_energy(                                      \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,                       \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity) noexcept;              \
  template Scalar<DTYPE(data)> specific_kinetic_energy(                       \
      const tnsr::I<DTYPE(data), DIM(data)>& velocity) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_TNSR_ARGS, (double, DataVector), (1, 2, 3))

#undef DIM

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE_SOUND_SPEED_SQUARED(_, data)                        \
  template void sound_speed_squared(                                    \
      const gsl::not_null<Scalar<DTYPE(data)>*> result,                 \
      const Scalar<DTYPE(data)>& mass_density,                          \
      const Scalar<DTYPE(data)>& specific_internal_energy,              \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& \
          equation_of_state) noexcept;                                  \
  template Scalar<DTYPE(data)> sound_speed_squared(                     \
      const Scalar<DTYPE(data)>& mass_density,                          \
      const Scalar<DTYPE(data)>& specific_internal_energy,              \
      const EquationsOfState::EquationOfState<false, THERMO_DIM(data)>& \
          equation_of_state) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE_SOUND_SPEED_SQUARED, (double, DataVector),
                        (1, 2))

#undef THERMO_DIM
#undef DTYPE

#undef INSTANTIATE_SOUND_SPEED_SQUARED
#undef INSTANTIATE_SCALAR_ARGS
#undef INSTANTIATE_TNSR_ARGS

}  // namespace NewtonianEuler
/// \endcond
