// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/VariableFixing/RadiallyFallingFloor.hpp"

#include <algorithm>
#include <pup.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace VariableFixing {

template <size_t Dim>
RadiallyFallingFloor<Dim>::RadiallyFallingFloor(
    const double minimum_radius_at_which_to_apply_floor,
    const double rest_mass_density_scale, const double rest_mass_density_power,
    const double pressure_scale, const double pressure_power)
    : minimum_radius_at_which_to_apply_floor_(
          minimum_radius_at_which_to_apply_floor),
      rest_mass_density_scale_(rest_mass_density_scale),
      rest_mass_density_power_(rest_mass_density_power),
      pressure_scale_(pressure_scale),
      pressure_power_(pressure_power) {}

template <size_t Dim>
void RadiallyFallingFloor<Dim>::pup(PUP::er& p) {  // NOLINT
  p | minimum_radius_at_which_to_apply_floor_;
  p | rest_mass_density_scale_;
  p | rest_mass_density_power_;
  p | pressure_scale_;
  p | pressure_power_;
}

template <size_t Dim>
template <size_t ThermodynamicDim>
void RadiallyFallingFloor<Dim>::operator()(
    const gsl::not_null<Scalar<DataVector>*> density,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
    const gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
    [[maybe_unused]] const gsl::not_null<Scalar<DataVector>*> temperature,
    [[maybe_unused]] const gsl::not_null<Scalar<DataVector>*> electron_fraction,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& coords,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) const {
  const auto radii = magnitude(coords);

  for (size_t i = 0; i < density->get().size(); i++) {
    const double& radius = radii.get()[i];
    if (UNLIKELY(radius < minimum_radius_at_which_to_apply_floor_)) {
      continue;
    }
    pressure->get()[i] = std::max(
        pressure->get()[i], pressure_scale_ * pow(radius, pressure_power_));

    density->get()[i] =
        std::max(density->get()[i], rest_mass_density_scale_ *
                                        pow(radius, rest_mass_density_power_));
    if constexpr (ThermodynamicDim == 1) {
      // For the 1D EoS, density will be used to calculate the remaining
      // primitives, even overwriting pressure.  This decision prioritizes
      // thermodynamic consistency over the user defined pressure profile.
      pressure->get()[i] = get(equation_of_state.pressure_from_density(
          Scalar<double>{density->get()[i]}));

      specific_internal_energy->get()[i] =
          get(equation_of_state.specific_internal_energy_from_density(
              Scalar<double>{density->get()[i]}));

      temperature->get()[i] = get(equation_of_state.temperature_from_density(
          Scalar<double>{density->get()[i]}));

    } else if constexpr (ThermodynamicDim == 2) {
      specific_internal_energy->get()[i] = get(
          equation_of_state.specific_internal_energy_from_density_and_pressure(
              Scalar<double>{density->get()[i]},
              Scalar<double>{pressure->get()[i]}));

      temperature->get()[i] =
          get(equation_of_state.temperature_from_density_and_energy(
              Scalar<double>{density->get()[i]},
              Scalar<double>{specific_internal_energy->get()[i]}));

    } else if constexpr (ThermodynamicDim == 3) {
      ERROR("RadiallyFallingFloor: 3D EoS currently not supported");
      // For posterity: 3D EoS
      // // We need to assume either specific internal energy or temperature
      // remain
      // // unchanged.  We choose temperature remaining the same.  A choice
      // needs
      // // to be made b/c no energy/temperature_from_density_and_pressure()
      // call
      // // exists, yet.
      // specific_internal_energy->get()[i] =
      //     get(equation_of_state
      //             .specific_internal_energy_from_density_and_temperature(
      //                 Scalar<double>{density->get()[i]},
      //                 Scalar<double>{temperature->get()[i]},
      //                 Scalar<double>{electron_fraction->get()[i]}));

      // // B/c temperature is assumed to be constant, we need to call pressure
      // EoS
      // // one more time
      // pressure->get()[i] =
      //     get(equation_of_state.pressure_from_density_and_energy(
      //         Scalar<double>{density->get()[i]},
      //         Scalar<double>{specific_internal_energy->get()[i]},
      //         Scalar<double>{electron_fraction->get()[i]}));

    } else {
      ERROR("RadiallyFallingFloor: Must specify 1D, 2D, or 3D EoS");
    }
    // Assumed relativistic EoS
    specific_enthalpy->get()[i] = 1.0 + specific_internal_energy->get()[i] +
                                  pressure->get()[i] / density->get()[i];
  }
}

template <size_t LocalDim>
bool operator==(const RadiallyFallingFloor<LocalDim>& lhs,
                const RadiallyFallingFloor<LocalDim>& rhs) {
  return lhs.minimum_radius_at_which_to_apply_floor_ ==
             rhs.minimum_radius_at_which_to_apply_floor_ and
         lhs.rest_mass_density_scale_ == rhs.rest_mass_density_scale_ and
         lhs.rest_mass_density_power_ == rhs.rest_mass_density_power_ and
         lhs.pressure_scale_ == rhs.pressure_scale_ and
         lhs.pressure_power_ == rhs.pressure_power_;
}

template <size_t Dim>
bool operator!=(const RadiallyFallingFloor<Dim>& lhs,
                const RadiallyFallingFloor<Dim>& rhs) {
  return not(lhs == rhs);
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template class RadiallyFallingFloor<GET_DIM(data)>;                       \
  template bool operator==(const RadiallyFallingFloor<GET_DIM(data)>& lhs,  \
                           const RadiallyFallingFloor<GET_DIM(data)>& rhs); \
  template bool operator!=(const RadiallyFallingFloor<GET_DIM(data)>& lhs,  \
                           const RadiallyFallingFloor<GET_DIM(data)>& rhs);
GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef INSTANTIATION

#define THERMO_DIM(data) BOOST_PP_TUPLE_ELEM(1, data)
#define INSTANTIATION(r, data)                                           \
  template void RadiallyFallingFloor<GET_DIM(data)>::operator()(         \
      gsl::not_null<Scalar<DataVector>*> density,                        \
      gsl::not_null<Scalar<DataVector>*> pressure,                       \
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,       \
      gsl::not_null<Scalar<DataVector>*> specific_enthalpy,              \
      gsl::not_null<Scalar<DataVector>*> temperature,                    \
      gsl::not_null<Scalar<DataVector>*> electron_fraction,              \
      const tnsr::I<DataVector, GET_DIM(data), Frame::Inertial>& coords, \
      const EquationsOfState::EquationOfState<true, THERMO_DIM(data)>&   \
          equation_of_state) const;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3), (1, 2, 3))
#undef INSTANTIATION
#undef THERMO_DIM
#undef GET_DIM

}  // namespace VariableFixing
