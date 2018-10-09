// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"              // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Overloader.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

/// \cond

namespace {
std::array<DataVector, 9> compute_characteristic_speeds(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& alfven_speed_squared,
    const tnsr::i<DataVector, 3>& normal) noexcept {
  auto characteristic_speeds =
      make_array<9, DataVector>(-1.0 * get(dot_product(normal, shift)));

  const auto auxiliary_speeds =
      RelativisticEuler::Valencia::characteristic_speeds(
          lapse, shift, spatial_velocity, spatial_velocity_squared,
          Scalar<DataVector>{get(sound_speed_squared) +
                             get(alfven_speed_squared) *
                                 (1.0 - get(sound_speed_squared))},
          normal);

  characteristic_speeds[0] -= get(lapse);
  characteristic_speeds[1] = auxiliary_speeds[0];
  for (size_t i = 2; i < 7; ++i) {
    // auxiliary_speeds[1], auxiliary_speeds[2], and auxiliary_speeds[3]
    // are the same.
    gsl::at(characteristic_speeds, i) = auxiliary_speeds[2];
  }
  characteristic_speeds[7] = auxiliary_speeds[4];
  characteristic_speeds[8] += get(lapse);

  return characteristic_speeds;
}
}  // namespace

namespace grmhd {
namespace ValenciaDivClean {
template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) noexcept {
  const auto spatial_velocity_one_form =
      raise_or_lower_index(spatial_velocity, spatial_metric);
  const auto magnetic_field_one_form =
      raise_or_lower_index(magnetic_field, spatial_metric);
  const auto magnetic_field_dot_spatial_velocity =
      dot_product(magnetic_field, spatial_velocity_one_form);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity_one_form);
  // non const to save allocation later
  auto magnetic_field_squared =
      dot_product(magnetic_field, magnetic_field_one_form);
  const DataVector comoving_magnetic_field_squared =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));

  Scalar<DataVector> alfven_speed_squared = std::move(magnetic_field_squared);
  get(alfven_speed_squared) = comoving_magnetic_field_squared /
                              (comoving_magnetic_field_squared +
                               get(rest_mass_density) * get(specific_enthalpy));

  Scalar<DataVector> sound_speed_squared = make_overloader(
      [&rest_mass_density](const EquationsOfState::EquationOfState<true, 1>&
                               the_equation_of_state) noexcept {
        return Scalar<DataVector>{
            get(the_equation_of_state.chi_from_density(rest_mass_density)) +
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density(
                        rest_mass_density))};
      },
      [&rest_mass_density, &specific_internal_energy ](
          const EquationsOfState::EquationOfState<true, 2>&
              the_equation_of_state) noexcept {
        return Scalar<DataVector>{
            get(the_equation_of_state.chi_from_density_and_energy(
                rest_mass_density, specific_internal_energy)) +
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density_and_energy(
                        rest_mass_density, specific_internal_energy))};
      })(equation_of_state);
  get(sound_speed_squared) /= get(specific_enthalpy);

  return compute_characteristic_speeds(
      lapse, shift, spatial_velocity, spatial_velocity_squared,
      sound_speed_squared, alfven_speed_squared, unit_normal);
}
}  // namespace ValenciaDivClean
}  // namespace grmhd

template struct grmhd::ValenciaDivClean::ComputeCharacteristicSpeeds<1>;
template struct grmhd::ValenciaDivClean::ComputeCharacteristicSpeeds<2>;
/// \endcond
