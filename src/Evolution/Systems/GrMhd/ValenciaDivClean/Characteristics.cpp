// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void compute_characteristic_speeds_approximate_mhd(
    const gsl::not_null<std::array<DataVector, 9>*> pchar_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& alfven_speed_squared,
    const tnsr::i<DataVector, 3>& normal) {
  const size_t num_grid_points = get(lapse).size();
  auto& char_speeds = *pchar_speeds;
  if (char_speeds[0].size() != num_grid_points) {
    char_speeds[0] = DataVector(num_grid_points);
  }
  // Mapping of indices between GRMHD char speeds and relativistic Euler char
  // speeds arrays.
  //
  // GRMHD     Rel Euler
  //   1           0
  //   2           1
  //   3           2
  //   4           3
  //   5           1
  //   6           1
  //   7           4
  Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::TempScalar<5>>>
      temp_tensors{num_grid_points};

  // Because we don't require char_speeds to be of the correct size we use a
  // temp buffer for the dot product, then multiply by -1 assigning the result
  // to char_speeds.
  {
    Scalar<DataVector>& normal_shift = get<::Tags::TempScalar<0>>(temp_tensors);
    dot_product(make_not_null(&normal_shift), normal, shift);
    char_speeds[0] = -1.0 * get(normal_shift);
    char_speeds[1] = char_speeds[0];
  }
  Scalar<DataVector>& scaled_sound_speed_squared =
      get<::Tags::TempScalar<5>>(temp_tensors);
  get(scaled_sound_speed_squared) =
      get(sound_speed_squared) +
      get(alfven_speed_squared) * (1.0 - get(sound_speed_squared));
  // Dim-fold degenerate eigenvalue, reuse normal_shift allocation
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), normal, spatial_velocity);
  char_speeds[2] = char_speeds[1] + get(lapse) * get(normal_velocity);
  char_speeds[3] = char_speeds[2];
  char_speeds[4] = char_speeds[3];
  char_speeds[5] = char_speeds[2];
  char_speeds[6] = char_speeds[2];

  Scalar<DataVector>& one_minus_v_sqrd_cs_sqrd =
      get<::Tags::TempScalar<1>>(temp_tensors);
  get(one_minus_v_sqrd_cs_sqrd) =
      1.0 - get(spatial_velocity_squared) * get(scaled_sound_speed_squared);
  Scalar<DataVector>& vn_times_one_minus_cs_sqrd =
      get<::Tags::TempScalar<2>>(temp_tensors);
  get(vn_times_one_minus_cs_sqrd) =
      get(normal_velocity) * (1.0 - get(scaled_sound_speed_squared));

  Scalar<DataVector>& first_term = get<::Tags::TempScalar<3>>(temp_tensors);
  get(first_term) = get(lapse) / get(one_minus_v_sqrd_cs_sqrd);
  Scalar<DataVector>& second_term = get<::Tags::TempScalar<4>>(temp_tensors);
  get(second_term) =
      get(first_term) * sqrt(get(scaled_sound_speed_squared)) *
      sqrt((1.0 - get(spatial_velocity_squared)) *
           (get(one_minus_v_sqrd_cs_sqrd) -
            get(normal_velocity) * get(vn_times_one_minus_cs_sqrd)));
  get(first_term) *= get(vn_times_one_minus_cs_sqrd);

  char_speeds[7] = char_speeds[1] + get(first_term) + get(second_term);
  char_speeds[1] += get(first_term) - get(second_term);

  char_speeds[8] = char_speeds[0] + get(lapse);
  char_speeds[0] -= get(lapse);
}
}  // namespace

namespace grmhd::ValenciaDivClean {
template <size_t ThermodynamicDim>
void characteristic_speeds_approximate_mhd(
    const gsl::not_null<std::array<DataVector, 9>*> char_speeds,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  // Remaining places to reduce allocations:
  // - EoS calls: 2 allocations
  // - Pass temp pointer to Rel Euler: 1 allocation
  // - Return a DataVectorArray (not yet implemented): 9 allocations
  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::MagneticFieldOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
                       hydro::Tags::MagneticFieldSquared<DataVector>,
                       hydro::Tags::ComovingMagneticFieldSquared<DataVector>,
                       hydro::Tags::SoundSpeedSquared<DataVector>>>
      temp_tensors{get<0>(shift).size()};

  const auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(
          temp_tensors)),
      spatial_velocity, spatial_metric);
  const auto& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(
          &get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors)),
      magnetic_field, spatial_metric);
  const auto& magnetic_field_dot_spatial_velocity =
      get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
          temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>>(
              temp_tensors)),
      magnetic_field, spatial_velocity_one_form);
  const auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors)),
      spatial_velocity, spatial_velocity_one_form);

  const auto& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  dot_product(make_not_null(&get<hydro::Tags::MagneticFieldSquared<DataVector>>(
                  temp_tensors)),
              magnetic_field, magnetic_field_one_form);
  const auto& comoving_magnetic_field_squared =
      get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(temp_tensors);
  get(get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(
      temp_tensors)) =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));

  // reuse magnetic_field_squared allocation for Alfven speed squared
  const auto& alfven_speed_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  get(get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors)) =
      get(comoving_magnetic_field_squared) /
      (get(comoving_magnetic_field_squared) +
       get(rest_mass_density) * get(specific_enthalpy));

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 3) {
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
  }

  compute_characteristic_speeds_approximate_mhd(
      char_speeds, lapse, shift, spatial_velocity, spatial_velocity_squared,
      sound_speed_squared, alfven_speed_squared, unit_normal);
}

template <size_t ThermodynamicDim>
std::array<DataVector, 9> characteristic_speeds_approximate_mhd(
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  std::array<DataVector, 9> char_speeds{};
  characteristic_speeds_approximate_mhd(
      make_not_null(&char_speeds), rest_mass_density, electron_fraction,
      specific_internal_energy, specific_enthalpy, spatial_velocity,
      lorentz_factor, magnetic_field, lapse, shift, spatial_metric, unit_normal,
      equation_of_state);
  return char_speeds;
}

namespace Tags {

template <size_t ThermodynamicDim>
void CharacteristicSpeedsCompute::function(
    const gsl::not_null<return_type*> result,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& /* electron_fraction */,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  characteristic_speeds_approximate_mhd<ThermodynamicDim>(
      result, rest_mass_density, /*electron_fraction*/ {},
      specific_internal_energy, specific_enthalpy, spatial_velocity,
      lorentz_factor, magnetic_field, lapse, shift, spatial_metric, unit_normal,
      equation_of_state);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define FUNCTION_INSTANTIATION(r, data)                                     \
  template void CharacteristicSpeedsCompute::function<DIM(data)>(           \
      const gsl::not_null<return_type*> result,                             \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& electron_fraction,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, DIM(data)>&             \
          equation_of_state);

GENERATE_INSTANTIATIONS(FUNCTION_INSTANTIATION, (1, 2, 3))
#undef DIM
#undef FUNCTION_INSTANTIATION

}  // namespace Tags

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template std::array<DataVector, 9>                                        \
  characteristic_speeds_approximate_mhd<GET_DIM(data)>(                     \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& electron_fraction,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state);                                               \
  template void characteristic_speeds_approximate_mhd<GET_DIM(data)>(       \
      const gsl::not_null<std::array<DataVector, 9>*> char_speeds,          \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& electron_fraction,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace grmhd::ValenciaDivClean
