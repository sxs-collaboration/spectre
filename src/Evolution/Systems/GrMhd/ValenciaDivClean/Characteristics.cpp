// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Tags.hpp"              // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Overloader.hpp"

// IWYU pragma: no_forward_declare EquationsOfState::EquationOfState
// IWYU pragma: no_forward_declare Tensor

/// \cond

namespace {
void compute_characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 9>*> pchar_speeds,
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,
    const tnsr::I<DataVector, 3>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const Scalar<DataVector>& alfven_speed_squared,
    const tnsr::i<DataVector, 3>& normal) noexcept {
  const size_t num_grid_points = get(lapse).size();
  auto& char_speeds = *pchar_speeds;
  if (char_speeds[0].size() != num_grid_points) {
    char_speeds[0] = DataVector(num_grid_points);
  }
  Scalar<DataVector> temp0(char_speeds[0].data(), num_grid_points);
  dot_product(make_not_null(&temp0), normal, shift);
  char_speeds[0] *= -1.0;
  char_speeds[8] = char_speeds[0] + get(lapse);

  char_speeds[0] -= get(lapse);
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
  //
  // Create an array of non-owning DataVectors to be passed to the Relativistic
  // Euler char speed computation as a not_null<array<DataVectors>>.
  std::array<DataVector, 5> rel_euler_char_speeds{};
  for (size_t i = 0; i < 4; ++i) {
    if (gsl::at(char_speeds, i + 1).size() != num_grid_points) {
      gsl::at(char_speeds, i + 1) = DataVector(num_grid_points);
    }
    gsl::at(rel_euler_char_speeds, i)
        .set_data_ref(&gsl::at(char_speeds, i + 1));
  }
  if (gsl::at(char_speeds, 7).size() != num_grid_points) {
    gsl::at(char_speeds, 7) = DataVector(num_grid_points);
  }
  rel_euler_char_speeds[4].set_data_ref(&(char_speeds[7]));

  RelativisticEuler::Valencia::characteristic_speeds(
      make_not_null(&rel_euler_char_speeds), lapse, shift, spatial_velocity,
      spatial_velocity_squared,
      Scalar<DataVector>{get(sound_speed_squared) +
                         get(alfven_speed_squared) *
                             (1.0 - get(sound_speed_squared))},
      normal);

  for (size_t i = 5; i < 7; ++i) {
    gsl::at(char_speeds, i) = rel_euler_char_speeds[1];
  }
}
}  // namespace

namespace grmhd {
namespace ValenciaDivClean {
template <size_t ThermodynamicDim>
void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 9>*> char_speeds,
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
  // Use a Variables to reduce total number of allocations. This is especially
  // important in a multithreaded environment.
  // Remaining places to reduce allocations:
  // - EoS calls: 2 allocations
  // - Pass temp pointer to Rel Euler: 1 allocation
  // - Return a DataVectorArray (not yet implemented): 9 allocations
  Variables<tmpl::list<
      hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>,
      hydro::Tags::SpatialVelocitySquared<DataVector>,
      hydro::Tags::MagneticFieldOneForm<DataVector, 3, Frame::Inertial>,
      hydro::Tags::MagneticFieldDotSpatialVelocity<DataVector>,
      hydro::Tags::MagneticFieldSquared<DataVector>,
      hydro::Tags::ComovingMagneticFieldSquared<DataVector>,
      hydro::Tags::SoundSpeedSquared<DataVector>>>
      temp_tensors{get<0>(shift).size()};

  const auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3, Frame::Inertial>>(
          temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<
                        DataVector, 3, Frame::Inertial>>(temp_tensors)),
      spatial_velocity, spatial_metric);
  const auto& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3, Frame::Inertial>>(
          temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::MagneticFieldOneForm<
                        DataVector, 3, Frame::Inertial>>(temp_tensors)),
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
  make_overloader(
      [&rest_mass_density, &
       sound_speed_squared ](const EquationsOfState::EquationOfState<true, 1>&
                                 the_equation_of_state) noexcept {
        get(sound_speed_squared) =
            get(the_equation_of_state.chi_from_density(rest_mass_density)) +
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density(
                        rest_mass_density));
      },
      [&rest_mass_density, &specific_internal_energy, &
       sound_speed_squared ](const EquationsOfState::EquationOfState<true, 2>&
                                 the_equation_of_state) noexcept {
        get(sound_speed_squared) =
            get(the_equation_of_state.chi_from_density_and_energy(
                rest_mass_density, specific_internal_energy)) +
            get(the_equation_of_state
                    .kappa_times_p_over_rho_squared_from_density_and_energy(
                        rest_mass_density, specific_internal_energy));
      })(equation_of_state);
  get(sound_speed_squared) /= get(specific_enthalpy);

  compute_characteristic_speeds(char_speeds, lapse, shift, spatial_velocity,
                                spatial_velocity_squared, sound_speed_squared,
                                alfven_speed_squared, unit_normal);
}

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
  std::array<DataVector, 9> char_speeds{};
  characteristic_speeds(make_not_null(&char_speeds), rest_mass_density,
                        specific_internal_energy, specific_enthalpy,
                        spatial_velocity, lorentz_factor, magnetic_field, lapse,
                        shift, spatial_metric, unit_normal, equation_of_state);
  return char_speeds;
}

#define GET_DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATION(r, data)                                              \
  template std::array<DataVector, 9> characteristic_speeds<GET_DIM(data)>(  \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state) noexcept;                                      \
  template void characteristic_speeds<GET_DIM(data)>(                       \
      const gsl::not_null<std::array<DataVector, 9>*> char_speeds,          \
      const Scalar<DataVector>& rest_mass_density,                          \
      const Scalar<DataVector>& specific_internal_energy,                   \
      const Scalar<DataVector>& specific_enthalpy,                          \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,      \
      const Scalar<DataVector>& lorentz_factor,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,        \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift, \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,       \
      const tnsr::i<DataVector, 3>& unit_normal,                            \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&         \
          equation_of_state) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace ValenciaDivClean
}  // namespace grmhd
/// \endcond
