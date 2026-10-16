// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <cstddef>
#include <gsl/gsl_complex.h>
#include <gsl/gsl_complex_math.h>
#include <gsl/gsl_eigen.h>
#include <gsl/gsl_math.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"
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

template <size_t ThermodynamicDim>
void characteristic_speeds_hydro(
    const gsl::not_null<std::array<DataVector, 3>*> char_speeds,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points = get(lorentz_factor).size();
  if ((*char_speeds)[0].size() != num_grid_points) {
    for (auto& cs : (*char_speeds)) {
      cs = DataVector(num_grid_points, 0.0);
    }
  }

  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::SoundSpeedSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>>>
      temp_tensors{num_grid_points};

  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), unit_normal, spatial_velocity);

  const auto& spatial_velocity_one_form =
      get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(temp_tensors);
  raise_or_lower_index(
      make_not_null(&get<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>>(
          temp_tensors)),
      spatial_velocity, spatial_metric);

  const auto& spatial_velocity_squared =
      get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors);
  dot_product(
      make_not_null(
          &get<hydro::Tags::SpatialVelocitySquared<DataVector>>(temp_tensors)),
      spatial_velocity, spatial_velocity_one_form);

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

  // Calculate the characteristic speed for non-degenerate ones
  Scalar<DataVector>& denom = get<::Tags::TempScalar<1>>(temp_tensors);
  get(denom) = 1.0 - get(spatial_velocity_squared) * get(sound_speed_squared);

  Scalar<DataVector>& first_term = get<::Tags::TempScalar<2>>(temp_tensors);
  get(first_term) =
      (1.0 - get(sound_speed_squared)) * get(normal_velocity) / get(denom);

  Scalar<DataVector>& second_term = get<::Tags::TempScalar<3>>(temp_tensors);
  get(second_term) =
      sqrt(get(sound_speed_squared)) *
      sqrt(get(denom) - get(normal_velocity) * get(normal_velocity) *
                            (1 - get(sound_speed_squared))) /
      (get(lorentz_factor) * get(denom));

  // Degenerate eigenvalue (normal dot velocity)
  (*char_speeds)[HydroSpeed::NormalDotVelocity] = get(normal_velocity);
  (*char_speeds)[HydroSpeed::LambdaPlus] = get(first_term) + get(second_term);
  (*char_speeds)[HydroSpeed::LambdaMinus] = get(first_term) - get(second_term);
}

template <size_t ThermodynamicDim>
std::array<DataVector, 3> characteristic_speeds_hydro(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  std::array<DataVector, 3> char_speeds{};
  characteristic_speeds_hydro(
      make_not_null(&char_speeds), spatial_velocity, rest_mass_density,
      specific_internal_energy, specific_enthalpy, electron_fraction,
      lorentz_factor, unit_normal, spatial_metric, equation_of_state);
  return char_speeds;
}

template <size_t ThermodynamicDim>
void eigenvectors_hydro(
    const gsl::not_null<std::array<tnsr::i<DataVector, 6, Frame::Inertial>,
                                   6>*>& right_eigenvectors,
    const gsl::not_null<std::array<tnsr::I<DataVector, 6, Frame::Inertial>,
                                   6>*>& left_eigenvectors,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& specific_enthalpy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor, const Scalar<DataVector>& kappa,
    const Scalar<DataVector>& zeta, const tnsr::i<DataVector, 3>& unit_normal,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points = get(lorentz_factor).size();

  auto allocate_and_zero = [num_grid_points](auto& vec_array) {
    if (vec_array[0].get(0).size() != num_grid_points) {
      for (auto& vec : vec_array) {
        for (size_t a = 0; a < 6; ++a) {
          vec.get(a) = DataVector(num_grid_points, 0.0);
        }
      }
    } else {
      for (auto& vec : vec_array) {
        for (size_t a = 0; a < 6; ++a) {
          vec.get(a) = 0.0;
        }
      }
    }
  };
  allocate_and_zero(*right_eigenvectors);
  allocate_and_zero(*left_eigenvectors);

  // This is for the case for zeta = 0.
  const double zeta_max_abs = max(abs(get(zeta)));

  Scalar<DataVector> det_spatial_metric{num_grid_points};
  tnsr::II<DataVector, 3, Frame::Inertial> inv_spatial_metric{num_grid_points};
  determinant_and_inverse(make_not_null(&det_spatial_metric),
                          make_not_null(&inv_spatial_metric), spatial_metric);

  tnsr::i<DataVector, 3, Frame::Inertial> tangent_one_form_1{num_grid_points};
  orthonormal_oneform(make_not_null(&tangent_one_form_1), unit_normal,
                      inv_spatial_metric);

  tnsr::i<DataVector, 3, Frame::Inertial> tangent_one_form_2{num_grid_points};
  orthonormal_oneform(make_not_null(&tangent_one_form_2), unit_normal,
                      tangent_one_form_1, spatial_metric, det_spatial_metric);

  tnsr::I<DataVector, 3, Frame::Inertial> unit_normal_vector{num_grid_points};
  raise_or_lower_index(make_not_null(&unit_normal_vector), unit_normal,
                       inv_spatial_metric);

  tnsr::I<DataVector, 3, Frame::Inertial> tangent_vector_1{num_grid_points};
  raise_or_lower_index(make_not_null(&tangent_vector_1), tangent_one_form_1,
                       inv_spatial_metric);

  tnsr::I<DataVector, 3, Frame::Inertial> tangent_vector_2{num_grid_points};
  raise_or_lower_index(make_not_null(&tangent_vector_2), tangent_one_form_2,
                       inv_spatial_metric);

  Scalar<DataVector> v_dot_tangent_1{num_grid_points};
  dot_product(make_not_null(&v_dot_tangent_1), tangent_one_form_1,
              spatial_velocity);

  Scalar<DataVector> v_dot_tangent_2{num_grid_points};
  dot_product(make_not_null(&v_dot_tangent_2), tangent_one_form_2,
              spatial_velocity);

  Scalar<DataVector> normal_velocity{num_grid_points};
  dot_product(make_not_null(&normal_velocity), unit_normal, spatial_velocity);

  const DataVector one_minus_normal_velocity_squared =
      1.0 - get(normal_velocity) * get(normal_velocity);

  tnsr::i<DataVector, 3, Frame::Inertial> spatial_velocity_one_form{
      num_grid_points};
  raise_or_lower_index(make_not_null(&spatial_velocity_one_form),
                       spatial_velocity, spatial_metric);

  Scalar<DataVector> spatial_velocity_squared{num_grid_points};
  dot_product(make_not_null(&spatial_velocity_squared), spatial_velocity,
              spatial_velocity_one_form);

  Scalar<DataVector> sound_speed_squared{num_grid_points};
  Scalar<DataVector> pressure{num_grid_points};

  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(pressure) =
        get(equation_of_state.pressure_from_density(rest_mass_density));
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
  } else if constexpr (ThermodynamicDim == 3) {
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction));
  }

  const DataVector sound_speed = sqrt(get(sound_speed_squared));
  const DataVector W_squared = square(get(lorentz_factor));

  // Variables in the ordering (D, S_i, tau, DYe)
  // RIGHT eigenvectors

  // R1 and R2
  for (size_t i = 0; i < 3; ++i) {
    (*right_eigenvectors)[R1].get(i + 1) =
        get(specific_enthalpy) *
        (tangent_one_form_1.get(i) + 2.0 * W_squared * get(v_dot_tangent_1) *
                                         spatial_velocity_one_form.get(i));

    (*right_eigenvectors)[R2].get(i + 1) =
        get(specific_enthalpy) *
        (tangent_one_form_2.get(i) + 2.0 * W_squared * get(v_dot_tangent_2) *
                                         spatial_velocity_one_form.get(i));
  }

  (*right_eigenvectors)[R1].get(0) = get(lorentz_factor) * get(v_dot_tangent_1);
  (*right_eigenvectors)[R2].get(0) = get(lorentz_factor) * get(v_dot_tangent_2);

  (*right_eigenvectors)[R1].get(4) =
      get(lorentz_factor) *
      (2.0 * get(specific_enthalpy) * get(lorentz_factor) - 1.0) *
      get(v_dot_tangent_1);

  (*right_eigenvectors)[R2].get(4) =
      get(lorentz_factor) *
      (2.0 * get(specific_enthalpy) * get(lorentz_factor) - 1.0) *
      get(v_dot_tangent_2);

  (*right_eigenvectors)[R1].get(5) =
      get(electron_fraction) * (*right_eigenvectors)[R1].get(0);
  (*right_eigenvectors)[R2].get(5) =
      get(electron_fraction) * (*right_eigenvectors)[R2].get(0);

  // R3
  const DataVector common_R3 =
      get(specific_enthalpy) * get(lorentz_factor) *
      (get(kappa) - get(rest_mass_density) * get(sound_speed_squared));
  for (size_t i = 0; i < 3; ++i) {
    (*right_eigenvectors)[R3].get(i + 1) =
        common_R3 * spatial_velocity_one_form.get(i);
  }
  (*right_eigenvectors)[R3].get(0) = get(kappa);
  (*right_eigenvectors)[R3].get(4) = common_R3 - get(kappa);
  (*right_eigenvectors)[R3].get(5) =
      get(electron_fraction) * (*right_eigenvectors)[R3].get(0);

  // R4
  if (zeta_max_abs < 1e-14) {
    (*right_eigenvectors)[R4].get(5) = 1.0;
  } else {
    for (size_t i = 0; i < 3; ++i) {
      (*right_eigenvectors)[R4].get(i + 1) = spatial_velocity_one_form.get(i);
    }
    (*right_eigenvectors)[R4].get(4) = 1.0;
    (*right_eigenvectors)[R4].get(5) =
        -get(kappa) / (get(zeta) * get(lorentz_factor));
  }

  // R±
  const DataVector denom =
      get(lorentz_factor) *
      sqrt(1.0 - get(spatial_velocity_squared) * get(sound_speed_squared) -
           get(normal_velocity) * get(normal_velocity) *
               (1.0 - get(sound_speed_squared)));

  const DataVector sound_speed_over_denom = sound_speed / denom;

  for (size_t i = 0; i < 3; ++i) {
    (*right_eigenvectors)[Rplus].get(i + 1) =
        get(specific_enthalpy) * get(lorentz_factor) *
        (spatial_velocity_one_form.get(i) +
         sound_speed_over_denom * unit_normal.get(i));

    (*right_eigenvectors)[Rminus].get(i + 1) =
        get(specific_enthalpy) * get(lorentz_factor) *
        (spatial_velocity_one_form.get(i) -
         sound_speed_over_denom * unit_normal.get(i));
  }

  (*right_eigenvectors)[Rplus].get(0) = 1.0;
  (*right_eigenvectors)[Rminus].get(0) = 1.0;

  (*right_eigenvectors)[Rplus].get(4) =
      get(specific_enthalpy) * get(lorentz_factor) *
          (1.0 + sound_speed * get(normal_velocity) / denom) -
      1.0;

  (*right_eigenvectors)[Rminus].get(4) =
      get(specific_enthalpy) * get(lorentz_factor) *
          (1.0 - sound_speed * get(normal_velocity) / denom) -
      1.0;

  (*right_eigenvectors)[Rplus].get(5) = get(electron_fraction);
  (*right_eigenvectors)[Rminus].get(5) = get(electron_fraction);

  // LEFT eigenvectors
  const DataVector prefactor_L12 =
      1.0 / (get(specific_enthalpy) * one_minus_normal_velocity_squared);

  // L1
  (*left_eigenvectors)[L1].get(0) = -get(v_dot_tangent_1) * prefactor_L12;
  (*left_eigenvectors)[L1].get(4) = -get(v_dot_tangent_1) * prefactor_L12;
  for (size_t i = 0; i < 3; ++i) {
    (*left_eigenvectors)[L1].get(i + 1) =
        (get(v_dot_tangent_1) * get(normal_velocity) *
             unit_normal_vector.get(i) +
         one_minus_normal_velocity_squared * tangent_vector_1.get(i)) *
        prefactor_L12;
  }

  // L2
  (*left_eigenvectors)[L2].get(0) = -get(v_dot_tangent_2) * prefactor_L12;
  (*left_eigenvectors)[L2].get(4) = -get(v_dot_tangent_2) * prefactor_L12;
  for (size_t i = 0; i < 3; ++i) {
    (*left_eigenvectors)[L2].get(i + 1) =
        (get(v_dot_tangent_2) * get(normal_velocity) *
             unit_normal_vector.get(i) +
         one_minus_normal_velocity_squared * tangent_vector_2.get(i)) *
        prefactor_L12;
  }

  // L3
  {
    const DataVector prefactor_L3 =
        1.0 / (get(rest_mass_density) * get(specific_enthalpy) *
               get(sound_speed_squared));

    const DataVector h_minus_one =
        get(specific_internal_energy) + get(pressure) / get(rest_mass_density);

    const DataVector W_minus_one = get(spatial_velocity_squared) *
                                   square(get(lorentz_factor)) /
                                   (get(lorentz_factor) + 1.0);

    const DataVector h_minus_W = h_minus_one - W_minus_one;

    (*left_eigenvectors)[L3].get(0) =
        (h_minus_W + get(zeta) * get(electron_fraction) / get(kappa)) *
        prefactor_L3;

    for (size_t i = 0; i < 3; ++i) {
      (*left_eigenvectors)[L3].get(i + 1) =
          (get(lorentz_factor) * spatial_velocity.get(i)) * prefactor_L3;
    }

    (*left_eigenvectors)[L3].get(4) = (-get(lorentz_factor)) * prefactor_L3;
    (*left_eigenvectors)[L3].get(5) = (-get(zeta) / get(kappa)) * prefactor_L3;
  }

  // L4
  {
    if (zeta_max_abs < 1e-14) {
      (*left_eigenvectors)[L4].get(0) = -get(electron_fraction);
      (*left_eigenvectors)[L4].get(5) = 1.0;
    } else {
      const DataVector prefactor_L4 =
          get(zeta) * get(lorentz_factor) / get(kappa);
      (*left_eigenvectors)[L4].get(0) = prefactor_L4 * get(electron_fraction);
      (*left_eigenvectors)[L4].get(5) = -prefactor_L4;
    }
  }

  // L±
  {
    const DataVector a =
        square(get(lorentz_factor)) * one_minus_normal_velocity_squared *
        (get(kappa) + get(rest_mass_density) * get(sound_speed_squared));

    const DataVector c_plus = get(rest_mass_density) * sound_speed *
                              (sound_speed + get(normal_velocity) * denom);
    const DataVector c_minus = get(rest_mass_density) * sound_speed *
                               (sound_speed - get(normal_velocity) * denom);

    const DataVector b_plus = a - c_plus;
    const DataVector b_minus = a - c_minus;

    const DataVector k_term =
        get(kappa) - get(rest_mass_density) * get(sound_speed_squared) +
        get(zeta) * get(electron_fraction) / get(specific_enthalpy);

    const DataVector prefactor_Lpm =
        1.0 / (2.0 * get(rest_mass_density) * get(specific_enthalpy) *
               get(lorentz_factor) * get(sound_speed_squared) *
               one_minus_normal_velocity_squared);

    // S_i
    for (size_t i = 0; i < 3; ++i) {
      (*left_eigenvectors)[Lplus].get(i + 1) =
          (-a * spatial_velocity.get(i) +
           get(rest_mass_density) * sound_speed *
               (sound_speed * get(normal_velocity) + denom) *
               unit_normal_vector.get(i)) *
          prefactor_Lpm;

      (*left_eigenvectors)[Lminus].get(i + 1) =
          (-a * spatial_velocity.get(i) +
           get(rest_mass_density) * sound_speed *
               (sound_speed * get(normal_velocity) - denom) *
               unit_normal_vector.get(i)) *
          prefactor_Lpm;
    }

    // D
    (*left_eigenvectors)[Lplus].get(0) =
        (b_plus - get(specific_enthalpy) * get(lorentz_factor) * k_term *
                      one_minus_normal_velocity_squared) *
        prefactor_Lpm;

    (*left_eigenvectors)[Lminus].get(0) =
        (b_minus - get(specific_enthalpy) * get(lorentz_factor) * k_term *
                       one_minus_normal_velocity_squared) *
        prefactor_Lpm;

    // tau
    (*left_eigenvectors)[Lplus].get(4) = b_plus * prefactor_Lpm;
    (*left_eigenvectors)[Lminus].get(4) = b_minus * prefactor_Lpm;

    // DYe
    (*left_eigenvectors)[Lplus].get(5) =
        (get(zeta) * get(lorentz_factor) * one_minus_normal_velocity_squared) *
        prefactor_Lpm;
    (*left_eigenvectors)[Lminus].get(5) =
        (get(zeta) * get(lorentz_factor) * one_minus_normal_velocity_squared) *
        prefactor_Lpm;
  }
}

namespace detail {

template <size_t ThermodynamicDim>
void flux_jacobian_hydro(
    gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  Variables<tmpl::list<hydro::Tags::SoundSpeedSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>>>
      temp_tensors{get<0, 0>(spatial_metric).size()};

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  // We define kappa as the partial derivative of pressure with respect to
  // specific internal energy
  Scalar<DataVector>& kappa = get<::Tags::TempScalar<0>>(temp_tensors);
  // We define zeta as the partial derivative of pressure with respect to
  // electron fraction
  Scalar<DataVector>& zeta = get<::Tags::TempScalar<1>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(kappa) = 0.0;
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(equation_of_state
                 .kappa_times_p_over_rho_squared_from_density_and_energy(
                     rest_mass_density, specific_internal_energy))) /
        get(specific_enthalpy);
    const Scalar<DataVector> kappa_times_p_over_rho_squared =
        equation_of_state
            .kappa_times_p_over_rho_squared_from_density_and_energy(
                rest_mass_density, specific_internal_energy);
    const Scalar<DataVector> pressure =
        equation_of_state.pressure_from_density_and_energy(
            rest_mass_density, specific_internal_energy);
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    // The following computation works for a general 3D EoS, but it doesn't
    // allow getting kappa.
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
    // So, we're currently using the equations from an ideal fluid EoS to set
    // kappa, assuming the same adiabatic index as used in the tests. This
    // approach will need to be improved during the code review process..
    const double adiabatic_index = 1.5;
    const Scalar<DataVector> chi =
        tenex::evaluate(specific_internal_energy() * (adiabatic_index - 1.0));
    const Scalar<DataVector> kappa_times_p_over_rho_squared = tenex::evaluate(
        square(adiabatic_index - 1.0) * specific_internal_energy());
    const DataVector sound_speed_squared_ideal_fluid =
        (get(chi) + get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    ASSERT(max(abs(get(sound_speed_squared) -
                   sound_speed_squared_ideal_fluid)) < 1e-10,
           "The ideal fluid approximation for kappa is not valid.");
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction);
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    // For now, we assume that we are at compositional equilibrium, so we set
    // zeta to zero.
    get(zeta) = 0.0;
  }

  // Intermediate variables
  const auto Z = tenex::evaluate(rest_mass_density() * specific_enthalpy() *
                                 square(lorentz_factor()));
  const auto D = tenex::evaluate(rest_mass_density() * lorentz_factor());
  const auto normal_velocity =
      tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));
  const auto unit_vector = tenex::evaluate<ti::I>(
      inv_spatial_metric(ti::I, ti::J) * unit_normal(ti::j));
  const auto spatial_velocity_one_form = tenex::evaluate<ti::i>(
      spatial_metric(ti::i, ti::j) * spatial_velocity(ti::J));
  const auto mixed_spatial_metric = tenex::evaluate<ti::I, ti::j>(
      inv_spatial_metric(ti::I, ti::K) * spatial_metric(ti::k, ti::j));

  // Derivatives of Z
  const auto dzdD = tenex::evaluate(
      -((lorentz_factor() *
         (kappa() * (-specific_enthalpy() + lorentz_factor()) -
          zeta() * electron_fraction() +
          (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
              rest_mass_density())) /
        ((-square(lorentz_factor()) +
          sound_speed_squared() * (-1. + square(lorentz_factor()))) *
         rest_mass_density())));
  const auto dzds = tenex::evaluate<ti::I>(
      (spatial_velocity(ti::I) * square(lorentz_factor()) *
       (kappa() + sound_speed_squared() * rest_mass_density())) /
      ((-square(lorentz_factor()) +
        sound_speed_squared() * (-1. + square(lorentz_factor()))) *
       rest_mass_density()));
  const auto dzdtau = tenex::evaluate(
      -((square(lorentz_factor()) * (kappa() + rest_mass_density())) /
        ((-square(lorentz_factor()) +
          sound_speed_squared() * (-1. + square(lorentz_factor()))) *
         rest_mass_density())));
  const auto dzdye = tenex::evaluate(
      (zeta() * lorentz_factor()) /
      ((square(lorentz_factor()) -
        sound_speed_squared() * (-1. + square(lorentz_factor()))) *
       rest_mass_density()));

  // Put analytic expressions into characteristic matrix
  characteristic_matrix->get(0, 0) =
      ((get(Z) - get(D) * get(dzdD)) * get(normal_velocity)) / get(Z);
  for (size_t B = 0; B < 3; ++B) {
    characteristic_matrix->get(0, B + 1) =
        (get(D) * (unit_vector.get(B) - dzds.get(B) * get(normal_velocity))) /
        get(Z);
  }
  characteristic_matrix->get(0, 4) =
      -((get(D) * get(dzdtau) * get(normal_velocity)) / get(Z));
  characteristic_matrix->get(0, 5) =
      -((get(D) * get(dzdye) * get(normal_velocity)) / get(Z));
  for (size_t c = 0; c < 3; ++c) {
    characteristic_matrix->get(c + 1, 0) =
        (-1.0 + get(dzdD)) * unit_normal.get(c) -
        get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(c);
    for (size_t B = 0; B < 3; ++B) {
      characteristic_matrix->get(c + 1, B + 1) =
          mixed_spatial_metric.get(B, c) * get(normal_velocity) +
          unit_vector.get(B) * spatial_velocity_one_form.get(c) +
          dzds.get(B) *
              (unit_normal.get(c) -
               get(normal_velocity) * spatial_velocity_one_form.get(c));
    }
    characteristic_matrix->get(c + 1, 4) =
        (-1.0 + get(dzdtau)) * unit_normal.get(c) -
        get(dzdtau) * get(normal_velocity) * spatial_velocity_one_form.get(c);
    characteristic_matrix->get(c + 1, 5) =
        get(dzdye) * (unit_normal.get(c) -
                      get(normal_velocity) * spatial_velocity_one_form.get(c));
  }
  characteristic_matrix->get(4, 0) =
      -(((get(Z) - get(D) * get(dzdD)) * get(normal_velocity)) / get(Z));
  for (size_t B = 0; B < 3; ++B) {
    characteristic_matrix->get(4, B + 1) =
        ((get(Z) - get(D)) * unit_vector.get(B) +
         get(D) * dzds.get(B) * get(normal_velocity)) /
        get(Z);
  }
  characteristic_matrix->get(4, 4) =
      (get(D) * get(dzdtau) * get(normal_velocity)) / get(Z);
  characteristic_matrix->get(4, 5) =
      (get(D) * get(dzdye) * get(normal_velocity)) / get(Z);
  characteristic_matrix->get(5, 0) =
      -((get(D) * get(electron_fraction) * get(dzdD) * get(normal_velocity)) /
        get(Z));
  for (size_t B = 0; B < 3; ++B) {
    characteristic_matrix->get(5, B + 1) =
        (get(D) * get(electron_fraction) *
         (unit_vector.get(B) - dzds.get(B) * get(normal_velocity))) /
        get(Z);
  }
  characteristic_matrix->get(5, 4) =
      -((get(D) * get(electron_fraction) * get(dzdtau) * get(normal_velocity)) /
        get(Z));
  characteristic_matrix->get(5, 5) =
      ((get(Z) - get(D) * get(electron_fraction) * get(dzdye)) *
       get(normal_velocity)) /
      get(Z);
}
}  // namespace detail

template <size_t ThermodynamicDim>
void numerical_eigensystem(
    gsl::not_null<std::array<Scalar<DataVector>, 6>*> all_eigenvalues,
    gsl::not_null<std::array<tnsr::i<DataVector, 6>, 6>*>
        all_right_eigenvectors,
    gsl::not_null<std::array<tnsr::I<DataVector, 6>, 6>*> all_left_eigenvectors,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  // Build characteristic matrix
  tnsr::iJ<DataVector, 6> characteristic_matrix =
      make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
  detail::flux_jacobian_hydro(
      make_not_null(&characteristic_matrix), spatial_velocity,
      rest_mass_density, specific_internal_energy, electron_fraction,
      /* other helpful quantities */
      lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
      unit_normal, equation_of_state);

  // Allocate memory to work with blaze::geev (outside of loop to save
  // time/memory)
  constexpr size_t matrix_size = 6;
  Matrix blaze_point_matrix(matrix_size, matrix_size);
  blaze::DynamicVector<blaze::complex<double>> blaze_complex_eigenvalues(
      matrix_size);
  blaze::DynamicMatrix<blaze::complex<double>> blaze_complex_L(matrix_size,
                                                               matrix_size);
  blaze::DynamicMatrix<blaze::complex<double>> blaze_complex_R(matrix_size,
                                                               matrix_size);
  std::vector<double> blaze_real_eigenvalues(matrix_size);

  // Loop over each grid point
  const size_t num_points = get<0, 0>(spatial_metric).size();
  for (size_t point = 0; point < num_points; ++point) {
    // Build matrix at this grid point
    for (size_t row = 0; row < matrix_size; ++row) {
      for (size_t col = 0; col < matrix_size; ++col) {
        blaze_point_matrix(row, col) =
            characteristic_matrix.get(row, col)[point];
      }
    }

    // Solve eigensystem using blaze:geev
    blaze::geev(blaze_point_matrix, blaze_complex_L, blaze_complex_eigenvalues,
                blaze_complex_R);
    const double tolerance = 1.0e-12;
    for (size_t i = 0; i < matrix_size; ++i) {
      ASSERT(std::abs(blaze_complex_eigenvalues[i].imag()) < tolerance,
             "Complex eigenvalue: "
                 << blaze_complex_eigenvalues[i].real() << " + "
                 << blaze_complex_eigenvalues[i].imag() << " i.");
      blaze_real_eigenvalues[i] = blaze_complex_eigenvalues[i].real();
    }

// We found cases in which blaze::geev returns a wrong eigenvalue. As a sanity
// check, we use GSL to compute the eigenvalues for the same matrix and check
// that they are the same.
#ifdef SPECTRE_DEBUG
    // Allocate memory to work with GSL:
    // Workspace for computing eigenvalues and eigenvectors
    gsl_eigen_nonsymm_workspace* gsl_workspace =
        gsl_eigen_nonsymm_alloc(matrix_size);
    // Containers for the right eigensystem (A * R = lambda * R)
    gsl_matrix* gsl_point_matrix = gsl_matrix_alloc(matrix_size, matrix_size);
    gsl_vector_complex* gsl_complex_eigenvalues =
        gsl_vector_complex_alloc(matrix_size);
    gsl_matrix_complex* gsl_complex_right_eigenvectors =
        gsl_matrix_complex_alloc(matrix_size, matrix_size);
    std::vector<double> gsl_real_eigenvalues(matrix_size);

    // Build matrix at this grid point
    for (size_t row = 0; row < matrix_size; ++row) {
      for (size_t col = 0; col < matrix_size; ++col) {
        const double entry = characteristic_matrix.get(row, col)[point];
        gsl_matrix_set(gsl_point_matrix, row, col, entry);
      }
    }

    // Solve for eigenvalues using GSL
    gsl_eigen_nonsymm(gsl_point_matrix, gsl_complex_eigenvalues, gsl_workspace);
    for (size_t i = 0; i < matrix_size; ++i) {
      gsl_real_eigenvalues[i] =
          GSL_REAL(gsl_vector_complex_get(gsl_complex_eigenvalues, i));
    }

    // Sort both blaze's and GSL's eigenvalues to facilitate comparison
    std::vector<double> sorted_blaze_real_eigenvalues = blaze_real_eigenvalues;
    std::sort(sorted_blaze_real_eigenvalues.begin(),
              sorted_blaze_real_eigenvalues.end());
    std::sort(gsl_real_eigenvalues.begin(), gsl_real_eigenvalues.end());

    // Compare eigenvalues
    for (size_t i = 0; i < matrix_size; ++i) {
      if (UNLIKELY(std::abs(sorted_blaze_real_eigenvalues[i] -
                            gsl_real_eigenvalues[i]) > 1.e-6)) {
        std::ostringstream matrix_stream;
        matrix_stream << "Point matrix:\n";
        for (size_t row = 0; row < matrix_size; ++row) {
          for (size_t col = 0; col < matrix_size; ++col) {
            matrix_stream << blaze_point_matrix(row, col) << " ";
          }
          matrix_stream << "\n";
        }
        matrix_stream << "Blaze eigenvalue: "
                      << sorted_blaze_real_eigenvalues[i]
                      << ", GSL eigenvalue: " << gsl_real_eigenvalues[i]
                      << "\n";

        ERROR(
            "The eigensolvers from blaze and GSL found different eigenvalues "
            "for the same matrix.\n"
            << matrix_stream.str());
      }
    }

    // Free GSL allocations
    gsl_eigen_nonsymm_free(gsl_workspace);
    gsl_matrix_free(gsl_point_matrix);
    gsl_vector_complex_free(gsl_complex_eigenvalues);
    gsl_matrix_complex_free(gsl_complex_right_eigenvectors);
#endif

    // Check and save results
    // Track which indices have been processed to avoid double-counting partners
    std::vector<bool> processed_right_eigenvectors(matrix_size, false);
    std::vector<bool> processed_left_eigenvectors(matrix_size, false);
    // Note: we're looping through the ith eigenvalue/vectors, not the ith row!
    for (size_t i = 0; i < matrix_size; ++i) {
      // Store eigenvalue
      get(gsl::at(*all_eigenvalues, i))[point] = blaze_real_eigenvalues[i];

      // For each PAIR of degenerate eigenvalues, it is possible that GSL
      // returns a PAIR of complex eigenvectors that are complex conjugates of
      // each other. Here, we check if the ith right/left eigenvector is
      // complex.
      bool right_eigenvector_is_complex = false;
      bool left_eigenvector_is_complex = false;
      for (size_t k = 0; k < matrix_size; ++k) {
        if (std::abs(blaze_complex_R(k, i).imag()) > tolerance) {
          right_eigenvector_is_complex = true;
        }

        if (std::abs(blaze_complex_L(k, i).imag()) > tolerance) {
          left_eigenvector_is_complex = true;
        }
      }

      // If either eigenvector is complex, then we know that the vectors formed
      // by their real and imaginary parts are also linearly-independent
      // eigenvectors. Here, we use this fact to build real-valued left and
      // right eigenvectors.
      if (not processed_right_eigenvectors[i]) {
        // If needed, find index of complex conjugate eigenvector
        // Note: most time this will be i+1, but not always.
        size_t conjugate_i = 0;
        if (right_eigenvector_is_complex) {
          for (conjugate_i = i + 1; conjugate_i < matrix_size; ++conjugate_i) {
            // Assume that this eigenvector is the conjugate until we find
            // otherwise by checking each component
            bool is_conjugate = true;
            for (size_t k = 0; k < matrix_size; ++k) {
              const auto component = blaze_complex_R(k, i);
              const auto conjugate_component = blaze_complex_R(k, conjugate_i);
              if (std::abs(component.real() - conjugate_component.real()) >
                      tolerance or
                  std::abs(component.imag() + conjugate_component.imag()) >
                      tolerance) {
                is_conjugate = false;
                break;
              }
            }
            // If we made it through all components, then we've found the
            // conjugate eigenvector
            if (is_conjugate) {
              break;
            }
          }
        }
        ASSERT(not right_eigenvector_is_complex or
                   (conjugate_i > i and conjugate_i < matrix_size),
               "Found complex right eigenvector without identifying its "
               "conjugate partner.");
        // If the eigenvector is complex, then its respective eigenvalue must
        // be degenerate
        if (right_eigenvector_is_complex) {
          ASSERT(
              blaze_real_eigenvalues[i] - blaze_real_eigenvalues[conjugate_i] <
                  1.e-8,
              "Expected degenerate eigenvalues for complex eigenvectors, but "
              "eigenvalues differ by "
                  << blaze_real_eigenvalues[i] -
                         blaze_real_eigenvalues[conjugate_i]
                  << ".");
        }
        // Process the ith right eigenvector (and its complex conjugate if
        // needed)
        for (size_t k = 0; k < matrix_size; ++k) {
          gsl::at(*all_right_eigenvectors, i).get(k)[point] =
              blaze_complex_R(k, i).real();
          if (right_eigenvector_is_complex) {
            gsl::at(*all_right_eigenvectors, conjugate_i).get(k)[point] =
                blaze_complex_R(k, i).imag();
          }
        }
        processed_right_eigenvectors[i] = true;
        if (right_eigenvector_is_complex) {
          processed_right_eigenvectors[conjugate_i] = true;
        }
      }

      // Same as above, but for left eigenvectors
      if (not processed_left_eigenvectors[i]) {
        // If needed, find index of complex conjugate eigenvector
        // Note: most time this will be i+1, but not always.
        size_t conjugate_i = 0;
        if (left_eigenvector_is_complex) {
          for (conjugate_i = i + 1; conjugate_i < matrix_size; ++conjugate_i) {
            // Assume that this eigenvector is the conjugate until we find
            // otherwise by checking each component
            bool is_conjugate = true;
            for (size_t k = 0; k < matrix_size; ++k) {
              const auto component = blaze_complex_L(k, i);
              const auto conjugate_component = blaze_complex_L(k, conjugate_i);
              if (std::abs(component.real() - conjugate_component.real()) >
                      tolerance or
                  std::abs(component.imag() + conjugate_component.imag()) >
                      tolerance) {
                is_conjugate = false;
                break;
              }
            }
            // If we made it through all components, then we've found the
            // conjugate eigenvector
            if (is_conjugate) {
              break;
            }
          }
        }
        ASSERT(not left_eigenvector_is_complex or
                   (conjugate_i > i and conjugate_i < matrix_size),
               "Found complex left eigenvector without identifying its "
               "conjugate partner.");
        // If the eigenvector is complex, then its respective eigenvalue must
        // be degenerate
        if (left_eigenvector_is_complex) {
          ASSERT(
              blaze_real_eigenvalues[i] - blaze_real_eigenvalues[conjugate_i] <
                  1.e-8,
              "Expected degenerate eigenvalues for complex eigenvectors, but "
              "eigenvalues differ by "
                  << blaze_real_eigenvalues[i] -
                         blaze_real_eigenvalues[conjugate_i]
                  << ".");
        }
        // Process the ith left eigenvector (and its complex conjugate if
        // needed)
        for (size_t k = 0; k < matrix_size; ++k) {
          gsl::at(*all_left_eigenvectors, i).get(k)[point] =
              blaze_complex_L(k, i).real();
          if (left_eigenvector_is_complex) {
            gsl::at(*all_left_eigenvectors, conjugate_i).get(k)[point] =
                blaze_complex_L(k, i).imag();
          }
        }
        processed_left_eigenvectors[i] = true;
        if (left_eigenvector_is_complex) {
          processed_left_eigenvectors[conjugate_i] = true;
        }
      }
    }
  }
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

#define INSTANTIATION(r, data)                                                 \
  template std::array<DataVector, 9>                                           \
  characteristic_speeds_approximate_mhd<GET_DIM(data)>(                        \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_approximate_mhd<GET_DIM(data)>(          \
      const gsl::not_null<std::array<DataVector, 9>*> char_speeds,             \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
      const Scalar<DataVector>& lapse, const tnsr::I<DataVector, 3>& shift,    \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template std::array<DataVector, 3>                                           \
  characteristic_speeds_hydro<GET_DIM(data)>(                                  \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_hydro<GET_DIM(data)>(                    \
      const gsl::not_null<std::array<DataVector, 3>*> char_speeds,             \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void eigenvectors_hydro<GET_DIM(data)>(                             \
      const gsl::not_null<std::array<tnsr::i<DataVector, 6, Frame::Inertial>,  \
                                     6>*>& right_eigenvectors,                 \
      const gsl::not_null<std::array<tnsr::I<DataVector, 6, Frame::Inertial>,  \
                                     6>*>& left_eigenvectors,                  \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& kappa, const Scalar<DataVector>& zeta,         \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void detail::flux_jacobian_hydro<GET_DIM(data)>(                    \
      const gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,     \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,      \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void numerical_eigensystem<GET_DIM(data)>(                          \
      const gsl::not_null<std::array<Scalar<DataVector>, 6>*> all_eigenvalues, \
      const gsl::not_null<std::array<tnsr::i<DataVector, 6>, 6>*>              \
          all_right_eigenvectors,                                              \
      const gsl::not_null<std::array<tnsr::I<DataVector, 6>, 6>*>              \
          all_left_eigenvectors,                                               \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,      \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);

GENERATE_INSTANTIATIONS(INSTANTIATION, (1, 2, 3))

#undef GET_DIM
#undef INSTANTIATION
}  // namespace grmhd::ValenciaDivClean
