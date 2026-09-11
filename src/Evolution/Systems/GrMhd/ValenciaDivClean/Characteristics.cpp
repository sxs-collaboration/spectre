// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <algorithm>
#include <array>
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

template <size_t ThermodynamicDim>
void characteristic_speeds_hydro(
    const gsl::not_null<tnsr::i<DataVector, 3>*> characteristic_speeds,
    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    /* other helpful quantities */
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  const size_t num_grid_points = get(lorentz_factor).size();
  if (characteristic_speeds->get(0).size() != num_grid_points) {
    for (size_t i = 0; i < 3; ++i) {
      characteristic_speeds->get(i) = DataVector(num_grid_points, 0.0);
    }
  }

  Variables<tmpl::list<hydro::Tags::SpatialVelocityOneForm<DataVector, 3>,
                       hydro::Tags::SpatialVelocitySquared<DataVector>,
                       hydro::Tags::Temperature<DataVector>,
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
    Scalar<DataVector>& temperature =
        get<hydro::Tags::Temperature<DataVector>>(temp_tensors);
    get(temperature) =
        get(equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction));
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

  // Degenerate characteristic speed (normal dot velocity)
  characteristic_speeds->get(HydroSpeed::NormalDotVelocity) =
      get(normal_velocity);
  characteristic_speeds->get(HydroSpeed::LambdaPlus) =
      get(first_term) + get(second_term);
  characteristic_speeds->get(HydroSpeed::LambdaMinus) =
      get(first_term) - get(second_term);
}


template <size_t ThermodynamicDim>
tnsr::i<DataVector, 3> characteristic_speeds_hydro(
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& lorentz_factor,
    const Scalar<DataVector>& specific_enthalpy,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::i<DataVector, 3>& unit_normal,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  tnsr::i<DataVector, 3> characteristic_speeds{};
  characteristic_speeds_hydro(make_not_null(&characteristic_speeds),
                              spatial_velocity, rest_mass_density,
                              specific_internal_energy, electron_fraction,
                              lorentz_factor, specific_enthalpy, spatial_metric,
                              unit_normal, equation_of_state);
  return characteristic_speeds;
}

template <size_t ThermodynamicDim>
void flux_jacobian_hydro(
    const gsl::not_null<tnsr::iJ<DataVector, 6>*> characteristic_matrix,
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
  // Use Variables to reduce total number of allocations
  Variables<tmpl::list<
      hydro::Tags::SoundSpeedSquared<DataVector>,
      hydro::Tags::Pressure<DataVector>, ::Tags::TempScalar<0>,
      ::Tags::TempScalar<1>, ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
      ::Tags::TempScalar<4>, ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
      ::Tags::TempScalar<7>, ::Tags::TempScalar<8>, ::Tags::TempScalar<9>,
      ::Tags::TempScalar<10>, ::Tags::TempScalar<11>, ::Tags::TempScalar<12>,
      ::Tags::TempScalar<13>, ::Tags::TempI<0, 3>, ::Tags::Tempi<0, 3>,
      ::Tags::TempI<1, 3>>>
      temp_tensors{get<0, 0>(spatial_metric).size()};

  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  // We define kappa as the partial derivative of pressure with respect to
  // specific internal energy
  Scalar<DataVector>& kappa = get<::Tags::TempScalar<0>>(temp_tensors);
  // We define zeta as the partial derivative of pressure with respect to
  // electron fraction
  Scalar<DataVector>& zeta = get<::Tags::TempScalar<1>>(temp_tensors);
  Scalar<DataVector>& pressure =
      get<hydro::Tags::Pressure<DataVector>>(temp_tensors);
  if constexpr (ThermodynamicDim == 1) {
    get(sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(sound_speed_squared) /= get(specific_enthalpy);
    get(kappa) = 0.0;
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    Scalar<DataVector>& kappa_times_p_over_rho_squared =
        get<::Tags::TempScalar<2>>(temp_tensors);
    get(kappa_times_p_over_rho_squared) =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    get(pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
    get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
                 square(get(rest_mass_density));
    get(zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    Scalar<DataVector>& temperature = get<::Tags::TempScalar<2>>(temp_tensors);
    get(temperature) =
        get(equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction));
    get(sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
    get(pressure) = get(equation_of_state.pressure_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
    get(kappa) = get(equation_of_state.kappa_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
    get(zeta) = get(equation_of_state.zeta_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
  }

  // The expressions in this function have been iteratively optimized by Codex
  // with 3 main goals:
  //   1. Avoid catastrophic cancellations by rewriting terms like W-1 to, e.g.,
  //      v^2 W^2 / (W + 1). Known tricks were listed in a skill used by Codex.
  //   2. Avoid repeated calculations by storing intermediate results in
  //      temporary variables and reusing previous allocations when possible.
  //   3. Try different rearrangements and run benchmark to find the best
  //      optimization.

  // Intermediate variables
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<3>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_velocity),
                  spatial_velocity(ti::I) * unit_normal(ti::i));
  tnsr::I<DataVector, 3>& unit_vector = get<::Tags::TempI<0, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(make_not_null(&unit_vector),
                         inv_spatial_metric(ti::I, ti::J) * unit_normal(ti::j));
  tnsr::i<DataVector, 3>& spatial_velocity_one_form =
      get<::Tags::Tempi<0, 3>>(temp_tensors);
  tenex::evaluate<ti::i>(
      make_not_null(&spatial_velocity_one_form),
      spatial_metric(ti::i, ti::j) * spatial_velocity(ti::J));

  // Cancellation-safe thermodynamic / kinematic differences
  Scalar<DataVector>& h_minus_1 = get<::Tags::TempScalar<9>>(temp_tensors);
  get(h_minus_1) =
      get(specific_internal_energy) + get(pressure) / get(rest_mass_density);
  Scalar<DataVector>& velocity_squared =
      get<::Tags::TempScalar<10>>(temp_tensors);
  tenex::evaluate(make_not_null(&velocity_squared),
                  spatial_velocity(ti::I) * spatial_velocity_one_form(ti::i));
  Scalar<DataVector>& lorentz_factor_squared =
      get<::Tags::TempScalar<11>>(temp_tensors);
  get(lorentz_factor_squared) = square(get(lorentz_factor));
  Scalar<DataVector>& W_minus_1 = get<::Tags::TempScalar<12>>(temp_tensors);
  get(W_minus_1) = get(velocity_squared) * get(lorentz_factor_squared) /
                   (get(lorentz_factor) + 1.0);
  Scalar<DataVector>& h_minus_W = get<::Tags::TempScalar<13>>(temp_tensors);
  get(h_minus_W) = get(h_minus_1) - get(W_minus_1);

  // Derivatives of Z
  Scalar<DataVector>& inv_derivative_denom =
      get<::Tags::TempScalar<7>>(temp_tensors);
  get(inv_derivative_denom) =
      1.0 / ((-get(lorentz_factor_squared) + get(sound_speed_squared) *
                                                 get(velocity_squared) *
                                                 get(lorentz_factor_squared)) *
             get(rest_mass_density));
  Scalar<DataVector>& dzdD = get<::Tags::TempScalar<4>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdD),
      -(lorentz_factor() *
        (-kappa() * h_minus_W() - zeta() * electron_fraction() +
         (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
             rest_mass_density()) *
        inv_derivative_denom()));
  tnsr::I<DataVector, 3>& dzds = get<::Tags::TempI<1, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(
      make_not_null(&dzds),
      (spatial_velocity(ti::I) * lorentz_factor_squared() *
       (kappa() + sound_speed_squared() * rest_mass_density()) *
       inv_derivative_denom()));
  Scalar<DataVector>& dzdtau = get<::Tags::TempScalar<5>>(temp_tensors);
  tenex::evaluate(make_not_null(&dzdtau),
                  -(lorentz_factor_squared() * (kappa() + rest_mass_density()) *
                    inv_derivative_denom()));
  Scalar<DataVector>& dzdye = get<::Tags::TempScalar<6>>(temp_tensors);
  tenex::evaluate(make_not_null(&dzdye),
                  -(zeta() * lorentz_factor() * inv_derivative_denom()));

  // Common factors used repeatedly in the characteristic matrix entries
  Scalar<DataVector>& D_over_Z = get<::Tags::TempScalar<8>>(temp_tensors);
  get(D_over_Z) = 1.0 / (get(specific_enthalpy) * get(lorentz_factor));
  // Put analytic expressions into characteristic matrix
  characteristic_matrix->get(0, 0) =
      (1.0 - get(D_over_Z) * get(dzdD)) * get(normal_velocity);
  characteristic_matrix->get(0, 1) =
      get(D_over_Z) * (unit_vector.get(0) - dzds.get(0) * get(normal_velocity));
  characteristic_matrix->get(0, 2) =
      get(D_over_Z) * (unit_vector.get(1) - dzds.get(1) * get(normal_velocity));
  characteristic_matrix->get(0, 3) =
      get(D_over_Z) * (unit_vector.get(2) - dzds.get(2) * get(normal_velocity));
  characteristic_matrix->get(4, 1) =
      unit_vector.get(0) - characteristic_matrix->get(0, 1);
  characteristic_matrix->get(4, 2) =
      unit_vector.get(1) - characteristic_matrix->get(0, 2);
  characteristic_matrix->get(4, 3) =
      unit_vector.get(2) - characteristic_matrix->get(0, 3);
  characteristic_matrix->get(5, 1) =
      get(electron_fraction) * characteristic_matrix->get(0, 1);
  characteristic_matrix->get(5, 2) =
      get(electron_fraction) * characteristic_matrix->get(0, 2);
  characteristic_matrix->get(5, 3) =
      get(electron_fraction) * characteristic_matrix->get(0, 3);
  characteristic_matrix->get(0, 4) =
      -(get(D_over_Z) * get(dzdtau) * get(normal_velocity));
  characteristic_matrix->get(0, 5) =
      -(get(D_over_Z) * get(dzdye) * get(normal_velocity));
  characteristic_matrix->get(1, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(0) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(0);
  characteristic_matrix->get(2, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(1) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(1);
  characteristic_matrix->get(3, 0) =
      (-1.0 + get(dzdD)) * unit_normal.get(2) -
      get(dzdD) * get(normal_velocity) * spatial_velocity_one_form.get(2);

  characteristic_matrix->get(1, 1) =
      get(normal_velocity) +
      unit_vector.get(0) * spatial_velocity_one_form.get(0) +
      dzds.get(0) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(1, 2) =
      unit_vector.get(1) * spatial_velocity_one_form.get(0) +
      dzds.get(1) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(1, 3) =
      unit_vector.get(2) * spatial_velocity_one_form.get(0) +
      dzds.get(2) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 1) =
      unit_vector.get(0) * spatial_velocity_one_form.get(1) +
      dzds.get(0) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(2, 2) =
      get(normal_velocity) +
      unit_vector.get(1) * spatial_velocity_one_form.get(1) +
      dzds.get(1) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(2, 3) =
      unit_vector.get(2) * spatial_velocity_one_form.get(1) +
      dzds.get(2) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 1) =
      unit_vector.get(0) * spatial_velocity_one_form.get(2) +
      dzds.get(0) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(3, 2) =
      unit_vector.get(1) * spatial_velocity_one_form.get(2) +
      dzds.get(1) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(3, 3) =
      get(normal_velocity) +
      unit_vector.get(2) * spatial_velocity_one_form.get(2) +
      dzds.get(2) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));

  characteristic_matrix->get(1, 4) =
      -unit_normal.get(0) +
      get(dzdtau) * (unit_normal.get(0) -
                     get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 4) =
      -unit_normal.get(1) +
      get(dzdtau) * (unit_normal.get(1) -
                     get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 4) =
      -unit_normal.get(2) +
      get(dzdtau) * (unit_normal.get(2) -
                     get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(1, 5) =
      get(dzdye) * (unit_normal.get(0) -
                    get(normal_velocity) * spatial_velocity_one_form.get(0));
  characteristic_matrix->get(2, 5) =
      get(dzdye) * (unit_normal.get(1) -
                    get(normal_velocity) * spatial_velocity_one_form.get(1));
  characteristic_matrix->get(3, 5) =
      get(dzdye) * (unit_normal.get(2) -
                    get(normal_velocity) * spatial_velocity_one_form.get(2));
  characteristic_matrix->get(4, 0) = -characteristic_matrix->get(0, 0);
  characteristic_matrix->get(4, 4) = -characteristic_matrix->get(0, 4);
  characteristic_matrix->get(4, 5) = -characteristic_matrix->get(0, 5);
  characteristic_matrix->get(5, 0) = -get(electron_fraction) * get(D_over_Z) *
                                     get(dzdD) * get(normal_velocity);
  characteristic_matrix->get(5, 4) =
      get(electron_fraction) * characteristic_matrix->get(0, 4);
  characteristic_matrix->get(5, 5) =
      (1.0 - get(electron_fraction) * get(D_over_Z) * get(dzdye)) *
      get(normal_velocity);
}

template <size_t ThermodynamicDim>
void numerical_characteristics(
    const gsl::not_null<tnsr::i<DataVector, 6>*> characteristic_speeds,
    const gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,
    const gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors,
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
  Variables<tmpl::list<::Tags::TempiJ<0, 6>>> temp_tensors{
      get<0, 0>(spatial_metric).size()};
  tnsr::iJ<DataVector, 6>& characteristic_matrix =
      get<::Tags::TempiJ<0, 6>>(temp_tensors);
  flux_jacobian_hydro(make_not_null(&characteristic_matrix), spatial_velocity,
                      rest_mass_density, specific_internal_energy,
                      electron_fraction,
                      /* other helpful quantities */
                      lorentz_factor, specific_enthalpy, spatial_metric,
                      inv_spatial_metric, unit_normal, equation_of_state);

  // Allocate memory to work with blaze::geev (outside of loop to save
  // time/memory)
  constexpr size_t matrix_size = 6;
  blaze::StaticMatrix<double, matrix_size, matrix_size> blaze_point_matrix{};
  blaze::StaticVector<blaze::complex<double>, matrix_size>
      blaze_complex_eigenvalues{};
  blaze::StaticMatrix<blaze::complex<double>, matrix_size, matrix_size>
      blaze_complex_L{};
  blaze::StaticMatrix<blaze::complex<double>, matrix_size, matrix_size>
      blaze_complex_R{};
  std::array<double, matrix_size> blaze_real_eigenvalues{};

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
      ASSERT(std::abs(blaze_complex_eigenvalues.at(i).imag()) < tolerance,
             "Complex eigenvalue: "
                 << blaze_complex_eigenvalues.at(i).real() << " + "
                 << blaze_complex_eigenvalues.at(i).imag() << " i.");
      gsl::at(blaze_real_eigenvalues, i) =
          blaze_complex_eigenvalues.at(i).real();
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
    std::array<double, matrix_size> gsl_real_eigenvalues{};

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
      gsl::at(gsl_real_eigenvalues, i) =
          GSL_REAL(gsl_vector_complex_get(gsl_complex_eigenvalues, i));
    }

    // Sort both blaze's and GSL's eigenvalues to facilitate comparison
    std::array<double, matrix_size> sorted_blaze_real_eigenvalues =
        blaze_real_eigenvalues;
    std::sort(sorted_blaze_real_eigenvalues.begin(),
              sorted_blaze_real_eigenvalues.end());
    std::sort(gsl_real_eigenvalues.begin(), gsl_real_eigenvalues.end());

    // Compare eigenvalues
    for (size_t i = 0; i < matrix_size; ++i) {
      if (UNLIKELY(std::abs(gsl::at(sorted_blaze_real_eigenvalues, i) -
                            gsl::at(gsl_real_eigenvalues, i)) > 1.e-6)) {
        std::ostringstream matrix_stream;
        matrix_stream << "Point matrix:\n";
        for (size_t row = 0; row < matrix_size; ++row) {
          for (size_t col = 0; col < matrix_size; ++col) {
            matrix_stream << blaze_point_matrix(row, col) << " ";
          }
          matrix_stream << "\n";
        }
        matrix_stream << "Blaze eigenvalue: "
                      << gsl::at(sorted_blaze_real_eigenvalues, i)
                      << ", GSL eigenvalue: "
                      << gsl::at(gsl_real_eigenvalues, i) << "\n";

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
    std::array<bool, matrix_size> processed_right_eigenvectors{};
    std::array<bool, matrix_size> processed_left_eigenvectors{};
    // Note: we're looping through the ith eigenvalue/vectors, not the ith row!
    for (size_t i = 0; i < matrix_size; ++i) {
      // Store eigenvalue
      characteristic_speeds->get(i)[point] = gsl::at(blaze_real_eigenvalues, i);

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
      if (not gsl::at(processed_right_eigenvectors, i)) {
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
              gsl::at(blaze_real_eigenvalues, i) -
                      gsl::at(blaze_real_eigenvalues, conjugate_i) <
                  1.e-8,
              "Expected degenerate eigenvalues for complex eigenvectors, but "
              "eigenvalues differ by "
                  << gsl::at(blaze_real_eigenvalues, i) -
                         gsl::at(blaze_real_eigenvalues, conjugate_i)
                  << ".");
        }
        // Process the ith right eigenvector (and its complex conjugate if
        // needed)
        for (size_t k = 0; k < matrix_size; ++k) {
          characteristic_modes->get(i, k)[point] = blaze_complex_R(k, i).real();
          if (right_eigenvector_is_complex) {
            characteristic_modes->get(conjugate_i, k)[point] =
                blaze_complex_R(k, i).imag();
          }
        }
        gsl::at(processed_right_eigenvectors, i) = true;
        if (right_eigenvector_is_complex) {
          gsl::at(processed_right_eigenvectors, conjugate_i) = true;
        }
      }

      // Same as above, but for left eigenvectors
      if (not gsl::at(processed_left_eigenvectors, i)) {
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
              gsl::at(blaze_real_eigenvalues, i) -
                      gsl::at(blaze_real_eigenvalues, conjugate_i) <
                  1.e-8,
              "Expected degenerate eigenvalues for complex eigenvectors, but "
              "eigenvalues differ by "
                  << gsl::at(blaze_real_eigenvalues, i) -
                         gsl::at(blaze_real_eigenvalues, conjugate_i)
                  << ".");
        }
        // Process the ith left eigenvector (and its complex conjugate if
        // needed)
        for (size_t k = 0; k < matrix_size; ++k) {
          characteristic_projectors->get(i, k)[point] =
              blaze_complex_L(k, i).real();
          if (left_eigenvector_is_complex) {
            characteristic_projectors->get(conjugate_i, k)[point] =
                blaze_complex_L(k, i).imag();
          }
        }
        gsl::at(processed_left_eigenvectors, i) = true;
        if (left_eigenvector_is_complex) {
          gsl::at(processed_left_eigenvectors, conjugate_i) = true;
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
  template tnsr::i<DataVector, 3> characteristic_speeds_hydro<GET_DIM(data)>(  \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void characteristic_speeds_hydro<GET_DIM(data)>(                    \
      const gsl::not_null<tnsr::i<DataVector, 3>*> characteristic_speeds,      \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const Scalar<DataVector>& rest_mass_density,                             \
      const Scalar<DataVector>& specific_internal_energy,                      \
      const Scalar<DataVector>& electron_fraction,                             \
      const Scalar<DataVector>& lorentz_factor,                                \
      const Scalar<DataVector>& specific_enthalpy,                             \
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,          \
      const tnsr::i<DataVector, 3>& unit_normal,                               \
      const EquationsOfState::EquationOfState<true, GET_DIM(data)>&            \
          equation_of_state);                                                  \
  template void flux_jacobian_hydro<GET_DIM(data)>(                            \
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
  template void numerical_characteristics<GET_DIM(data)>(                      \
      const gsl::not_null<tnsr::i<DataVector, 6>*> characteristic_speeds,      \
      const gsl::not_null<tnsr::ij<DataVector, 6>*> characteristic_modes,      \
      const gsl::not_null<tnsr::IJ<DataVector, 6>*> characteristic_projectors, \
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
