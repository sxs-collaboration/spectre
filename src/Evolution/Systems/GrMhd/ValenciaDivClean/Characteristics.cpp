// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <tuple>
#include <utility>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "NumericalAlgorithms/LinearAlgebra/FindRealEigensystem.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
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

// Compute the sound speed squared from the equation of state.
template <size_t ThermodynamicDim>
void sound_speed_squared_from_eos(
    const gsl::not_null<Scalar<DataVector>*> sound_speed_squared,
    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& specific_internal_energy,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& specific_enthalpy,
    const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
        equation_of_state) {
  if constexpr (ThermodynamicDim == 1) {
    get(*sound_speed_squared) =
        get(equation_of_state.chi_from_density(rest_mass_density)) +
        get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
            rest_mass_density));
    get(*sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 2) {
    get(*sound_speed_squared) =
        get(equation_of_state.chi_from_density_and_energy(
            rest_mass_density, specific_internal_energy)) +
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(*sound_speed_squared) /= get(specific_enthalpy);
  } else if constexpr (ThermodynamicDim == 3) {
    const auto temperature =
        equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction);
    get(*sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
  }
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
  sound_speed_squared_from_eos(make_not_null(&sound_speed_squared),
                               rest_mass_density, specific_internal_energy,
                               electron_fraction, specific_enthalpy,
                               equation_of_state);

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
  sound_speed_squared_from_eos(make_not_null(&sound_speed_squared),
                               rest_mass_density, specific_internal_energy,
                               electron_fraction, specific_enthalpy,
                               equation_of_state);

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
      ::Tags::TempScalar<13>, ::Tags::TempScalar<14>, ::Tags::TempScalar<15>,
      ::Tags::TempI<0, 3>, ::Tags::Tempi<0, 3>, ::Tags::TempI<1, 3>>>
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
    get(pressure) =
        get(equation_of_state.pressure_from_density(rest_mass_density));
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
    // Note: the 2D EoS interface only provides kappa p / rho^2, so we need a
    //       positive pressure to recover kappa.
    if (UNLIKELY(min(get(pressure)) <= 0.0)) {
      ERROR(
          "flux_jacobian_hydro with a 2D equation of state needs a positive "
          "pressure, but the minimum pressure is "
          << min(get(pressure)) << ".");
    }
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

  // The expressions in this function are written to:
  //   1. Avoid catastrophic cancellations by rewriting terms like W - 1 as,
  //      e.g., v^2 W^2 / (W + 1).
  //   2. Avoid repeated calculations by storing intermediate results in
  //      temporary variables and reusing previous allocations when possible.
  //   3. Be fast, as measured by the benchmark in the unit test.

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
  // Note: h - 1 is computed as eps + p / rho instead of from specific_enthalpy,
  //       so the two must be consistent, h = 1 + eps + p / rho.
  Scalar<DataVector>& specific_enthalpy_minus_one =
      get<::Tags::TempScalar<9>>(temp_tensors);
  get(specific_enthalpy_minus_one) =
      get(specific_internal_energy) + get(pressure) / get(rest_mass_density);
  // Check that the specific enthalpy is consistent with the equation of state.
  constexpr double enthalpy_tolerance = 1.0e-8;
  ASSERT(
      max(abs(get(specific_enthalpy) - 1.0 - get(specific_enthalpy_minus_one)) /
          get(specific_enthalpy)) <= enthalpy_tolerance,
      "The specific enthalpy is inconsistent with the equation of state: "
      "the largest relative difference between h and 1 + eps + p / rho is "
          << max(abs(get(specific_enthalpy) - 1.0 -
                     get(specific_enthalpy_minus_one)) /
                 get(specific_enthalpy))
          << ".");
  Scalar<DataVector>& velocity_squared =
      get<::Tags::TempScalar<10>>(temp_tensors);
  tenex::evaluate(make_not_null(&velocity_squared),
                  spatial_velocity(ti::I) * spatial_velocity_one_form(ti::i));
  Scalar<DataVector>& lorentz_factor_squared =
      get<::Tags::TempScalar<11>>(temp_tensors);
  get(lorentz_factor_squared) = square(get(lorentz_factor));
  Scalar<DataVector>& lorentz_factor_minus_one =
      get<::Tags::TempScalar<12>>(temp_tensors);
  get(lorentz_factor_minus_one) = get(velocity_squared) *
                                  get(lorentz_factor_squared) /
                                  (get(lorentz_factor) + 1.0);
  Scalar<DataVector>& specific_enthalpy_minus_lorentz_factor =
      get<::Tags::TempScalar<13>>(temp_tensors);
  get(specific_enthalpy_minus_lorentz_factor) =
      get(specific_enthalpy_minus_one) - get(lorentz_factor_minus_one);

  // Derivatives of Z = rho h W^2 with respect to the conserved variables
  // (D, S_i, tau, D Y_e), where S_i = Z v_i
  Scalar<DataVector>& inv_derivative_denom =
      get<::Tags::TempScalar<7>>(temp_tensors);
  get(inv_derivative_denom) =
      1.0 / ((-get(lorentz_factor_squared) + get(sound_speed_squared) *
                                                 get(velocity_squared) *
                                                 get(lorentz_factor_squared)) *
             get(rest_mass_density));
  Scalar<DataVector>& dz_d_density = get<::Tags::TempScalar<4>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dz_d_density),
      -(lorentz_factor() *
        (-kappa() * specific_enthalpy_minus_lorentz_factor() -
         zeta() * electron_fraction() +
         (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
             rest_mass_density()) *
        inv_derivative_denom()));
  tnsr::I<DataVector, 3>& dz_d_momentum =
      get<::Tags::TempI<1, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(
      make_not_null(&dz_d_momentum),
      (spatial_velocity(ti::I) * lorentz_factor_squared() *
       (kappa() + sound_speed_squared() * rest_mass_density()) *
       inv_derivative_denom()));
  Scalar<DataVector>& dz_d_energy = get<::Tags::TempScalar<5>>(temp_tensors);
  tenex::evaluate(make_not_null(&dz_d_energy),
                  -(lorentz_factor_squared() * (kappa() + rest_mass_density()) *
                    inv_derivative_denom()));
  // Note: dZ/dtau - 1 is computed directly to avoid cancellation, since
  //       dZ/dtau = 1 at rest when kappa = 0.
  Scalar<DataVector>& dz_d_energy_minus_one =
      get<::Tags::TempScalar<14>>(temp_tensors);
  tenex::evaluate(make_not_null(&dz_d_energy_minus_one),
                  -(lorentz_factor_squared() *
                    (kappa() + sound_speed_squared() * rest_mass_density() *
                                   velocity_squared()) *
                    inv_derivative_denom()));
  Scalar<DataVector>& dz_d_electron_fraction =
      get<::Tags::TempScalar<6>>(temp_tensors);
  tenex::evaluate(make_not_null(&dz_d_electron_fraction),
                  -(zeta() * lorentz_factor() * inv_derivative_denom()));

  // Common factors used repeatedly in the characteristic matrix entries
  Scalar<DataVector>& inv_enthalpy_times_lorentz_factor =
      get<::Tags::TempScalar<8>>(temp_tensors);
  get(inv_enthalpy_times_lorentz_factor) =
      1.0 / (get(specific_enthalpy) * get(lorentz_factor));
  // Put analytic expressions into characteristic matrix
  characteristic_matrix->get(0, 0) =
      (1.0 - get(inv_enthalpy_times_lorentz_factor) * get(dz_d_density)) *
      get(normal_velocity);
  characteristic_matrix->get(0, 1) =
      get(inv_enthalpy_times_lorentz_factor) *
      (unit_vector.get(0) - dz_d_momentum.get(0) * get(normal_velocity));
  characteristic_matrix->get(0, 2) =
      get(inv_enthalpy_times_lorentz_factor) *
      (unit_vector.get(1) - dz_d_momentum.get(1) * get(normal_velocity));
  characteristic_matrix->get(0, 3) =
      get(inv_enthalpy_times_lorentz_factor) *
      (unit_vector.get(2) - dz_d_momentum.get(2) * get(normal_velocity));
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
  characteristic_matrix->get(0, 4) = -(get(inv_enthalpy_times_lorentz_factor) *
                                       get(dz_d_energy) * get(normal_velocity));
  characteristic_matrix->get(0, 5) =
      -(get(inv_enthalpy_times_lorentz_factor) * get(dz_d_electron_fraction) *
        get(normal_velocity));
  // Momentum rows
  Scalar<DataVector>& normal_minus_velocity_term =
      get<::Tags::TempScalar<15>>(temp_tensors);
  for (size_t i = 0; i < 3; ++i) {
    // n_i - v_n v_i, shared by columns 1 to 3 and 5
    get(normal_minus_velocity_term) =
        unit_normal.get(i) -
        get(normal_velocity) * spatial_velocity_one_form.get(i);
    characteristic_matrix->get(i + 1, 0) =
        (-1.0 + get(dz_d_density)) * unit_normal.get(i) -
        get(dz_d_density) * get(normal_velocity) *
            spatial_velocity_one_form.get(i);
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        characteristic_matrix->get(i + 1, j + 1) =
            get(normal_velocity) +
            unit_vector.get(j) * spatial_velocity_one_form.get(i) +
            dz_d_momentum.get(j) * get(normal_minus_velocity_term);
      } else {
        characteristic_matrix->get(i + 1, j + 1) =
            unit_vector.get(j) * spatial_velocity_one_form.get(i) +
            dz_d_momentum.get(j) * get(normal_minus_velocity_term);
      }
    }
    // Note: this entry does not use n_i - v_n v_i, since
    //       -n_i + dZ/dtau (n_i - v_n v_i) would cancel at rest, where
    //       dZ/dtau = 1.
    characteristic_matrix->get(i + 1, 4) =
        get(dz_d_energy_minus_one) * unit_normal.get(i) -
        get(dz_d_energy) * get(normal_velocity) *
            spatial_velocity_one_form.get(i);
    characteristic_matrix->get(i + 1, 5) =
        get(dz_d_electron_fraction) * get(normal_minus_velocity_term);
  }
  characteristic_matrix->get(4, 0) = -characteristic_matrix->get(0, 0);
  characteristic_matrix->get(4, 4) = -characteristic_matrix->get(0, 4);
  characteristic_matrix->get(4, 5) = -characteristic_matrix->get(0, 5);
  characteristic_matrix->get(5, 0) = -get(electron_fraction) *
                                     get(inv_enthalpy_times_lorentz_factor) *
                                     get(dz_d_density) * get(normal_velocity);
  characteristic_matrix->get(5, 4) =
      get(electron_fraction) * characteristic_matrix->get(0, 4);
  characteristic_matrix->get(5, 5) =
      (1.0 - get(electron_fraction) * get(inv_enthalpy_times_lorentz_factor) *
                 get(dz_d_electron_fraction)) *
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
  // Build characteristic matrix.
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

  // Find the eigensystem of the characteristic matrix.
  find_real_eigensystem(characteristic_speeds, characteristic_modes,
                        characteristic_projectors, characteristic_matrix);
}

template <size_t ThermodynamicDim>
void characteristic_speeds_mhd(
    const gsl::not_null<tnsr::i<DataVector, 7>*> characteristic_speeds,

    /* primitive variables */
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
    const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,
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
  const size_t num_points = get(lorentz_factor).size();
  if (characteristic_speeds->get(0).size() != num_points) {
    for (size_t i = 0; i < 7; ++i) {
      characteristic_speeds->get(i) = DataVector(num_points, 0.0);
    }
  }

  // Use Variables to reduce total number of allocations
  Variables<tmpl::list<hydro::Tags::SoundSpeedSquared<DataVector>,
                       hydro::Tags::MagneticFieldOneForm<DataVector, 3>,
                       hydro::Tags::MagneticFieldSquared<DataVector>,
                       hydro::Tags::ComovingMagneticFieldSquared<DataVector>,
                       ::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                       ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                       ::Tags::TempScalar<4>, ::Tags::TempScalar<5>>>
      temp_tensors{num_points};

  // Get sound speed from EoS
  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  sound_speed_squared_from_eos(make_not_null(&sound_speed_squared),
                               rest_mass_density, specific_internal_energy,
                               electron_fraction, specific_enthalpy,
                               equation_of_state);

  // The system is hyperbolic only for 0 <= c_s^2 < 1 (see the documentation
  // of this function), which the solve below relies on.
  for (size_t point = 0; point < num_points; ++point) {
    const double cs2 = get(sound_speed_squared)[point];
    if (not(cs2 >= 0.0) or not(cs2 < 1.0)) {
      ERROR("The sound speed squared is outside [0, 1): c_s^2 = "
            << cs2 << " at point " << point << ", rest-mass density = "
            << get(rest_mass_density)[point] << ", specific internal energy = "
            << get(specific_internal_energy)[point]
            << ", specific enthalpy = " << get(specific_enthalpy)[point]
            << ". Either the equation of state is acausal here, or the "
               "caller passed a state that is not consistent with it.");
    }
  }

  // Entropy speed
  /*
    This is just the normal component of the fluid velocity. Since this quantity
    will be used later on, we also define normal_velocity, which points to the
    same DataVector as the entropy speed.
  */
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<0>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_velocity),
                  spatial_velocity(ti::I) * unit_normal(ti::i));
  get<MhdSpeed::Entropy>(*characteristic_speeds) = get(normal_velocity);

  // Alfven speeds, in mu = lambda - v_n.
  // Intermediate magnetic variables
  Scalar<DataVector>& normal_magnetic_field =
      get<::Tags::TempScalar<1>>(temp_tensors);
  tenex::evaluate(make_not_null(&normal_magnetic_field),
                  magnetic_field(ti::I) * unit_normal(ti::i));
  tnsr::i<DataVector, 3>& magnetic_field_one_form =
      get<hydro::Tags::MagneticFieldOneForm<DataVector, 3>>(temp_tensors);
  tenex::evaluate<ti::i>(make_not_null(&magnetic_field_one_form),
                         spatial_metric(ti::i, ti::j) * magnetic_field(ti::J));
  Scalar<DataVector>& magnetic_field_dot_spatial_velocity =
      get<::Tags::TempScalar<2>>(temp_tensors);
  tenex::evaluate(make_not_null(&magnetic_field_dot_spatial_velocity),
                  magnetic_field_one_form(ti::i) * spatial_velocity(ti::I));
  Scalar<DataVector>& magnetic_field_squared =
      get<hydro::Tags::MagneticFieldSquared<DataVector>>(temp_tensors);
  tenex::evaluate(make_not_null(&magnetic_field_squared),
                  magnetic_field_one_form(ti::i) * magnetic_field(ti::I));
  Scalar<DataVector>& comoving_magnetic_field_squared =
      get<hydro::Tags::ComovingMagneticFieldSquared<DataVector>>(temp_tensors);
  get(comoving_magnetic_field_squared) =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));
  Scalar<DataVector>& rho_h_star = get<::Tags::TempScalar<3>>(temp_tensors);
  get(rho_h_star) = get(rest_mass_density) * get(specific_enthalpy) +
                    get(comoving_magnetic_field_squared);
  // Ordered by the sign of each speed (see the documentation).
  DataVector& mu_alfven_minus = get(get<::Tags::TempScalar<4>>(temp_tensors));
  DataVector& mu_alfven_plus = get(get<::Tags::TempScalar<5>>(temp_tensors));
  mu_alfven_plus =
      abs(get(normal_magnetic_field)) /
      (square(get(lorentz_factor)) *
       (sqrt(get(rho_h_star)) + sign(get(normal_magnetic_field)) *
                                    get(magnetic_field_dot_spatial_velocity)));
  mu_alfven_minus =
      -abs(get(normal_magnetic_field)) /
      (square(get(lorentz_factor)) *
       (sqrt(get(rho_h_star)) - sign(get(normal_magnetic_field)) *
                                    get(magnetic_field_dot_spatial_velocity)));
  get<MhdSpeed::AlfvenMinus>(*characteristic_speeds) =
      get(normal_velocity) + mu_alfven_minus;
  get<MhdSpeed::AlfvenPlus>(*characteristic_speeds) =
      get(normal_velocity) + mu_alfven_plus;

  // A root of f between x_minus and x_plus, given f < 0 between x_minus and
  // the root and f > 0 between the root and x_plus: Newton-Raphson safeguarded
  // by bisection, based on rtsafe of Numerical Recipes (3rd ed., Sec. 9.4).
  // `f_and_df(x)` returns f(x) and f'(x).
  //
  // RootFinder::toms748 is not used because it rejects ends where f has the
  // same sign and returns an end where f vanishes. At the degenerate states
  // described below, f vanishes at the Alfven end up to round-off, so toms748
  // could fail, or return a slow speed in place of the fast one.
  //
  // Unlike rtsafe, the signs of f at the ends are not checked first. Here they
  // are known from the theory, and at a degenerate state f vanishes at an end
  // of the bracket, where round-off can give it either sign. A check would
  // then reject a valid bracket. f is only evaluated inside the bracket, so
  // its value at the ends is never needed.
  const auto safe_newton = [](const auto& f_and_df, double x_minus,
                              double x_plus) {
    constexpr double tolerance = 1.0e-14;
    constexpr size_t max_iterations = 100;
    double x = 0.5 * (x_minus + x_plus);
    double delta = x_plus - x_minus;
    auto [f, df] = f_and_df(x);
    for (size_t iteration = 0; iteration < max_iterations; ++iteration) {
      const double delta_old = delta;
      // Check if the Newton step would leave the bracket
      const bool leaves_bracket =
          ((x - x_minus) * df - f) * ((x - x_plus) * df - f) > 0.0;
      // Check if the Newton step would shrink the bracket slower than bisection
      const bool too_slow = std::abs(2.0 * f) > std::abs(delta_old * df);
      if (leaves_bracket or too_slow or df == 0.0) {
        // Bisection
        delta = 0.5 * (x_plus - x_minus);
        x = x_minus + delta;
      } else {
        // Newton
        delta = f / df;
        x -= delta;
      }
      if (std::abs(delta) < tolerance) {
        return x;
      }
      std::tie(f, df) = f_and_df(x);
      if (f < 0.0) {
        x_minus = x;
      } else {
        x_plus = x;
      }
    }
    ERROR("Failed to find a magnetosonic speed: no convergence in "
          << max_iterations << " iterations. x = " << x << ", bracket = ["
          << x_minus << ", " << x_plus << "]");
  };

  // Magnetosonic speeds
  /*
    Solved in mu = lambda - v_n as described in the documentation of this
    function: the fast speeds by `safe_newton` on Q(mu) between the Alfven
    speeds and the speed of light, the slow speeds by deflation.
  */
  const DataVector& vn = get(normal_velocity);
  for (size_t point = 0; point < num_points; ++point) {
    const double cs2 = get(sound_speed_squared)[point];
    const double vn_i = vn[point];
    const double w = get(lorentz_factor)[point];
    const double w2 = square(w);
    // Magnetic quantities divided by sqrt(rho h), i.e., the tilde quantities
    // of the documentation (bn, bv and b2 for tilde B_n, tilde B_v and
    // tilde b^2, and b_q below for tilde B)
    const double inv_rho_h =
        1.0 / (get(rest_mass_density)[point] * get(specific_enthalpy)[point]);
    const double inv_sqrt_rho_h = sqrt(inv_rho_h);
    const double bn = get(normal_magnetic_field)[point] * inv_sqrt_rho_h;
    const double bv =
        get(magnetic_field_dot_spatial_velocity)[point] * inv_sqrt_rho_h;
    const double b2 = get(comoving_magnetic_field_squared)[point] * inv_rho_h;
    const double one_minus_vn = 1.0 - vn_i;
    const double one_plus_vn = 1.0 + vn_i;

    // Coefficients tilde N_4, tilde N_1 and tilde N_0 of tilde N(mu)
    const double n4 = -w2 * ((1.0 - cs2) * w2 + b2 + cs2 * (1.0 - square(bv)));
    const double n1 = 2.0 * cs2 * bn / w2 *
                      (bv * one_minus_vn * one_plus_vn * w2 + vn_i * bn);
    const double n0 = -cs2 * square(bn) * one_minus_vn * one_plus_vn / w2;
    const double inv_n4 = 1.0 / n4;

    // Q(mu) = tilde N(mu) / tilde N_4 and Q'(mu), from the factors a, B and G
    const auto q_and_dq = [&](const double mu) {
      const double g = (one_minus_vn - mu) * (one_plus_vn + mu);
      const double dg = (one_minus_vn - mu) - (one_plus_vn + mu);
      const double a2 = w2 * square(mu);
      const double da2 = 2.0 * w2 * mu;
      const double b_q = bn / w - w * bv * mu;
      const double db_q = -w * bv;
      const double n =
          a2 * g * b2 - square(a2) + cs2 * (a2 * (a2 + g) - square(b_q) * g);
      const double dn = (da2 * g + a2 * dg) * b2 - 2.0 * a2 * da2 +
                        cs2 * (da2 * (a2 + g) + a2 * (da2 + dg) -
                               2.0 * b_q * db_q * g - square(b_q) * dg);
      return std::pair{n * inv_n4, dn * inv_n4};
    };

    // Fast speeds, each between an Alfven speed and the speed of light.
    // Degenerate states (Anton et al. 2010, Sec. 3) need no special treatment:
    // - When Alfven = fast, Q(mu) > 0 inside the bracket, so the bracket
    //   closes onto the Alfven speed, which is then returned.
    // - When Alfven = slow, Q(mu) < 0 just inside the bracket, so the solve
    //   finds the fast speed as usual (the deflation below then recovers the
    //   slow one).
    const double mu_fast_plus =
        safe_newton(q_and_dq, mu_alfven_plus[point], one_minus_vn);
    const double mu_fast_minus =
        safe_newton(q_and_dq, mu_alfven_minus[point], -one_plus_vn);

    // Slow speeds: the quadratic mu^2 + q_1 mu + q_0 left after dividing Q by
    // the fast roots, computed from the constant term up (backward deflation,
    // stable because the fast speeds are the roots of largest |mu|). This
    // divides by the product of the fast speeds, which is zero only for a cold,
    // unmagnetized fluid (c_s^2 = b^2 = 0). There every speed is v_n, so the
    // slow speeds are left at v_n.
    const double product = mu_fast_minus * mu_fast_plus;
    double mu_slow_minus = 0.0;
    double mu_slow_plus = 0.0;
    if (product != 0.0) {
      const double q0 = n0 * inv_n4 / product;
      const double q1 =
          (n1 * inv_n4 + (mu_fast_minus + mu_fast_plus) * q0) / product;
      const double discriminant = std::max(square(q1) - 4.0 * q0, 0.0);
      // The root of larger magnitude, without cancellation
      const double r =
          -0.5 * (q1 + (q1 >= 0.0 ? 1.0 : -1.0) * std::sqrt(discriminant));
      if (r != 0.0) {
        mu_slow_minus = std::min(r, q0 / r);
        mu_slow_plus = std::max(r, q0 / r);
      }
    }

    get<MhdSpeed::FastMagnetosonicMinus>(*characteristic_speeds)[point] =
        vn_i + mu_fast_minus;
    get<MhdSpeed::SlowMagnetosonicMinus>(*characteristic_speeds)[point] =
        vn_i + mu_slow_minus;
    get<MhdSpeed::SlowMagnetosonicPlus>(*characteristic_speeds)[point] =
        vn_i + mu_slow_plus;
    get<MhdSpeed::FastMagnetosonicPlus>(*characteristic_speeds)[point] =
        vn_i + mu_fast_plus;
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
  template void characteristic_speeds_mhd<GET_DIM(data)>(                      \
      const gsl::not_null<tnsr::i<DataVector, 7>*> characteristic_speeds,      \
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,         \
      const tnsr::I<DataVector, 3, Frame::Inertial>& magnetic_field,           \
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
