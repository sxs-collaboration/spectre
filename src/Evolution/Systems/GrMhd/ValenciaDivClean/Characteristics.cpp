// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"
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
#include "Utilities/SetNumberOfGridPoints.hpp"

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

// Compute the sound speed squared, the pressure, kappa (the partial derivative
// of the pressure with respect to the specific internal energy), and zeta (the
// partial derivative of the pressure with respect to the electron fraction)
// from the equation of state, and check that the specific enthalpy is
// consistent with it. `buffer` is used as a temporary.
// Note 1: kappa and zeta vanish for a 1D equation of state, and zeta vanishes
//         for a 2D one.
// Note 2: the callers compute h - 1 as eps + p / rho instead of from the
//         specific enthalpy, which is why the two must be consistent.
template <size_t ThermodynamicDim>
void thermodynamic_derivatives_from_eos(
    const gsl::not_null<Scalar<DataVector>*> sound_speed_squared,
    const gsl::not_null<Scalar<DataVector>*> pressure,
    const gsl::not_null<Scalar<DataVector>*> kappa,
    const gsl::not_null<Scalar<DataVector>*> zeta,
    const gsl::not_null<Scalar<DataVector>*> buffer,
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
    get(*pressure) =
        get(equation_of_state.pressure_from_density(rest_mass_density));
    get(*kappa) = 0.0;
    get(*zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 2) {
    Scalar<DataVector>& kappa_times_p_over_rho_squared = *buffer;
    get(kappa_times_p_over_rho_squared) =
        get(equation_of_state
                .kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy));
    get(*sound_speed_squared) =
        (get(equation_of_state.chi_from_density_and_energy(
             rest_mass_density, specific_internal_energy)) +
         get(kappa_times_p_over_rho_squared)) /
        get(specific_enthalpy);
    get(*pressure) = get(equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy));
    // Note: the 2D EoS interface only provides kappa p / rho^2, so we need a
    //       positive pressure to recover kappa.
    if (UNLIKELY(min(get(*pressure)) <= 0.0)) {
      ERROR(
          "The hydro characteristics with a 2D equation of state need a "
          "positive pressure, but the minimum pressure is "
          << min(get(*pressure)) << ".");
    }
    get(*kappa) = get(kappa_times_p_over_rho_squared) / get(*pressure) *
                  square(get(rest_mass_density));
    get(*zeta) = 0.0;
  } else if constexpr (ThermodynamicDim == 3) {
    Scalar<DataVector>& temperature = *buffer;
    get(temperature) =
        get(equation_of_state.temperature_from_density_and_energy(
            rest_mass_density, specific_internal_energy, electron_fraction));
    get(*sound_speed_squared) =
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
    get(*pressure) =
        get(equation_of_state.pressure_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction));
    get(*kappa) = get(equation_of_state.kappa_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
    get(*zeta) = get(equation_of_state.zeta_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
  }
  // Check that the specific enthalpy is consistent with the equation of state.
  constexpr double enthalpy_tolerance = 1.0e-8;
  ASSERT(max(abs(get(specific_enthalpy) - 1.0 - get(specific_internal_energy) -
                 get(*pressure) / get(rest_mass_density)) /
             get(specific_enthalpy)) <= enthalpy_tolerance,
         "The specific enthalpy is inconsistent with the equation of state: "
         "the largest relative difference between h and 1 + eps + p / rho is "
             << max(abs(get(specific_enthalpy) - 1.0 -
                        get(specific_internal_energy) -
                        get(*pressure) / get(rest_mass_density)) /
                    get(specific_enthalpy))
             << ".");
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
  thermodynamic_derivatives_from_eos(
      make_not_null(&sound_speed_squared), make_not_null(&pressure),
      make_not_null(&kappa), make_not_null(&zeta),
      make_not_null(&get<::Tags::TempScalar<2>>(temp_tensors)),
      rest_mass_density, specific_internal_energy, electron_fraction,
      specific_enthalpy, equation_of_state);

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
void analytic_characteristics(
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
  const size_t num_points = get(lorentz_factor).size();
  set_number_of_grid_points(characteristic_speeds, num_points);
  set_number_of_grid_points(characteristic_modes, num_points);
  set_number_of_grid_points(characteristic_projectors, num_points);

  // Use Variables to reduce total number of allocations
  Variables<tmpl::list<
      hydro::Tags::SoundSpeedSquared<DataVector>,
      hydro::Tags::Pressure<DataVector>, ::Tags::TempScalar<0>,
      ::Tags::TempScalar<1>, ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
      ::Tags::TempScalar<4>, ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
      ::Tags::TempScalar<7>, ::Tags::TempScalar<8>, ::Tags::TempScalar<9>,
      ::Tags::TempScalar<10>, ::Tags::TempScalar<11>, ::Tags::TempScalar<12>,
      ::Tags::TempScalar<13>, ::Tags::TempScalar<14>, ::Tags::TempScalar<15>,
      ::Tags::TempScalar<16>, ::Tags::TempScalar<17>, ::Tags::Tempi<0, 3>,
      ::Tags::Tempi<1, 3>, ::Tags::Tempi<2, 3>, ::Tags::TempI<0, 3>,
      ::Tags::TempI<1, 3>, ::Tags::TempI<2, 3>>>
      temp_tensors{num_points};

  // Thermodynamic quantities
  Scalar<DataVector>& sound_speed_squared =
      get<hydro::Tags::SoundSpeedSquared<DataVector>>(temp_tensors);
  Scalar<DataVector>& pressure =
      get<hydro::Tags::Pressure<DataVector>>(temp_tensors);
  Scalar<DataVector>& kappa = get<::Tags::TempScalar<0>>(temp_tensors);
  Scalar<DataVector>& zeta = get<::Tags::TempScalar<1>>(temp_tensors);
  Scalar<DataVector>& zeta_over_kappa =
      get<::Tags::TempScalar<2>>(temp_tensors);
  thermodynamic_derivatives_from_eos(
      make_not_null(&sound_speed_squared), make_not_null(&pressure),
      make_not_null(&kappa), make_not_null(&zeta),
      make_not_null(&zeta_over_kappa), rest_mass_density,
      specific_internal_energy, electron_fraction, specific_enthalpy,
      equation_of_state);
  // Where zeta = 0, we take zeta / kappa = 0, which also covers kappa = 0 with
  // a 1D equation of state.
  for (size_t point = 0; point < num_points; ++point) {
    if (get(zeta)[point] == 0.0) {
      get(zeta_over_kappa)[point] = 0.0;
    } else if (UNLIKELY(get(kappa)[point] == 0.0)) {
      ERROR(
          "The analytic hydro characteristics need a nonzero kappa = dp/deps "
          "where zeta = dp/dY_e is nonzero, but kappa = 0 and zeta = "
          << get(zeta)[point] << " at rest-mass density "
          << get(rest_mass_density)[point] << ", specific internal energy "
          << get(specific_internal_energy)[point] << ", and electron fraction "
          << get(electron_fraction)[point]
          << ". A table from ConvertComposeTable has kappa = 0 if the CompOSE "
             "table lacks kappa (Q11).");
    } else {
      get(zeta_over_kappa)[point] = get(zeta)[point] / get(kappa)[point];
    }
  }
  Scalar<DataVector>& sound_speed = get<::Tags::TempScalar<3>>(temp_tensors);
  get(sound_speed) = sqrt(get(sound_speed_squared));

  // Velocity and geometry
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<4>>(temp_tensors);
  dot_product(make_not_null(&normal_velocity), unit_normal, spatial_velocity);
  Scalar<DataVector>& one_minus_normal_velocity_squared =
      get<::Tags::TempScalar<5>>(temp_tensors);
  get(one_minus_normal_velocity_squared) = 1.0 - square(get(normal_velocity));
  tnsr::i<DataVector, 3>& spatial_velocity_one_form =
      get<::Tags::Tempi<0, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&spatial_velocity_one_form),
                       spatial_velocity, spatial_metric);
  Scalar<DataVector>& spatial_velocity_squared =
      get<::Tags::TempScalar<6>>(temp_tensors);
  dot_product(make_not_null(&spatial_velocity_squared), spatial_velocity,
              spatial_velocity_one_form);
  tnsr::I<DataVector, 3>& unit_normal_vector =
      get<::Tags::TempI<0, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&unit_normal_vector), unit_normal,
                       inv_spatial_metric);

  // Two unit one-forms orthogonal to each other and to the normal, their
  // raised forms, and the velocity along them
  Scalar<DataVector>& det_spatial_metric =
      get<::Tags::TempScalar<7>>(temp_tensors);
  determinant(make_not_null(&det_spatial_metric), spatial_metric);
  tnsr::i<DataVector, 3>& first_tangent =
      get<::Tags::Tempi<1, 3>>(temp_tensors);
  orthonormal_oneform(make_not_null(&first_tangent), unit_normal,
                      inv_spatial_metric);
  tnsr::i<DataVector, 3>& second_tangent =
      get<::Tags::Tempi<2, 3>>(temp_tensors);
  orthonormal_oneform(make_not_null(&second_tangent), unit_normal,
                      first_tangent, spatial_metric, det_spatial_metric);
  tnsr::I<DataVector, 3>& first_tangent_vector =
      get<::Tags::TempI<1, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&first_tangent_vector), first_tangent,
                       inv_spatial_metric);
  tnsr::I<DataVector, 3>& second_tangent_vector =
      get<::Tags::TempI<2, 3>>(temp_tensors);
  raise_or_lower_index(make_not_null(&second_tangent_vector), second_tangent,
                       inv_spatial_metric);
  Scalar<DataVector>& first_tangent_velocity =
      get<::Tags::TempScalar<8>>(temp_tensors);
  dot_product(make_not_null(&first_tangent_velocity), first_tangent,
              spatial_velocity);
  Scalar<DataVector>& second_tangent_velocity =
      get<::Tags::TempScalar<9>>(temp_tensors);
  dot_product(make_not_null(&second_tangent_velocity), second_tangent,
              spatial_velocity);

  // Characteristic speeds, computed as in characteristic_speeds_hydro
  // Note: d = W sqrt(1 - v^2 c_s^2 - v_n^2 (1 - c_s^2)) = W radical, where
  //       radical is computed first.
  Scalar<DataVector>& one_minus_v_squared_cs_squared =
      get<::Tags::TempScalar<10>>(temp_tensors);
  get(one_minus_v_squared_cs_squared) =
      1.0 - get(spatial_velocity_squared) * get(sound_speed_squared);
  Scalar<DataVector>& d = get<::Tags::TempScalar<11>>(temp_tensors);
  get(d) = sqrt(get(one_minus_v_squared_cs_squared) -
                get(normal_velocity) * get(normal_velocity) *
                    (1.0 - get(sound_speed_squared)));
  Scalar<DataVector>& first_term = get<::Tags::TempScalar<12>>(temp_tensors);
  get(first_term) = (1.0 - get(sound_speed_squared)) * get(normal_velocity) /
                    get(one_minus_v_squared_cs_squared);
  Scalar<DataVector>& second_term = get<::Tags::TempScalar<13>>(temp_tensors);
  get(second_term) =
      get(sound_speed) * get(d) /
      (get(lorentz_factor) * get(one_minus_v_squared_cs_squared));
  get(d) *= get(lorentz_factor);
  characteristic_speeds->get(0) = get(first_term) - get(second_term);
  for (size_t i = 1; i < 5; ++i) {
    characteristic_speeds->get(i) = get(normal_velocity);
  }
  characteristic_speeds->get(5) = get(first_term) + get(second_term);

  // h - 1, W - 1, and h W - 1, written to avoid cancellations when h and W are
  // close to 1
  Scalar<DataVector>& specific_enthalpy_minus_one =
      get<::Tags::TempScalar<12>>(temp_tensors);
  get(specific_enthalpy_minus_one) =
      get(specific_internal_energy) + get(pressure) / get(rest_mass_density);
  Scalar<DataVector>& lorentz_factor_minus_one =
      get<::Tags::TempScalar<13>>(temp_tensors);
  get(lorentz_factor_minus_one) = get(spatial_velocity_squared) *
                                  square(get(lorentz_factor)) /
                                  (get(lorentz_factor) + 1.0);
  Scalar<DataVector>& enthalpy_times_lorentz_factor_minus_one =
      get<::Tags::TempScalar<14>>(temp_tensors);
  get(enthalpy_times_lorentz_factor_minus_one) =
      get(specific_enthalpy_minus_one) * get(lorentz_factor) +
      get(lorentz_factor_minus_one);
  Scalar<DataVector>& factor = get<::Tags::TempScalar<15>>(temp_tensors);

  // The characteristic modes R_{in} (right eigenvectors), with the conserved
  // variables in the order (D, S_j, tau, D Y_e)
  auto& modes = *characteristic_modes;
  // R_- (i = 0) and R_+ (i = 5)
  get(factor) =
      get(specific_enthalpy) * get(lorentz_factor) * get(sound_speed) / get(d);
  const auto set_acoustic_mode = [&](const size_t i, const double sign) {
    modes.get(i, 0) = 1.0;
    for (size_t j = 0; j < 3; ++j) {
      modes.get(i, j + 1) = get(specific_enthalpy) * get(lorentz_factor) *
                                spatial_velocity_one_form.get(j) +
                            sign * get(factor) * unit_normal.get(j);
    }
    modes.get(i, 4) = get(enthalpy_times_lorentz_factor_minus_one) +
                      sign * get(factor) * get(normal_velocity);
    modes.get(i, 5) = get(electron_fraction);
  };
  set_acoustic_mode(0, -1.0);
  set_acoustic_mode(5, 1.0);
  // R_1 (i = 1) and R_2 (i = 2)
  const auto set_tangent_mode =
      [&](const size_t i, const tnsr::i<DataVector, 3>& tangent,
          const Scalar<DataVector>& tangent_velocity) {
        modes.get(i, 0) = get(lorentz_factor) * get(tangent_velocity);
        for (size_t j = 0; j < 3; ++j) {
          modes.get(i, j + 1) =
              get(specific_enthalpy) *
              (tangent.get(j) + 2.0 * square(get(lorentz_factor)) *
                                    get(tangent_velocity) *
                                    spatial_velocity_one_form.get(j));
        }
        modes.get(i, 4) =
            get(lorentz_factor) *
            (2.0 * get(specific_enthalpy) * get(lorentz_factor) - 1.0) *
            get(tangent_velocity);
        modes.get(i, 5) = get(electron_fraction) * modes.get(i, 0);
      };
  set_tangent_mode(1, first_tangent, first_tangent_velocity);
  set_tangent_mode(2, second_tangent, second_tangent_velocity);
  // R_3 (i = 3)
  // Note: the tau component h W (kappa - rho c_s^2) - kappa is written as
  //       kappa (h W - 1) - h W rho c_s^2, which avoids subtracting the nearly
  //       equal h W kappa and kappa when h W is close to 1.
  get(factor) = get(specific_enthalpy) * get(lorentz_factor) *
                get(rest_mass_density) * get(sound_speed_squared);
  modes.get(3, 0) = get(kappa);
  for (size_t j = 0; j < 3; ++j) {
    modes.get(3, j + 1) =
        (get(specific_enthalpy) * get(lorentz_factor) * get(kappa) -
         get(factor)) *
        spatial_velocity_one_form.get(j);
  }
  modes.get(3, 4) =
      get(kappa) * get(enthalpy_times_lorentz_factor_minus_one) - get(factor);
  modes.get(3, 5) = get(kappa) * get(electron_fraction);
  // R_4 (i = 4)
  get(factor) = get(zeta_over_kappa) * get(lorentz_factor);
  modes.get(4, 0) = 0.0;
  for (size_t j = 0; j < 3; ++j) {
    modes.get(4, j + 1) = -get(factor) * spatial_velocity_one_form.get(j);
  }
  modes.get(4, 4) = -get(factor);
  modes.get(4, 5) = 1.0;

  // The characteristic projectors L^{im} (left eigenvectors), with the
  // conserved variables in the same order
  auto& projectors = *characteristic_projectors;
  // L^- (i = 0) and L^+ (i = 5)
  // Note: with a = W^2 (1 - v_n^2) (kappa + rho c_s^2) and
  //       c_+- = rho c_s (c_s +- v_n d), the tau component is a - c_+-, and
  //       the D component a - c_+- - h W (kappa - rho c_s^2 + zeta Y_e / h)
  //       (1 - v_n^2) is written as
  //       W (1 - v_n^2) (rho c_s^2 (W + h) - kappa (h - W) - zeta Y_e) - c_+-,
  //       which avoids subtracting the nearly equal W^2 kappa and h W kappa
  //       when h and W are close to 1.
  Scalar<DataVector>& a = get<::Tags::TempScalar<16>>(temp_tensors);
  get(a) = square(get(lorentz_factor)) *
           get(one_minus_normal_velocity_squared) *
           (get(kappa) + get(rest_mass_density) * get(sound_speed_squared));
  Scalar<DataVector>& density_term = get<::Tags::TempScalar<17>>(temp_tensors);
  get(density_term) =
      get(lorentz_factor) * get(one_minus_normal_velocity_squared) *
      (get(rest_mass_density) * get(sound_speed_squared) *
           (get(lorentz_factor) + get(specific_enthalpy)) -
       get(kappa) *
           (get(specific_enthalpy_minus_one) - get(lorentz_factor_minus_one)) -
       get(zeta) * get(electron_fraction));
  get(factor) = 0.5 / (get(rest_mass_density) * get(specific_enthalpy) *
                       get(lorentz_factor) * get(sound_speed_squared) *
                       get(one_minus_normal_velocity_squared));
  const auto set_acoustic_projector = [&](const size_t i, const double sign) {
    // c_+-, stored in the D component
    projectors.get(i, 0) =
        get(rest_mass_density) * get(sound_speed) *
        (get(sound_speed) + sign * get(normal_velocity) * get(d));
    projectors.get(i, 4) = (get(a) - projectors.get(i, 0)) * get(factor);
    projectors.get(i, 0) =
        (get(density_term) - projectors.get(i, 0)) * get(factor);
    for (size_t j = 0; j < 3; ++j) {
      projectors.get(i, j + 1) =
          (-get(a) * spatial_velocity.get(j) +
           get(rest_mass_density) * get(sound_speed) *
               (get(sound_speed) * get(normal_velocity) + sign * get(d)) *
               unit_normal_vector.get(j)) *
          get(factor);
    }
    projectors.get(i, 5) = get(zeta) * get(lorentz_factor) *
                           get(one_minus_normal_velocity_squared) * get(factor);
  };
  set_acoustic_projector(0, -1.0);
  set_acoustic_projector(5, 1.0);
  // L^1 (i = 1) and L^2 (i = 2)
  get(factor) =
      1.0 / (get(specific_enthalpy) * get(one_minus_normal_velocity_squared));
  const auto set_tangent_projector =
      [&](const size_t i, const tnsr::I<DataVector, 3>& tangent_vector,
          const Scalar<DataVector>& tangent_velocity) {
        projectors.get(i, 0) = -get(tangent_velocity) * get(factor);
        for (size_t j = 0; j < 3; ++j) {
          projectors.get(i, j + 1) =
              (get(tangent_velocity) * get(normal_velocity) *
                   unit_normal_vector.get(j) +
               get(one_minus_normal_velocity_squared) * tangent_vector.get(j)) *
              get(factor);
        }
        projectors.get(i, 4) = projectors.get(i, 0);
        projectors.get(i, 5) = 0.0;
      };
  set_tangent_projector(1, first_tangent_vector, first_tangent_velocity);
  set_tangent_projector(2, second_tangent_vector, second_tangent_velocity);
  // L^3 (i = 3)
  // Note: h - W is written as (h - 1) - (W - 1).
  get(factor) = 1.0 / (get(rest_mass_density) * get(specific_enthalpy) *
                       get(sound_speed_squared));
  projectors.get(3, 0) =
      (get(specific_enthalpy_minus_one) - get(lorentz_factor_minus_one) +
       get(zeta_over_kappa) * get(electron_fraction)) *
      get(factor);
  for (size_t j = 0; j < 3; ++j) {
    projectors.get(3, j + 1) =
        get(lorentz_factor) * spatial_velocity.get(j) * get(factor);
  }
  projectors.get(3, 4) = -get(lorentz_factor) * get(factor);
  projectors.get(3, 5) = -get(zeta_over_kappa) * get(factor);
  // L^4 (i = 4)
  projectors.get(4, 0) = -get(electron_fraction);
  for (size_t j = 0; j < 4; ++j) {
    projectors.get(4, j + 1) = 0.0;
  }
  projectors.get(4, 5) = 1.0;
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
          equation_of_state);                                                  \
  template void analytic_characteristics<GET_DIM(data)>(                       \
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
