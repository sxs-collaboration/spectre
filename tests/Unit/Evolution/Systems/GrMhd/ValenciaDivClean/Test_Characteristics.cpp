// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <numbers>
#include <type_traits>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/GrMhd/ValenciaDivClean/Characteristics.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "Helpers/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "Informer/InfoFromBuild.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Tabulated3d.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

// This namespace is meant to hold straightforward implementations of the
// expressions used in the GRMHD characteristics that are not optimized for
// performance. We keep them here to compare against the optimized versions in
// Characteristics.cpp to ensure the optimized versions are correct.
namespace unoptimized {

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
  Variables<tmpl::list<
      hydro::Tags::SoundSpeedSquared<DataVector>,
      hydro::Tags::Pressure<DataVector>, ::Tags::TempScalar<0>,
      ::Tags::TempScalar<1>, ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
      ::Tags::TempScalar<4>, ::Tags::TempScalar<5>, ::Tags::TempScalar<6>,
      ::Tags::TempScalar<7>, ::Tags::TempScalar<8>, ::Tags::TempI<0, 3>,
      ::Tags::Tempi<0, 3>, ::Tags::TempIj<0, 3>, ::Tags::TempI<1, 3>>>
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
    Scalar<DataVector>& pressure =
        get<hydro::Tags::Pressure<DataVector>>(temp_tensors);
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
    get(kappa) = get(equation_of_state.kappa_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
    get(zeta) = get(equation_of_state.zeta_from_density_and_temperature(
        rest_mass_density, temperature, electron_fraction));
  }

  // Intermediate variables
  Scalar<DataVector>& Z = get<::Tags::TempScalar<3>>(temp_tensors);
  tenex::evaluate(make_not_null(&Z), rest_mass_density() * specific_enthalpy() *
                                         square(lorentz_factor()));
  Scalar<DataVector>& D = get<::Tags::TempScalar<4>>(temp_tensors);
  tenex::evaluate(make_not_null(&D), rest_mass_density() * lorentz_factor());
  Scalar<DataVector>& normal_velocity =
      get<::Tags::TempScalar<5>>(temp_tensors);
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
  tnsr::Ij<DataVector, 3>& mixed_spatial_metric =
      get<::Tags::TempIj<0, 3>>(temp_tensors);
  tenex::evaluate<ti::I, ti::j>(
      make_not_null(&mixed_spatial_metric),
      inv_spatial_metric(ti::I, ti::K) * spatial_metric(ti::k, ti::j));

  // Derivatives of Z
  Scalar<DataVector>& dzdD = get<::Tags::TempScalar<6>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdD),
      -((lorentz_factor() *
         (kappa() * (-specific_enthalpy() + lorentz_factor()) -
          zeta() * electron_fraction() +
          (sound_speed_squared() * specific_enthalpy() + lorentz_factor()) *
              rest_mass_density())) /
        ((-square(lorentz_factor()) +
          sound_speed_squared() * (-1. + square(lorentz_factor()))) *
         rest_mass_density())));
  tnsr::I<DataVector, 3>& dzds = get<::Tags::TempI<1, 3>>(temp_tensors);
  tenex::evaluate<ti::I>(
      make_not_null(&dzds),
      (spatial_velocity(ti::I) * square(lorentz_factor()) *
       (kappa() + sound_speed_squared() * rest_mass_density())) /
          ((-square(lorentz_factor()) +
            sound_speed_squared() * (-1. + square(lorentz_factor()))) *
           rest_mass_density()));
  Scalar<DataVector>& dzdtau = get<::Tags::TempScalar<7>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdtau),
      -((square(lorentz_factor()) * (kappa() + rest_mass_density())) /
        ((-square(lorentz_factor()) +
          sound_speed_squared() * (-1. + square(lorentz_factor()))) *
         rest_mass_density())));
  Scalar<DataVector>& dzdye = get<::Tags::TempScalar<8>>(temp_tensors);
  tenex::evaluate(
      make_not_null(&dzdye),
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

}  // namespace unoptimized

void test_characteristic_speeds(const DataVector& /*used_for_size*/) {
  //  Arbitrary random numbers can produce a negative radicand in Lambda^\pm.
  //  This bound helps to prevent that situation.
  // const double max_value = 1.0 / sqrt(3);
  // pypp::check_with_random_values<7>(
  //     &grmhd::ValenciaDivClean::characteristic_speeds_approximate_mhd<3>,
  //     "CharacteristicSpeeds", "CharacteristicSpeeds",
  //     {{{0.0, 1.0},
  //       {-1.0, 1.0},
  //       {-max_value, max_value},
  //       {0.0, 1.0},
  //       {0.0, 1.0},
  //       {0.0, 1.0},
  //       {-max_value, max_value}}},
  //     used_for_size);
}

void test_with_normal_along_coordinate_axes(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const EquationsOfState::PolytropicFluid<true> eos(0.001, 4.0 / 3.0);
  const auto specific_internal_energy =
      eos.specific_internal_energy_from_density(rest_mass_density);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy,
      eos.pressure_from_density(rest_mass_density));

  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);

  const auto lapse = gr_helper::random_lapse(nn_gen, used_for_size);
  const auto shift = gr_helper::random_shift<3>(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);

  const auto magnetic_field = helper::random_magnetic_field(
      nn_gen, eos.pressure_from_density(rest_mass_density), spatial_metric);
  const auto magnetic_field_squared =
      dot_product(magnetic_field, magnetic_field, spatial_metric);
  const auto magnetic_field_dot_spatial_velocity =
      dot_product(spatial_velocity, magnetic_field, spatial_metric);
  const DataVector comoving_magnetic_field_squared =
      get(magnetic_field_squared) / square(get(lorentz_factor)) +
      square(get(magnetic_field_dot_spatial_velocity));
  const Scalar<DataVector> alfven_speed_squared{
      comoving_magnetic_field_squared /
      (comoving_magnetic_field_squared +
       get(rest_mass_density) * get(specific_enthalpy))};
  const Scalar<DataVector> sound_speed_squared{
      (get(eos.chi_from_density(rest_mass_density)) +
       get(eos.kappa_times_p_over_rho_squared_from_density(
           rest_mass_density))) /
      get(specific_enthalpy)};

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    const auto& eos_base =
        static_cast<const EquationsOfState::EquationOfState<true, 1>&>(eos);
    const Approx custom_approx = Approx::custom().epsilon(1.0e-10);
    CHECK_ITERABLE_CUSTOM_APPROX(
        grmhd::ValenciaDivClean::characteristic_speeds_approximate_mhd(
            rest_mass_density, electron_fraction, specific_internal_energy,
            specific_enthalpy, spatial_velocity, lorentz_factor, magnetic_field,
            lapse, shift, spatial_metric, normal, eos_base),
        (pypp::call<std::array<DataVector, 9>>(
            "CharacteristicSpeeds", "CharacteristicSpeeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            alfven_speed_squared, normal)),
        custom_approx);
  }
}

void test_hydro_characteristic_speed(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);

  const EquationsOfState::IdealFluid<true> base_eos(4.0 / 3.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();

  const auto temperature = eos_3d->temperature_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);

  const auto sound_speed_squared =
      eos_3d->sound_speed_squared_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy,
      eos_3d->pressure_from_density_and_energy(
          rest_mass_density, specific_internal_energy, electron_fraction));

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto unit_normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    const Approx custom_approx = Approx::custom().epsilon(1.0e-10);

    CHECK_ITERABLE_CUSTOM_APPROX(
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            electron_fraction, lorentz_factor, specific_enthalpy,
            spatial_metric, unit_normal, *eos_3d),
        (pypp::call<tnsr::i<DataVector, 3>>(
            "CharacteristicSpeeds", "characteristic_speeds_hydro",
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            lorentz_factor, unit_normal)),
        custom_approx);
  }
}

// Load the tabulated equation of state used in the tests.
// Note: it is the only equation of state with a nonzero derivative of the
//       pressure with respect to the electron fraction.
EquationsOfState::Tabulated3D<true> tabulated_equation_of_state() {
  return EquationsOfState::Tabulated3D<true>{
      unit_test_src_path() +
          "PointwiseFunctions/Hydro/EquationsOfState/dd2_unit_test.h5",
      "/dd2"};
}

// Generate a random density, specific internal energy, and electron fraction
// within the bounds of a tabulated equation of state.
template <typename Generator>
std::array<Scalar<DataVector>, 3> random_tabulated_state(
    const gsl::not_null<Generator*> generator,
    const EquationsOfState::Tabulated3D<true>& equation_of_state,
    const DataVector& used_for_size) {
  // Stay slightly inside the bounds of the table.
  const auto random_between = [&generator, &used_for_size](
                                  const double lower_bound,
                                  const double upper_bound) {
    const double margin = 0.01 * (upper_bound - lower_bound);
    std::uniform_real_distribution<> distribution(lower_bound + margin,
                                                  upper_bound - margin);
    return make_with_random_values<Scalar<DataVector>>(
        generator, make_not_null(&distribution), used_for_size);
  };
  Scalar<DataVector> rest_mass_density =
      random_between(log(equation_of_state.rest_mass_density_lower_bound()),
                     log(equation_of_state.rest_mass_density_upper_bound()));
  get(rest_mass_density) = exp(get(rest_mass_density));
  const Scalar<DataVector> temperature =
      random_between(equation_of_state.temperature_lower_bound(),
                     equation_of_state.temperature_upper_bound());
  Scalar<DataVector> electron_fraction =
      random_between(equation_of_state.electron_fraction_lower_bound(),
                     equation_of_state.electron_fraction_upper_bound());
  Scalar<DataVector> specific_internal_energy =
      equation_of_state.specific_internal_energy_from_density_and_temperature(
          rest_mass_density, temperature, electron_fraction);
  return {std::move(rest_mass_density), std::move(specific_internal_energy),
          std::move(electron_fraction)};
}

// Check an eigensystem of the hydro characteristic matrix: the eigenvalues are
// lambda-, the normal velocity 4 times, and lambda+, in this order; each
// eigenvalue and its eigenvectors satisfy the left and right eigensystem
// relations; and the left eigenvectors are the inverse of the right ones.
void check_hydro_eigensystem(
    const tnsr::i<DataVector, 6>& eigenvalues,
    const tnsr::ij<DataVector, 6>& right_eigenvectors,
    const tnsr::IJ<DataVector, 6>& left_eigenvectors,
    const tnsr::iJ<DataVector, 6>& characteristic_matrix,
    const tnsr::i<DataVector, 3>& analytic_speeds,
    const Scalar<DataVector>& normal_velocity,
    const Scalar<DataVector>& lorentz_factor) {
  constexpr size_t matrix_size = 6;
  const size_t num_points = get(lorentz_factor).size();

  // Tolerance for all checks of the eigensystem below.
  // Note: this is the degeneracy tolerance of find_real_eigensystem.
  constexpr double tolerance = 1e-10;

  // Frobenius norm of the characteristic matrix, which sets the scale of the
  // errors in the eigensystem
  Scalar<DataVector> frobenius_norm{num_points, 0.0};
  for (size_t m = 0; m < matrix_size; ++m) {
    for (size_t n = 0; n < matrix_size; ++n) {
      get(frobenius_norm) += square(characteristic_matrix.get(m, n));
    }
  }
  get(frobenius_norm) = sqrt(get(frobenius_norm));

  // Check that the eigenvalues are lambda-, the normal velocity 4
  // times, and lambda+, in this order.
  // Note 1: find_real_eigensystem returns the eigenvalues in
  //         increasing order, and lambda- < normal velocity < lambda+
  //         for a nonzero sound speed.
  // Note 2: the eigenvalues of the characteristic matrix and the
  //         analytic speeds both lose accuracy for larger Lorentz
  //         factors, so the tolerance grows with the Lorentz factor.
  const DataVector& lambda_minus =
      analytic_speeds.get(grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus);
  const DataVector& lambda_plus =
      analytic_speeds.get(grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus);
  for (size_t point = 0; point < num_points; ++point) {
    INFO("Lorentz factor: " << get(lorentz_factor)[point]);
    const double eigenvalue_tolerance =
        tolerance * get(lorentz_factor)[point] * get(frobenius_norm)[point];
    CHECK(std::abs(eigenvalues.get(0)[point] - lambda_minus[point]) <
          eigenvalue_tolerance);
    for (size_t i = 1; i < 5; ++i) {
      CHECK(std::abs(eigenvalues.get(i)[point] - get(normal_velocity)[point]) <
            eigenvalue_tolerance);
    }
    CHECK(std::abs(eigenvalues.get(5)[point] - lambda_plus[point]) <
          eigenvalue_tolerance);
  }

  // Check eigensystem relation for each eigenvalue/eigenvector
  // Note: the residuals are relative to the Frobenius norm of the
  //       matrix and the norm of the eigenvector.
  for (size_t i = 0; i < matrix_size; ++i) {
    // Use non-owning DataVector views to avoid copying data
    const Scalar<DataVector> eigenvalue{};
    const tnsr::i<DataVector, matrix_size> right_eigenvector{};
    const tnsr::I<DataVector, matrix_size> left_eigenvector{};
    make_const_view(make_not_null(&get(eigenvalue)), eigenvalues.get(i), 0,
                    num_points);
    for (size_t k = 0; k < matrix_size; ++k) {
      make_const_view(make_not_null(&right_eigenvector.get(k)),
                      right_eigenvectors.get(i, k), 0, num_points);
      make_const_view(make_not_null(&left_eigenvector.get(k)),
                      left_eigenvectors.get(i, k), 0, num_points);
    }

    const Scalar<DataVector> right_eigensystem_error{
        get(magnitude(tenex::evaluate<ti::k>(
            characteristic_matrix(ti::k, ti::J) * right_eigenvector(ti::j) -
            eigenvalue() * right_eigenvector(ti::k)))) /
        (get(frobenius_norm) * get(magnitude(right_eigenvector)))};

    const Scalar<DataVector> left_eigensystem_error{
        get(magnitude(tenex::evaluate<ti::K>(
            left_eigenvector(ti::J) * characteristic_matrix(ti::j, ti::K) -
            eigenvalue() * left_eigenvector(ti::K)))) /
        (get(frobenius_norm) * get(magnitude(left_eigenvector)))};

    CHECK(max(get(right_eigensystem_error)) < tolerance);
    CHECK(max(get(left_eigensystem_error)) < tolerance);
  }

  // Check that the projectors are the inverse of the modes
  // (i.e., that the left and right eigenvectors are inverses of each
  // other)
  const auto projected_modes = tenex::evaluate<ti::I, ti::j>(
      left_eigenvectors(ti::I, ti::M) * right_eigenvectors(ti::j, ti::m));
  // Note: each entry is relative to the norms of its left and right
  //       eigenvectors.
  std::array<DataVector, matrix_size> left_eigenvector_norms{};
  std::array<DataVector, matrix_size> right_eigenvector_norms{};
  for (size_t i = 0; i < matrix_size; ++i) {
    gsl::at(left_eigenvector_norms, i) = DataVector(num_points, 0.0);
    gsl::at(right_eigenvector_norms, i) = DataVector(num_points, 0.0);
    for (size_t m = 0; m < matrix_size; ++m) {
      gsl::at(left_eigenvector_norms, i) += square(left_eigenvectors.get(i, m));
      gsl::at(right_eigenvector_norms, i) +=
          square(right_eigenvectors.get(i, m));
    }
    gsl::at(left_eigenvector_norms, i) =
        sqrt(gsl::at(left_eigenvector_norms, i));
    gsl::at(right_eigenvector_norms, i) =
        sqrt(gsl::at(right_eigenvector_norms, i));
  }
  double projector_error = 0.0;
  for (size_t i = 0; i < matrix_size; ++i) {
    for (size_t j = 0; j < matrix_size; ++j) {
      const double kronecker_delta = i == j ? 1.0 : 0.0;
      projector_error =
          std::max(projector_error,
                   max(abs(projected_modes.get(i, j) - kronecker_delta) /
                       (gsl::at(left_eigenvector_norms, i) *
                        gsl::at(right_eigenvector_norms, j))));
    }
  }
  CHECK(projector_error < tolerance);
}

void test_hydro_characteristics(const DataVector& used_for_size) {
  // Initialize number generator
  MAKE_GENERATOR(generator);
  const auto nn_gen = make_not_null(&generator);

  // Generate random quantities
  auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(nn_gen, used_for_size);
  auto lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(nn_gen, used_for_size);
  auto spatial_velocity = TestHelpers::hydro::random_velocity(
      nn_gen, lorentz_factor, spatial_metric);
  const auto random_rest_mass_density =
      TestHelpers::hydro::random_density(nn_gen, used_for_size);
  const auto random_specific_internal_energy =
      TestHelpers::hydro::random_specific_internal_energy(nn_gen,
                                                          used_for_size);
  const auto random_electron_fraction =
      TestHelpers::hydro::random_electron_fraction(nn_gen, used_for_size);

  // Define equations of state with each thermodynamic dimension
  const EquationsOfState::PolytropicFluid<true> equation_of_state_1d(100.0,
                                                                     2.0);
  const auto equation_of_state_2d =
      EquationsOfState::IdealFluid<true>(1.5, 0.0);
  const auto equation_of_state_3d = equation_of_state_2d.promote_to_3d_eos();
  const auto equation_of_state_tabulated = tabulated_equation_of_state();

  // Generate the states for each equation of state
  // Note 1: a 1D EoS determines the specific internal energy from the
  //         density.
  // Note 2: the tabulated EoS needs a state within its bounds.
  const auto specific_internal_energy_1d =
      equation_of_state_1d.specific_internal_energy_from_density(
          random_rest_mass_density);
  const auto [tabulated_rest_mass_density, tabulated_specific_internal_energy,
              tabulated_electron_fraction] =
      random_tabulated_state(nn_gen, equation_of_state_tabulated,
                             used_for_size);

  // Check the numerical and analytic characteristics with a given equation of
  // state
  const auto check_with_equation_of_state =
      [&](const auto& equation_of_state,
          const Scalar<DataVector>& rest_mass_density,
          const Scalar<DataVector>& specific_internal_energy,
          const Scalar<DataVector>& electron_fraction) {
        constexpr size_t thermodynamic_dim =
            std::decay_t<decltype(equation_of_state)>::thermodynamic_dim;

        // Compute derived quantities
        Scalar<DataVector> pressure{};
        if constexpr (thermodynamic_dim == 1) {
          pressure = equation_of_state.pressure_from_density(rest_mass_density);
        } else if constexpr (thermodynamic_dim == 2) {
          pressure = equation_of_state.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy);
        } else {
          pressure = equation_of_state.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy, electron_fraction);
        }
        const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure);
        const auto& inv_spatial_metric =
            determinant_and_inverse(spatial_metric).second;

        // Note: the containers for the eigenvalues and eigenvectors below are
        //       left unsized to check that numerical_characteristics and
        //       analytic_characteristics size them.
        constexpr size_t matrix_size = 6;
        const size_t num_points = used_for_size.size();

        // Loop over directions
        for (const auto& direction : Direction<3>::all_directions()) {
          // Get unit normal and normal velocity in this direction
          const auto unit_normal =
              unit_basis_form(direction, inv_spatial_metric);
          const auto normal_velocity =
              tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));

          // Get analytic characteristic speeds to check the eigenvalues
          const tnsr::i<DataVector, 3> analytic_speeds =
              grmhd::ValenciaDivClean::characteristic_speeds_hydro(
                  spatial_velocity, rest_mass_density, specific_internal_energy,
                  electron_fraction, lorentz_factor, specific_enthalpy,
                  spatial_metric, unit_normal, equation_of_state);

          // Get characteristic matrix
          tnsr::iJ<DataVector, matrix_size> characteristic_matrix{num_points};
          grmhd::ValenciaDivClean::flux_jacobian_hydro(
              make_not_null(&characteristic_matrix), spatial_velocity,
              rest_mass_density, specific_internal_energy, electron_fraction,
              lorentz_factor, specific_enthalpy, spatial_metric,
              inv_spatial_metric, unit_normal, equation_of_state);

          // Check the numerical characteristics
          {
            INFO("Numerical characteristics");
            tnsr::i<DataVector, matrix_size> eigenvalues{};
            tnsr::ij<DataVector, matrix_size> right_eigenvectors{};
            tnsr::IJ<DataVector, matrix_size> left_eigenvectors{};
            grmhd::ValenciaDivClean::numerical_characteristics(
                make_not_null(&eigenvalues), make_not_null(&right_eigenvectors),
                make_not_null(&left_eigenvectors), spatial_velocity,
                rest_mass_density, specific_internal_energy, electron_fraction,
                lorentz_factor, specific_enthalpy, spatial_metric,
                inv_spatial_metric, unit_normal, equation_of_state);
            check_hydro_eigensystem(eigenvalues, right_eigenvectors,
                                    left_eigenvectors, characteristic_matrix,
                                    analytic_speeds, normal_velocity,
                                    lorentz_factor);
          }

          // Check the analytic characteristics the same way
          {
            INFO("Analytic characteristics");
            tnsr::i<DataVector, matrix_size> eigenvalues{};
            tnsr::ij<DataVector, matrix_size> right_eigenvectors{};
            tnsr::IJ<DataVector, matrix_size> left_eigenvectors{};
            grmhd::ValenciaDivClean::analytic_characteristics(
                make_not_null(&eigenvalues), make_not_null(&right_eigenvectors),
                make_not_null(&left_eigenvectors), spatial_velocity,
                rest_mass_density, specific_internal_energy, electron_fraction,
                lorentz_factor, specific_enthalpy, spatial_metric,
                inv_spatial_metric, unit_normal, equation_of_state);
            check_hydro_eigensystem(eigenvalues, right_eigenvectors,
                                    left_eigenvectors, characteristic_matrix,
                                    analytic_speeds, normal_velocity,
                                    lorentz_factor);

            // The speeds are those of characteristic_speeds_hydro
            CHECK_ITERABLE_APPROX(
                eigenvalues.get(0),
                analytic_speeds.get(
                    grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus));
            for (size_t i = 1; i < 5; ++i) {
              CHECK_ITERABLE_APPROX(eigenvalues.get(i), get(normal_velocity));
            }
            CHECK_ITERABLE_APPROX(
                eigenvalues.get(5),
                analytic_speeds.get(
                    grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus));

            // The composition projector is (-Y_e, 0, 0, 0, 0, 1) for any state.
            // The composition mode has a zero D component and a unit D Y_e
            // component, and is pure D Y_e where zeta = 0, as for every
            // equation of state with fewer than three thermodynamic variables.
            const DataVector zero(num_points, 0.0);
            const DataVector one(num_points, 1.0);
            const DataVector minus_electron_fraction = -get(electron_fraction);
            CHECK_ITERABLE_APPROX(left_eigenvectors.get(4, 0),
                                  minus_electron_fraction);
            for (size_t m = 1; m < 5; ++m) {
              CHECK_ITERABLE_APPROX(left_eigenvectors.get(4, m), zero);
            }
            CHECK_ITERABLE_APPROX(left_eigenvectors.get(4, 5), one);
            CHECK_ITERABLE_APPROX(right_eigenvectors.get(4, 0), zero);
            CHECK_ITERABLE_APPROX(right_eigenvectors.get(4, 5), one);
            if constexpr (thermodynamic_dim < 3) {
              for (size_t n = 1; n < 5; ++n) {
                CHECK_ITERABLE_APPROX(right_eigenvectors.get(4, n), zero);
              }
            }
          }
        }
      };
  check_with_equation_of_state(equation_of_state_1d, random_rest_mass_density,
                               specific_internal_energy_1d,
                               random_electron_fraction);
  check_with_equation_of_state(equation_of_state_2d, random_rest_mass_density,
                               random_specific_internal_energy,
                               random_electron_fraction);
  check_with_equation_of_state(*equation_of_state_3d, random_rest_mass_density,
                               random_specific_internal_energy,
                               random_electron_fraction);
  check_with_equation_of_state(
      equation_of_state_tabulated, tabulated_rest_mass_density,
      tabulated_specific_internal_energy, tabulated_electron_fraction);

  // Test also for fluid at rest with flat metric.
  // Note: at rest, computing each eigenvector separately (e.g., with dgeev)
  //       can give nearly parallel eigenvectors for the degenerate eigenvalue.
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      spatial_metric.get(i, j) = i == j ? 1.0 : 0.0;
    }
    spatial_velocity.get(i) = 0.0;
  }
  get(lorentz_factor) = 1.0;
  check_with_equation_of_state(equation_of_state_1d, random_rest_mass_density,
                               specific_internal_energy_1d,
                               random_electron_fraction);
  check_with_equation_of_state(equation_of_state_2d, random_rest_mass_density,
                               random_specific_internal_energy,
                               random_electron_fraction);
  check_with_equation_of_state(*equation_of_state_3d, random_rest_mass_density,
                               random_specific_internal_energy,
                               random_electron_fraction);
  check_with_equation_of_state(
      equation_of_state_tabulated, tabulated_rest_mass_density,
      tabulated_specific_internal_energy, tabulated_electron_fraction);
}

void test_hydro_characteristics_match_unoptimized_version(
    const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  const auto random_rest_mass_density =
      helper::random_density(nn_gen, used_for_size);
  const auto random_specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto random_electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const EquationsOfState::IdealFluid<true> base_eos(1.5, 0.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();
  const auto eos_tabulated = tabulated_equation_of_state();
  const auto [tabulated_rest_mass_density, tabulated_specific_internal_energy,
              tabulated_electron_fraction] =
      random_tabulated_state(nn_gen, eos_tabulated, used_for_size);

  // Check that both versions agree with a given 3D equation of state
  const auto check_with_equation_of_state =
      [&](const EquationsOfState::EquationOfState<true, 3>& equation_of_state,
          const Scalar<DataVector>& rest_mass_density,
          const Scalar<DataVector>& specific_internal_energy,
          const Scalar<DataVector>& electron_fraction) {
        const auto pressure =
            equation_of_state.pressure_from_density_and_energy(
                rest_mass_density, specific_internal_energy, electron_fraction);
        const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
            rest_mass_density, specific_internal_energy, pressure);

        constexpr size_t matrix_size = 6;
        const size_t num_points = used_for_size.size();
        const Approx custom_approx =
            Approx::custom().epsilon(1.0e-12).scale(1.0);
        for (const auto& direction : Direction<3>::all_directions()) {
          const auto unit_normal =
              unit_basis_form(direction, inv_spatial_metric);

          tnsr::iJ<DataVector, matrix_size> optimized_matrix{num_points};
          tnsr::iJ<DataVector, matrix_size> unoptimized_matrix{num_points};
          grmhd::ValenciaDivClean::flux_jacobian_hydro<3>(
              make_not_null(&optimized_matrix), spatial_velocity,
              rest_mass_density, specific_internal_energy, electron_fraction,
              lorentz_factor, specific_enthalpy, spatial_metric,
              inv_spatial_metric, unit_normal, equation_of_state);
          unoptimized::flux_jacobian_hydro<3>(
              make_not_null(&unoptimized_matrix), spatial_velocity,
              rest_mass_density, specific_internal_energy, electron_fraction,
              lorentz_factor, specific_enthalpy, spatial_metric,
              inv_spatial_metric, unit_normal, equation_of_state);
          CHECK_ITERABLE_CUSTOM_APPROX(optimized_matrix, unoptimized_matrix,
                                       custom_approx);
        }
      };
  check_with_equation_of_state(*eos_3d, random_rest_mass_density,
                               random_specific_internal_energy,
                               random_electron_fraction);
  check_with_equation_of_state(eos_tabulated, tabulated_rest_mass_density,
                               tabulated_specific_internal_energy,
                               tabulated_electron_fraction);
}

// Set a flat spatial metric and its inverse, and a fluid at rest.
void set_fluid_at_rest_with_flat_metric(
    const gsl::not_null<tnsr::ii<DataVector, 3>*> spatial_metric,
    const gsl::not_null<tnsr::II<DataVector, 3>*> inv_spatial_metric,
    const gsl::not_null<tnsr::I<DataVector, 3>*> spatial_velocity,
    const gsl::not_null<Scalar<DataVector>*> lorentz_factor,
    const size_t num_points) {
  *spatial_metric = tnsr::ii<DataVector, 3>{num_points, 0.0};
  *inv_spatial_metric = tnsr::II<DataVector, 3>{num_points, 0.0};
  for (size_t i = 0; i < 3; ++i) {
    spatial_metric->get(i, i) = 1.0;
    inv_spatial_metric->get(i, i) = 1.0;
  }
  *spatial_velocity = tnsr::I<DataVector, 3>{num_points, 0.0};
  *lorentz_factor = Scalar<DataVector>{num_points, 1.0};
}

// For an ideal fluid at rest, the D components of the acoustic projectors
// vanish, since rho c_s^2 h = kappa (h - 1). Check that they do for cold
// fluids, where separate terms of order kappa, which is much larger than
// rho c_s^2, would lose accuracy.
void test_acoustic_projectors_for_cold_fluid(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  constexpr size_t matrix_size = 6;
  const size_t num_points = used_for_size.size();
  tnsr::ii<DataVector, 3> spatial_metric{};
  tnsr::II<DataVector, 3> inv_spatial_metric{};
  tnsr::I<DataVector, 3> spatial_velocity{};
  Scalar<DataVector> lorentz_factor{};
  set_fluid_at_rest_with_flat_metric(
      make_not_null(&spatial_metric), make_not_null(&inv_spatial_metric),
      make_not_null(&spatial_velocity), make_not_null(&lorentz_factor),
      num_points);
  const auto unit_normal =
      unit_basis_form(Direction<3>::upper_xi(), inv_spatial_metric);

  const EquationsOfState::IdealFluid<true> equation_of_state(4.0 / 3.0);
  const auto rest_mass_density = TestHelpers::hydro::random_density(
      make_not_null(&generator), used_for_size);
  const auto electron_fraction = TestHelpers::hydro::random_electron_fraction(
      make_not_null(&generator), used_for_size);
  for (const double specific_internal_energy_value :
       {1.0e-3, 1.0e-6, 1.0e-9, 1.0e-12}) {
    CAPTURE(specific_internal_energy_value);
    const Scalar<DataVector> specific_internal_energy{
        num_points, specific_internal_energy_value};
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy);
    const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
        rest_mass_density, specific_internal_energy, pressure);
    tnsr::i<DataVector, matrix_size> eigenvalues{};
    tnsr::ij<DataVector, matrix_size> right_eigenvectors{};
    tnsr::IJ<DataVector, matrix_size> left_eigenvectors{};
    grmhd::ValenciaDivClean::analytic_characteristics(
        make_not_null(&eigenvalues), make_not_null(&right_eigenvectors),
        make_not_null(&left_eigenvectors), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        equation_of_state);
    // Note: the round-off is about 5e-16, whereas subtracting terms of order
    //       kappa gives errors of about 1e-16 / eps.
    CHECK(max(abs(left_eigenvectors.get(0, 0))) < 1.0e-14);
    CHECK(max(abs(left_eigenvectors.get(5, 0))) < 1.0e-14);
  }
}

// A tabulated equation of state with kappa = 0 and zeta != 0, as from a
// CompOSE table without kappa. Its two points per dimension give eps = 0.01 T
// and p = 0.01 rho T, which interpolation in log T and log rho reproduces.
EquationsOfState::Tabulated3D<true>
tabulated_equation_of_state_without_kappa() {
  using Tabulated3D = EquationsOfState::Tabulated3D<true>;
  const std::vector<double> electron_fraction{0.1, 0.5};
  const std::vector<double> log_density{std::log(1.0e-6), std::log(1.0e-2)};
  // T = 0.1 and 10
  const std::vector<double> log_temperature{-std::numbers::ln10,
                                            std::numbers::ln10};
  // Note: the temperature varies fastest, and kappa and the fields not set
  //       below are zero.
  std::vector<double> table_data(8 * Tabulated3D::NumberOfVars, 0.0);
  for (size_t i_ye = 0; i_ye < 2; ++i_ye) {
    for (size_t i_rho = 0; i_rho < 2; ++i_rho) {
      for (size_t i_t = 0; i_t < 2; ++i_t) {
        const size_t offset =
            (i_t + 2 * (i_rho + 2 * i_ye)) * Tabulated3D::NumberOfVars;
        table_data[offset + Tabulated3D::Epsilon] =
            std::log(0.01) + log_temperature[i_t];
        table_data[offset + Tabulated3D::Pressure] =
            std::log(0.01) + log_density[i_rho] + log_temperature[i_t];
        table_data[offset + Tabulated3D::CsSquared] = 0.1;
        table_data[offset + Tabulated3D::Zeta] = 1.0e-3;
      }
    }
  }
  return Tabulated3D{electron_fraction, log_density, log_temperature,
                     table_data,        0.0,         1.0};
}

// Check the errors for states the hydro characteristics do not support.
void test_hydro_characteristics_errors(const DataVector& used_for_size) {
  constexpr size_t matrix_size = 6;
  const size_t num_points = used_for_size.size();
  tnsr::ii<DataVector, 3> spatial_metric{};
  tnsr::II<DataVector, 3> inv_spatial_metric{};
  tnsr::I<DataVector, 3> spatial_velocity{};
  Scalar<DataVector> lorentz_factor{};
  set_fluid_at_rest_with_flat_metric(
      make_not_null(&spatial_metric), make_not_null(&inv_spatial_metric),
      make_not_null(&spatial_velocity), make_not_null(&lorentz_factor),
      num_points);
  const auto unit_normal =
      unit_basis_form(Direction<3>::upper_xi(), inv_spatial_metric);
  const Scalar<DataVector> rest_mass_density{num_points, 1.0e-4};
  const Scalar<DataVector> electron_fraction{num_points, 0.3};
  tnsr::iJ<DataVector, matrix_size> characteristic_matrix{num_points};
  tnsr::i<DataVector, matrix_size> eigenvalues{};
  tnsr::ij<DataVector, matrix_size> right_eigenvectors{};
  tnsr::IJ<DataVector, matrix_size> left_eigenvectors{};

  // A 2D equation of state needs a positive pressure
  {
    const EquationsOfState::IdealFluid<true> equation_of_state(4.0 / 3.0);
    const Scalar<DataVector> specific_internal_energy{num_points, 0.0};
    const Scalar<DataVector> specific_enthalpy{num_points, 1.0};
    CHECK_THROWS_WITH(
        grmhd::ValenciaDivClean::flux_jacobian_hydro(
            make_not_null(&characteristic_matrix), spatial_velocity,
            rest_mass_density, specific_internal_energy, electron_fraction,
            lorentz_factor, specific_enthalpy, spatial_metric,
            inv_spatial_metric, unit_normal, equation_of_state),
        Catch::Matchers::ContainsSubstring(
            "with a 2D equation of state need a positive pressure"));
    CHECK_THROWS_WITH(
        grmhd::ValenciaDivClean::analytic_characteristics(
            make_not_null(&eigenvalues), make_not_null(&right_eigenvectors),
            make_not_null(&left_eigenvectors), spatial_velocity,
            rest_mass_density, specific_internal_energy, electron_fraction,
            lorentz_factor, specific_enthalpy, spatial_metric,
            inv_spatial_metric, unit_normal, equation_of_state),
        Catch::Matchers::ContainsSubstring(
            "with a 2D equation of state need a positive pressure"));
  }

  // The analytic characteristics need kappa != 0 where zeta != 0
  {
    const auto equation_of_state = tabulated_equation_of_state_without_kappa();
    const Scalar<DataVector> temperature{num_points, 1.0};
    const auto specific_internal_energy =
        equation_of_state.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction);
    const auto pressure = equation_of_state.pressure_from_density_and_energy(
        rest_mass_density, specific_internal_energy, electron_fraction);
    const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
        rest_mass_density, specific_internal_energy, pressure);
    CHECK_THROWS_WITH(
        grmhd::ValenciaDivClean::analytic_characteristics(
            make_not_null(&eigenvalues), make_not_null(&right_eigenvectors),
            make_not_null(&left_eigenvectors), spatial_velocity,
            rest_mass_density, specific_internal_energy, electron_fraction,
            lorentz_factor, specific_enthalpy, spatial_metric,
            inv_spatial_metric, unit_normal, equation_of_state),
        Catch::Matchers::ContainsSubstring(
            "need a nonzero kappa = dp/deps where zeta = dp/dY_e is nonzero"));
  }
}

void run_hydro_characteristic_benchmarks(const bool enable) {
  if (not enable) {
    return;
  }

  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  constexpr size_t num_points = 1000000;
  const DataVector used_for_size(num_points);

  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  const EquationsOfState::IdealFluid<true> base_eos(1.5, 0.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();

  const auto pressure = eos_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);

  const auto unit_normal =
      unit_basis_form(Direction<3>::lower_xi(), inv_spatial_metric);

  // Benchmark ran on mbot (Iago Mendes, April 2026):
  // - flux_jacobian_hydro (optimized): ~134 ms
  // - flux_jacobian_hydro (unoptimized): ~159 ms
  // - optimization result: ~1.2x speedup (~20% faster)
  constexpr size_t matrix_size = 6;
  tnsr::iJ<DataVector, matrix_size> optimized_matrix{num_points};
  tnsr::iJ<DataVector, matrix_size> unoptimized_matrix{num_points};
  BENCHMARK("flux_jacobian_hydro (optimized)") {
    grmhd::ValenciaDivClean::flux_jacobian_hydro<3>(
        make_not_null(&optimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
  };
  BENCHMARK("flux_jacobian_hydro (unoptimized)") {
    unoptimized::flux_jacobian_hydro<3>(
        make_not_null(&unoptimized_matrix), spatial_velocity, rest_mass_density,
        specific_internal_energy, electron_fraction, lorentz_factor,
        specific_enthalpy, spatial_metric, inv_spatial_metric, unit_normal,
        *eos_3d);
  };
}

}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Characteristics",
                  "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  const DataVector dv(5);
  test_characteristic_speeds(dv);
  // Test with aligned normals to check the code works
  // with vector components being 0.
  test_with_normal_along_coordinate_axes(dv);
  test_hydro_characteristic_speed(dv);
  test_hydro_characteristics(dv);
  test_hydro_characteristics_match_unoptimized_version(dv);
  test_acoustic_projectors_for_cold_fluid(dv);
  test_hydro_characteristics_errors(dv);

  // Disable benchmark by default
  run_hydro_characteristic_benchmarks(false);

  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeedsCompute>(
      "CharacteristicSpeeds");
}
