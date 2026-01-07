// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/OrthonormalOneform.hpp"
#include "DataStructures/Tensor/Expressions/AddSubtract.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
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
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/Equilibrium3D.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace {

void test_characteristic_speeds(const DataVector& /*used_for_size*/) {
  //  Arbitrary random numbers can produce a negative radicand in Lambda^\pm.
  //  This bound helps to prevent that situation.
  // const double max_value = 1.0 / sqrt(3);
  // pypp::check_with_random_values<7>(
  //     &grmhd::ValenciaDivClean::characteristic_speeds_approximate_mhd<3>,
  //     "TestFunctions", "characteristic_speeds",
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
  EquationsOfState::PolytropicFluid<true> eos(0.001, 4.0 / 3.0);
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
  Scalar<DataVector> alfven_speed_squared{
      comoving_magnetic_field_squared /
      (comoving_magnetic_field_squared +
       get(rest_mass_density) * get(specific_enthalpy))};
  Scalar<DataVector> sound_speed_squared{
      (get(eos.chi_from_density(rest_mass_density)) +
       get(eos.kappa_times_p_over_rho_squared_from_density(
           rest_mass_density))) /
      get(specific_enthalpy)};

  for (const auto& direction : Direction<3>::all_directions()) {
    const auto normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);

    const auto& eos_base =
        static_cast<const EquationsOfState::EquationOfState<true, 1>&>(eos);
    Approx custom_approx = Approx::custom().epsilon(1.0e-10);
    CHECK_ITERABLE_CUSTOM_APPROX(
        grmhd::ValenciaDivClean::characteristic_speeds_approximate_mhd(
            rest_mass_density, electron_fraction, specific_internal_energy,
            specific_enthalpy, spatial_velocity, lorentz_factor, magnetic_field,
            lapse, shift, spatial_metric, normal, eos_base),
        (pypp::call<std::array<DataVector, 9>>(
            "TestFunctions", "characteristic_speeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            alfven_speed_squared, normal)),
        custom_approx);
  }
}

void test_hydro_analytic_eigenvectors(const DataVector& used_for_size) {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);

  // Generate random quantities
  const auto spatial_metric =
      gr_helper::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      helper::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity =
      helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      helper::random_specific_internal_energy(nn_gen, used_for_size);
  const auto electron_fraction =
      helper::random_electron_fraction(nn_gen, used_for_size);

  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  // Define equation of state
  const EquationsOfState::IdealFluid<true> base_eos(1.5, 0.0);
  const auto eos_3d = base_eos.promote_to_3d_eos();

  // Compute derived quantities
  const auto pressure = eos_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);

  // Get kappa from the 2d EoS and set zeta = 0 for now
  const Scalar<DataVector> kappa_times_p_over_rho_squared =
      base_eos.kappa_times_p_over_rho_squared_from_density_and_energy(
          rest_mass_density, specific_internal_energy);

  Scalar<DataVector> kappa{};
  get(kappa) = get(kappa_times_p_over_rho_squared) / get(pressure) *
               square(get(rest_mass_density));

  const Scalar<DataVector> zeta{DataVector(used_for_size.size(), 0.0)};

  // Loop over directions
  for (const auto& direction : Direction<3>::all_directions()) {
    // Get unit normal in this direction
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);

    // Analytic eigenvectors (RIGHT + LEFT)
    constexpr size_t matrix_size = 6;
    std::array<tnsr::i<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        right_eigenvectors{};
    std::array<tnsr::I<DataVector, matrix_size, Frame::Inertial>, matrix_size>
        left_eigenvectors{};
    grmhd::ValenciaDivClean::eigenvectors_hydro<3>(
        make_not_null(&right_eigenvectors), make_not_null(&left_eigenvectors),
        spatial_velocity, rest_mass_density, specific_internal_energy,
        specific_enthalpy, electron_fraction, lorentz_factor, kappa, zeta,
        unit_normal, spatial_metric, *eos_3d);

    // Analytic characteristic speeds
    const std::array<DataVector, 3> analytic_speeds =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *eos_3d);

    // Assemble eigenvalues in eigenvector ordering: (degenerate x4, +, -)
    std::array<Scalar<DataVector>, matrix_size> all_eigenvalues;
    const size_t num_points = used_for_size.size();
    for (size_t i = 0; i < matrix_size; ++i) {
      get(gsl::at(all_eigenvalues, i)).destructive_resize(num_points);
    }
    for (size_t i = 0; i < 4; ++i) {
      get(gsl::at(all_eigenvalues, i)) = analytic_speeds
          [grmhd::ValenciaDivClean::HydroSpeed::NormalDotVelocity];
    }
    get(gsl::at(all_eigenvalues, 4)) =
        analytic_speeds[grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus];
    get(gsl::at(all_eigenvalues, 5)) =
        analytic_speeds[grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus];

    // Get characteristic matrix to check eigensystem relations
    tnsr::iJ<DataVector, 6> characteristic_matrix =
        make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
    grmhd::ValenciaDivClean::detail::flux_jacobian_hydro(
        make_not_null(&characteristic_matrix), spatial_velocity,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *eos_3d);

    constexpr double tolerance = 1e-10;

    // Check scaled eigensystem relation for each eigenvalue/eigenvector
    for (size_t i = 0; i < matrix_size; ++i) {
      const Scalar<DataVector>& eigenvalue = gsl::at(all_eigenvalues, i);

      const tnsr::i<DataVector, 6>& right_eigenvector =
          gsl::at(right_eigenvectors, i);
      const Scalar<DataVector> right_residual =
          magnitude(tenex::evaluate<ti::i>(
              characteristic_matrix(ti::i, ti::J) * right_eigenvector(ti::j) -
              eigenvalue() * right_eigenvector(ti::i)));
      const Scalar<DataVector> right_norm = magnitude(right_eigenvector);

      const tnsr::I<DataVector, 6>& left_eigenvector =
          gsl::at(left_eigenvectors, i);
      const Scalar<DataVector> left_residual = magnitude(tenex::evaluate<ti::I>(
          left_eigenvector(ti::J) * characteristic_matrix(ti::j, ti::I) -
          eigenvalue() * left_eigenvector(ti::I)));
      const Scalar<DataVector> left_norm = magnitude(left_eigenvector);

      double max_scaled_error = 0.0;
      for (size_t point = 0; point < used_for_size.size(); ++point) {
        const double r_scale = std::max(1.0, std::abs(get(right_norm)[point]));
        const double l_scale = std::max(1.0, std::abs(get(left_norm)[point]));
        max_scaled_error = std::max(
            max_scaled_error, std::abs(get(right_residual)[point]) / r_scale);
        max_scaled_error = std::max(
            max_scaled_error, std::abs(get(left_residual)[point]) / l_scale);
      }

      CHECK(max_scaled_error < tolerance);
    }

    // Orthonormality check
    for (size_t i = 0; i < matrix_size; ++i) {
      const tnsr::I<DataVector, 6>& left_eigenvector =
          gsl::at(left_eigenvectors, i);
      const Scalar<DataVector> left_norm = magnitude(left_eigenvector);

      for (size_t j = 0; j < matrix_size; ++j) {
        const tnsr::i<DataVector, 6>& right_eigenvector =
            gsl::at(right_eigenvectors, j);
        const Scalar<DataVector> right_norm = magnitude(right_eigenvector);

        const Scalar<DataVector> dot_ij =
            tenex::evaluate(left_eigenvector(ti::J) * right_eigenvector(ti::j));

        const double target = (i == j ? 1.0 : 0.0);

        double max_scaled_error = 0.0;
        for (size_t point = 0; point < used_for_size.size(); ++point) {
          const double err = std::abs(get(dot_ij)[point] - target);
          const double scale =
              std::max(1.0, std::abs(get(left_norm)[point]) *
                                std::abs(get(right_norm)[point]));
          max_scaled_error = std::max(max_scaled_error, err / scale);
        }

        CHECK(max_scaled_error < tolerance);
      }
    }
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
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *eos_3d),
        (pypp::call<std::array<DataVector, 3>>(
            "TestFunctions", "characteristic_speeds_hydro", spatial_velocity,
            spatial_velocity_squared, sound_speed_squared, lorentz_factor,
            unit_normal)),
        custom_approx);
  }
}

void test_hydro_numerical_eigensystem(const DataVector& used_for_size) {
  // Initialize number generator
  MAKE_GENERATOR(generator);
  const auto nn_gen = make_not_null(&generator);

  // Generate random quantities
  const auto spatial_metric =
      TestHelpers::gr::random_spatial_metric<3>(nn_gen, used_for_size);
  const auto lorentz_factor =
      TestHelpers::hydro::random_lorentz_factor(nn_gen, used_for_size);
  const auto spatial_velocity = TestHelpers::hydro::random_velocity(
      nn_gen, lorentz_factor, spatial_metric);
  const auto rest_mass_density =
      TestHelpers::hydro::random_density(nn_gen, used_for_size);
  const auto specific_internal_energy =
      TestHelpers::hydro::random_specific_internal_energy(nn_gen,
                                                          used_for_size);
  const auto electron_fraction =
      TestHelpers::hydro::random_electron_fraction(nn_gen, used_for_size);

  // Define equation of state
  const auto equation_of_state_2d =
      EquationsOfState::IdealFluid<true>(1.5, 0.0);
  const auto equation_of_state_3d = equation_of_state_2d.promote_to_3d_eos();

  // Compute derived quantities
  const auto pressure = equation_of_state_3d->pressure_from_density_and_energy(
      rest_mass_density, specific_internal_energy, electron_fraction);
  const auto specific_enthalpy = hydro::relativistic_specific_enthalpy(
      rest_mass_density, specific_internal_energy, pressure);
  const auto& inv_spatial_metric =
      determinant_and_inverse(spatial_metric).second;

  // Loop over directions
  for (const auto& direction : Direction<3>::all_directions()) {
    // Get unit normal and normal velocity in this direction
    const auto unit_normal = unit_basis_form(direction, inv_spatial_metric);
    const auto normal_velocity =
        tenex::evaluate(spatial_velocity(ti::I) * unit_normal(ti::i));

    // Initialize containers for all eigenvalues and eigenvectors
    constexpr size_t matrix_size = 6;
    std::array<Scalar<DataVector>, matrix_size> all_eigenvalues;
    std::array<tnsr::i<DataVector, matrix_size>, matrix_size>
        all_right_eigenvectors;
    std::array<tnsr::I<DataVector, matrix_size>, matrix_size>
        all_left_eigenvectors;
    const size_t num_points = used_for_size.size();
    for (size_t i = 0; i < matrix_size; ++i) {
      get(gsl::at(all_eigenvalues, i)).destructive_resize(num_points);
      for (size_t k = 0; k < matrix_size; ++k) {
        gsl::at(all_right_eigenvectors, i)
            .get(k)
            .destructive_resize(num_points);
        gsl::at(all_left_eigenvectors, i).get(k).destructive_resize(num_points);
      }
    }

    // Solve numerical eigensystem
    grmhd::ValenciaDivClean::numerical_eigensystem(
        make_not_null(&all_eigenvalues), make_not_null(&all_right_eigenvectors),
        make_not_null(&all_left_eigenvectors), spatial_velocity,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *equation_of_state_3d);

    // Get analytic characteristic speeds to check the numeric eigenvalues
    const std::array<DataVector, 3> analytic_speeds =
        grmhd::ValenciaDivClean::characteristic_speeds_hydro<3>(
            spatial_velocity, rest_mass_density, specific_internal_energy,
            specific_enthalpy, electron_fraction, lorentz_factor, unit_normal,
            spatial_metric, *equation_of_state_3d);

    // Count degenerate eigenvalues and check the other speeds
    // Note 1: We expect 4 degenerate eigenvalues equal to the normal velocity.
    // Note 2: The characteristic matrix becomes more defective for larger
    //         Lorentz boosts. With the default random Lorentz factor generator,
    //         the largest Lorentz factor is ~20, which leads to an eigenvalue
    //         error of ~1e-10.
    constexpr double eigenvalue_tolerance = 1e-9;
    for (size_t point = 0; point < num_points; ++point) {
      int number_of_degenerate_eigenvalues = 0;
      bool found_lambda_plus = false;
      bool found_lambda_minus = false;
      for (size_t i = 0; i < 6; ++i) {
        const Scalar<DataVector>& eigenvalue = gsl::at(all_eigenvalues, i);
        const double diff_with_normal_velocity =
            std::abs(get(eigenvalue)[point] - get(normal_velocity)[point]);
        const double diff_with_lambda_plus = std::abs(
            get(eigenvalue)[point] -
            analytic_speeds[grmhd::ValenciaDivClean::HydroSpeed::LambdaPlus]
                           [point]);
        const double diff_with_lambda_minus = std::abs(
            get(eigenvalue)[point] -
            analytic_speeds[grmhd::ValenciaDivClean::HydroSpeed::LambdaMinus]
                           [point]);
        if (diff_with_normal_velocity < eigenvalue_tolerance) {
          number_of_degenerate_eigenvalues += 1;
        } else if (diff_with_lambda_plus < eigenvalue_tolerance) {
          CHECK_FALSE(found_lambda_plus);
          found_lambda_plus = true;
        } else if (diff_with_lambda_minus < eigenvalue_tolerance) {
          CHECK_FALSE(found_lambda_minus);
          found_lambda_minus = true;
        } else {
          FAIL(
              "Found an eigenvalue that does not match any expected "
              "characteristic speed.\n"
              "Lorentz factor: "
              << get(lorentz_factor)[point]
              << "\n"
                 "Differences with analytic eigenvalues: "
              << diff_with_normal_velocity << ", " << diff_with_lambda_plus
              << ", " << diff_with_lambda_minus);
        }
      }
      CHECK(number_of_degenerate_eigenvalues == 4);
      CHECK(found_lambda_plus);
      CHECK(found_lambda_minus);
    }

    // Get characteristic matrix to check eigensystem relations
    tnsr::iJ<DataVector, 6> characteristic_matrix =
        make_with_value<tnsr::iJ<DataVector, 6>>(spatial_metric, 0.0);
    grmhd::ValenciaDivClean::detail::flux_jacobian_hydro(
        make_not_null(&characteristic_matrix), spatial_velocity,
        rest_mass_density, specific_internal_energy, electron_fraction,
        lorentz_factor, specific_enthalpy, spatial_metric, inv_spatial_metric,
        unit_normal, *equation_of_state_3d);

    // Check eigensystem relation for each eigenvalue/eigenvector
    constexpr double numeric_tolerance = 1e-12;
    for (size_t i = 0; i < 6; ++i) {
      const Scalar<DataVector>& eigenvalue = gsl::at(all_eigenvalues, i);

      const tnsr::i<DataVector, 6>& right_eigenvector =
          gsl::at(all_right_eigenvectors, i);
      const Scalar<DataVector> right_eigensystem_error =
          magnitude(tenex::evaluate<ti::i>(
              characteristic_matrix(ti::i, ti::J) * right_eigenvector(ti::j) -
              eigenvalue() * right_eigenvector(ti::i)));

      const tnsr::I<DataVector, 6>& left_eigenvector =
          gsl::at(all_left_eigenvectors, i);
      const Scalar<DataVector> left_eigensystem_error =
          magnitude(tenex::evaluate<ti::I>(
              left_eigenvector(ti::J) * characteristic_matrix(ti::j, ti::I) -
              eigenvalue() * left_eigenvector(ti::I)));

      double eigensystem_error = 0.0;
      for (size_t point = 0; point < used_for_size.size(); ++point) {
        eigensystem_error = std::max(
            eigensystem_error, std::abs(get(right_eigensystem_error)[point]));
        eigensystem_error = std::max(
            eigensystem_error, std::abs(get(left_eigensystem_error)[point]));
      }
      CHECK(eigensystem_error < numeric_tolerance);
    }
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.GrMhd.ValenciaDivClean.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/GrMhd/ValenciaDivClean"};

  const DataVector dv(5);
  test_characteristic_speeds(dv);
  // Test with aligned normals to check the code works
  // with vector components being 0.
  test_with_normal_along_coordinate_axes(dv);
  test_hydro_characteristic_speed(dv);
  test_hydro_numerical_eigensystem(dv);
  test_hydro_analytic_eigenvectors(dv);

  TestHelpers::db::test_compute_tag<
      grmhd::ValenciaDivClean::Tags::CharacteristicSpeedsCompute>(
      "CharacteristicSpeeds");
}
