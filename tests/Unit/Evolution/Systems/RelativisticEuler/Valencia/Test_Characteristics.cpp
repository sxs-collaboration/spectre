// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Direction.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Characteristics.hpp"
#include "Evolution/Systems/RelativisticEuler/Valencia/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/PolytropicFluid.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "PointwiseFunctions/Hydro/SpecificEnthalpy.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"
#include "tests/Unit/Domain/DomainTestHelpers.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "tests/Unit/PointwiseFunctions/Hydro/TestHelpers.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/Pypp.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"
#include "tests/Utilities/RandomUnitNormal.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
void test_compute_item_in_databox(
    const Scalar<DataVector>& lapse, const tnsr::I<DataVector, Dim>& shift,
    const tnsr::ii<DataVector, Dim>& spatial_metric,
    const tnsr::I<DataVector, Dim>& spatial_velocity,
    const Scalar<DataVector>& spatial_velocity_squared,
    const Scalar<DataVector>& sound_speed_squared,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  TestHelpers::db::test_compute_tag<
      RelativisticEuler::Valencia::Tags::CharacteristicSpeedsCompute<Dim>>(
      "CharacteristicSpeeds");
  const auto box = db::create<
      db::AddSimpleTags<
          gr::Tags::Lapse<>, gr::Tags::Shift<Dim>, gr::Tags::SpatialMetric<Dim>,
          hydro::Tags::SpatialVelocity<DataVector, Dim>,
          hydro::Tags::SoundSpeedSquared<DataVector>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      db::AddComputeTags<
          RelativisticEuler::Valencia::Tags::CharacteristicSpeedsCompute<Dim>>>(
      lapse, shift, spatial_metric, spatial_velocity, sound_speed_squared,
      normal);
  CHECK(RelativisticEuler::Valencia::characteristic_speeds(
            lapse, shift, spatial_velocity, spatial_velocity_squared,
            sound_speed_squared, normal) ==
        db::get<RelativisticEuler::Valencia::Tags::CharacteristicSpeeds<Dim>>(
            box));
}

template <size_t Dim>
void test_characteristic_speeds(const DataVector& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  namespace helper = TestHelpers::hydro;
  namespace gr_helper = TestHelpers::gr;
  const auto nn_gen = make_not_null(&generator);
  const auto lapse = gr_helper::random_lapse(nn_gen, used_for_size);
  const auto shift = gr_helper::random_shift<Dim>(nn_gen, used_for_size);
  const auto spatial_metric =
      gr_helper::random_spatial_metric<Dim>(nn_gen, used_for_size);
  const auto spatial_velocity = helper::random_velocity(
      nn_gen, helper::random_lorentz_factor(nn_gen, used_for_size),
      spatial_metric);
  const auto spatial_velocity_squared =
      dot_product(spatial_velocity, spatial_velocity, spatial_metric);
  EquationsOfState::PolytropicFluid<true> eos(0.001, 4.0 / 3.0);
  const auto rest_mass_density = helper::random_density(nn_gen, used_for_size);
  const Scalar<DataVector> sound_speed_squared{
      (get(eos.chi_from_density(rest_mass_density)) +
       get(eos.kappa_times_p_over_rho_squared_from_density(
           rest_mass_density))) /
      get(eos.specific_enthalpy_from_density(rest_mass_density))};

  // test with normal along coordinate axes
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto normal = unit_basis_form(
        direction, determinant_and_inverse(spatial_metric).second);
    CHECK_ITERABLE_APPROX(
        RelativisticEuler::Valencia::characteristic_speeds(
            lapse, shift, spatial_velocity, spatial_velocity_squared,
            sound_speed_squared, normal),
        (pypp::call<std::array<DataVector, Dim + 2>>(
            "TestFunctions", "characteristic_speeds", lapse, shift,
            spatial_velocity, spatial_velocity_squared, sound_speed_squared,
            normal)));
  }

  // test with random normal
  const auto random_normal = raise_or_lower_index(
      random_unit_normal(nn_gen, spatial_metric), spatial_metric);
  CHECK_ITERABLE_APPROX(
      RelativisticEuler::Valencia::characteristic_speeds(
          lapse, shift, spatial_velocity, spatial_velocity_squared,
          sound_speed_squared, random_normal),
      (pypp::call<std::array<DataVector, Dim + 2>>(
          "TestFunctions", "characteristic_speeds", lapse, shift,
          spatial_velocity, spatial_velocity_squared, sound_speed_squared,
          random_normal)));
  test_compute_item_in_databox(lapse, shift, spatial_metric, spatial_velocity,
                               spatial_velocity_squared, sound_speed_squared,
                               random_normal);
}

template <size_t Dim>
Matrix matrix_of_eigenvalues(const tnsr::I<double, Dim>& spatial_velocity,
                             const Scalar<double>& spatial_velocity_squared,
                             const Scalar<double>& sound_speed_squared,
                             const tnsr::i<double, Dim>& normal) noexcept {
  const double v_n = get(dot_product(spatial_velocity, normal));
  const double v_squared = get(spatial_velocity_squared);
  const double c_squared = get(sound_speed_squared);
  const double prefactor = 1.0 / (1.0 - v_squared * c_squared);
  const double first_term = prefactor * (1.0 - c_squared);
  const double second_term =
      prefactor *
      sqrt(c_squared * (1.0 - v_squared) *
           (1.0 - v_squared * c_squared - square(v_n) * (1.0 - c_squared)));

  Matrix result(Dim + 2, Dim + 2, 0.0);
  for (size_t i = 0; i < Dim + 2; ++i) {
    result(i, i) = v_n;
  }
  result(0, 0) = result(0, 0) * first_term - second_term;
  result(Dim + 1, Dim + 1) =
      result(Dim + 1, Dim + 1) * first_term + second_term;
  return result;
}

template <size_t Dim>
Matrix expected_flux_jacobian(
    const tnsr::I<double, Dim>& spatial_velocity,
    const tnsr::i<double, Dim>& spatial_velocity_oneform,
    const Scalar<double>& lorentz_factor,
    const Scalar<double>& specific_enthalpy,
    const Scalar<double>& sound_speed_squared,
    const Scalar<double>& kappa_over_density,
    const tnsr::i<double, Dim>& normal,
    const tnsr::I<double, Dim>& normal_vector) noexcept {
  const double w = get(lorentz_factor);
  const double w_squared = square(w);
  const double v_n = get(dot_product(spatial_velocity, normal));
  const double h = get(specific_enthalpy);
  const double c_squared = get(sound_speed_squared);
  const double k = get(kappa_over_density);
  const double f = w_squared + c_squared * (1.0 - w_squared);
  const double prefactor = 1.0 / (h * f);

  Matrix jacobian(Dim + 2, Dim + 2, 0.0);
  // dF[D]/dUcons as a row vector
  jacobian(0, 0) =
      prefactor * (h * w_squared * (1.0 - c_squared) + k * (h - w) - w) * v_n;
  jacobian(0, 1) = -prefactor * w * (k + 1.0) * v_n;
  for (size_t j = 0; j < Dim; ++j) {
    jacobian(0, j + 2) =
        prefactor * (f * normal_vector.get(j) / w +
                     w * (k + c_squared) * v_n * spatial_velocity.get(j));
  }
  // dF[tau]/dUcons as a row vector
  jacobian(1, 0) = -jacobian(0, 0);
  jacobian(1, 1) = -jacobian(0, 1);
  for (size_t j = 0; j < Dim; ++j) {
    jacobian(1, j + 2) =
        prefactor * ((h - 1.0 / w) * f * normal_vector.get(j) -
                     w * (k + c_squared) * v_n * spatial_velocity.get(j));
  }
  // dF[S_i]/dUcons as a row vector (one for each component of S_i)
  const double inv_f = 1.0 / f;
  for (size_t i = 0; i < Dim; ++i) {
    jacobian(i + 2, 0) =
        inv_f * (((h * w - 1.0 + w_squared) * c_squared - k * w * (h - w)) *
                     normal.get(i) +
                 w * (h * (k - c_squared) - w * (k + 1.0)) * v_n *
                     spatial_velocity_oneform.get(i));
    jacobian(i + 2, 1) =
        inv_f *
        (((w_squared - 1.0) * c_squared + k * w_squared) * normal.get(i) -
         (k + 1.0) * w_squared * v_n * spatial_velocity_oneform.get(i));
    for (size_t j = 0; j < Dim; ++j) {
      jacobian(i + 2, j + 2) =
          spatial_velocity_oneform.get(i) * normal_vector.get(j) +
          inv_f * w_squared * (k + c_squared) *
              (v_n * spatial_velocity_oneform.get(i) - normal.get(i)) *
              spatial_velocity.get(j);
    }
    jacobian(i + 2, i + 2) += v_n;
  }
  return jacobian;
}

template <size_t Dim>
void test_characteristic_matrices(const double used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  const auto test_for_given_eos =
      [&used_for_size, &generator ](const auto& equation_of_state) noexcept {
    namespace helper = TestHelpers::hydro;
    namespace gr_helper = TestHelpers::gr;
    const auto nn_gen = make_not_null(&generator);

    const auto spatial_metric =
        gr_helper::random_spatial_metric<Dim>(nn_gen, used_for_size);
    const auto det_and_inv_metric = determinant_and_inverse(spatial_metric);
    const auto lorentz_factor =
        helper::random_lorentz_factor(nn_gen, used_for_size);
    const auto spatial_velocity =
        helper::random_velocity(nn_gen, lorentz_factor, spatial_metric);
    const auto spatial_velocity_oneform =
        raise_or_lower_index(spatial_velocity, spatial_metric);
    const auto spatial_velocity_squared =
        dot_product(spatial_velocity, spatial_velocity, spatial_metric);

    const auto random_normal_vector =
        random_unit_normal(nn_gen, spatial_metric);
    const auto random_normal =
        raise_or_lower_index(random_normal_vector, spatial_metric);

    // Make consistent set of thermodynamic variables, along with their
    // derived quantities for a given equation of state.
    const auto rest_mass_density =
        helper::random_density(nn_gen, used_for_size);
    Scalar<double> specific_internal_energy{};
    Scalar<double> pressure{};
    Scalar<double> specific_enthalpy{};
    Scalar<double> sound_speed_squared{};
    Scalar<double> kappa_over_density{};
    make_overloader(
        [
          &rest_mass_density, &specific_internal_energy, &pressure,
          &specific_enthalpy, &sound_speed_squared, &kappa_over_density
        ](const EquationsOfState::EquationOfState<true, 1>& eos) noexcept {
          specific_internal_energy =
              eos.specific_internal_energy_from_density(rest_mass_density);
          pressure = eos.pressure_from_density(rest_mass_density);
          specific_enthalpy = hydro::relativistic_specific_enthalpy(
              rest_mass_density, specific_internal_energy, pressure);
          sound_speed_squared = hydro::sound_speed_squared(
              rest_mass_density, specific_internal_energy, specific_enthalpy,
              eos);
          kappa_over_density = Scalar<double>{
              {{get(eos.kappa_times_p_over_rho_squared_from_density(
                    rest_mass_density)) *
                get(rest_mass_density) / get(pressure)}}};
        },
        [
          &nn_gen, &used_for_size, &rest_mass_density,
          &specific_internal_energy, &pressure, &specific_enthalpy,
          &sound_speed_squared, &kappa_over_density
        ](const EquationsOfState::EquationOfState<true, 2>& eos) noexcept {
          specific_internal_energy =
              helper::random_specific_internal_energy(nn_gen, used_for_size);
          pressure = eos.pressure_from_density_and_energy(
              rest_mass_density, specific_internal_energy);
          specific_enthalpy = hydro::relativistic_specific_enthalpy(
              rest_mass_density, specific_internal_energy, pressure);
          sound_speed_squared = hydro::sound_speed_squared(
              rest_mass_density, specific_internal_energy, specific_enthalpy,
              eos);
          kappa_over_density = Scalar<double>{
              {{get(eos.kappa_times_p_over_rho_squared_from_density_and_energy(
                    rest_mass_density, specific_internal_energy)) *
                get(rest_mass_density) / get(pressure)}}};
        })(equation_of_state);

    Matrix right_matrix = RelativisticEuler::Valencia::right_eigenvectors(
        rest_mass_density, spatial_velocity, specific_internal_energy, pressure,
        specific_enthalpy, kappa_over_density, sound_speed_squared,
        lorentz_factor, spatial_metric, det_and_inv_metric.second,
        det_and_inv_metric.first, random_normal);

    Matrix left_matrix = RelativisticEuler::Valencia::left_eigenvectors(
        rest_mass_density, spatial_velocity, specific_internal_energy, pressure,
        specific_enthalpy, kappa_over_density, sound_speed_squared,
        lorentz_factor, spatial_metric, det_and_inv_metric.second,
        det_and_inv_metric.first, random_normal);

    const auto check_identity = [](const auto& left,
                                   const auto& right) noexcept {
      // Very small values of the sound speed lead to large values of some left
      // eigenvectors (e.g Seed = 3106390430), so we use a moderate tolerance.
      Approx local_approx = Approx::custom().epsilon(1.0e-6).scale(1.0);
      const Matrix left_times_right = left * right;
      const Matrix right_times_left = right * left;
      for (size_t i = 0; i < Dim + 2; ++i) {
        for (size_t j = 0; j < Dim + 2; ++j) {
          const double delta_ij = (i == j) ? 1.0 : 0.0;
          CHECK(left_times_right(i, j) == local_approx(delta_ij));
          CHECK(right_times_left(i, j) == local_approx(delta_ij));
        }
      }
    };
    check_identity(left_matrix, right_matrix);

    const auto check_diagonalization = [](const auto& left, const auto& right,
                                          const auto& eigenvalues,
                                          const auto& expected) noexcept {
      Approx local_approx = Approx::custom().epsilon(1.0e-8).scale(1.0);
      const Matrix flux_jacobian = right * eigenvalues * left;
      for (size_t i = 0; i < Dim + 2; ++i) {
        for (size_t j = 0; j < Dim + 2; ++j) {
          CHECK(flux_jacobian(i, j) == local_approx(expected(i, j)));
        }
      }
    };
    check_diagonalization(
        left_matrix, right_matrix,
        matrix_of_eigenvalues(spatial_velocity, spatial_velocity_squared,
                              sound_speed_squared, random_normal),
        expected_flux_jacobian(spatial_velocity, spatial_velocity_oneform,
                               lorentz_factor, specific_enthalpy,
                               sound_speed_squared, kappa_over_density,
                               random_normal, random_normal_vector));

    // test for unit normals along coordinate axes
    for (const auto& direction : Direction<Dim>::all_directions()) {
      const auto normal = unit_basis_form(
          direction, determinant_and_inverse(spatial_metric).second);

      right_matrix = RelativisticEuler::Valencia::right_eigenvectors(
          rest_mass_density, spatial_velocity, specific_internal_energy,
          pressure, specific_enthalpy, kappa_over_density, sound_speed_squared,
          lorentz_factor, spatial_metric, det_and_inv_metric.second,
          det_and_inv_metric.first, normal);

      left_matrix = RelativisticEuler::Valencia::left_eigenvectors(
          rest_mass_density, spatial_velocity, specific_internal_energy,
          pressure, specific_enthalpy, kappa_over_density, sound_speed_squared,
          lorentz_factor, spatial_metric, det_and_inv_metric.second,
          det_and_inv_metric.first, normal);

      check_identity(left_matrix, right_matrix);
      check_diagonalization(
          left_matrix, right_matrix,
          matrix_of_eigenvalues(spatial_velocity, spatial_velocity_squared,
                                sound_speed_squared, normal),
          expected_flux_jacobian(
              spatial_velocity, spatial_velocity_oneform, lorentz_factor,
              specific_enthalpy, sound_speed_squared, kappa_over_density,
              normal, raise_or_lower_index(normal, det_and_inv_metric.second)));
    }
  };
  test_for_given_eos(EquationsOfState::IdealFluid<true>{5.0 / 3.0});
  test_for_given_eos(EquationsOfState::PolytropicFluid<true>{0.001, 4.0 / 3.0});
}

}  // namespace

SPECTRE_TEST_CASE("Unit.RelativisticEuler.Valencia.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/RelativisticEuler/Valencia"};

  GENERATE_UNINITIALIZED_DOUBLE_AND_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_characteristic_speeds, (1, 2, 3));
  CHECK_FOR_DOUBLES(test_characteristic_matrices, (1, 2, 3));
}
