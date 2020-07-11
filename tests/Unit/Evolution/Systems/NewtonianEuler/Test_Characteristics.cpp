// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <random>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/Systems/NewtonianEuler/Characteristics.hpp"
#include "Evolution/Systems/NewtonianEuler/SoundSpeedSquared.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/DomainTestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {

template <size_t Dim>
void test_compute_item_in_databox(
    const tnsr::I<DataVector, Dim>& velocity,
    const Scalar<DataVector>& sound_speed,
    const tnsr::i<DataVector, Dim>& normal) noexcept {
  TestHelpers::db::test_compute_tag<
      NewtonianEuler::Tags::CharacteristicSpeedsCompute<Dim>>(
      "CharacteristicSpeeds");
  const auto box = db::create<
      db::AddSimpleTags<
          NewtonianEuler::Tags::Velocity<DataVector, Dim>,
          NewtonianEuler::Tags::SoundSpeed<DataVector>,
          ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<Dim>>>,
      db::AddComputeTags<
          NewtonianEuler::Tags::CharacteristicSpeedsCompute<Dim>>>(
      velocity, sound_speed, normal);
  CHECK(NewtonianEuler::characteristic_speeds(velocity, sound_speed, normal) ==
        db::get<NewtonianEuler::Tags::CharacteristicSpeeds<Dim>>(box));
}

template <size_t Dim>
void test_characteristic_speeds(const DataVector& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(0.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto velocity = make_with_random_values<tnsr::I<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto sound_speed = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);

  // test for normal along coordinate axes
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto normal = euclidean_basis_vector(direction, used_for_size);

    CHECK_ITERABLE_APPROX(
        NewtonianEuler::characteristic_speeds(velocity, sound_speed, normal),
        (pypp::call<std::array<DataVector, Dim + 2>>(
            "TestFunctions", "characteristic_speeds", velocity, sound_speed,
            normal)));
  }

  // test for normal of random orientation
  pypp::check_with_random_values<3>(
      static_cast<std::array<DataVector, Dim + 2> (*)(
          const tnsr::I<DataVector, Dim>&, const Scalar<DataVector>&,
          const tnsr::i<DataVector, Dim>&)>(
          &NewtonianEuler::characteristic_speeds<Dim>),
      "TestFunctions", "characteristic_speeds",
      {{{-1.0, 1.0}, {0.0, 1.0}, {-1.0, 1.0}}}, used_for_size);
  test_compute_item_in_databox(
      velocity, sound_speed,
      make_with_random_values<tnsr::i<DataVector, Dim>>(
          nn_generator, nn_distribution, used_for_size));
}

template <size_t Dim>
void test_largest_characteristic_speed(
    const DataVector& used_for_size) noexcept {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1.0, 1.0);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);

  const auto velocity = make_with_random_values<tnsr::I<DataVector, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto sound_speed = make_with_random_values<Scalar<DataVector>>(
      nn_generator, nn_distribution, used_for_size);
  CHECK(NewtonianEuler::ComputeLargestCharacteristicSpeed<Dim>::apply(
            velocity, sound_speed) ==
        max(get(sound_speed) + get(magnitude(velocity))));
}

template <size_t Dim>
Matrix analytic_flux_jacobian(const tnsr::I<double, Dim>& velocity,
                              double kappa_over_density, double b_times_theta,
                              double specific_enthalpy,
                              const tnsr::i<double, Dim>& unit_normal) noexcept;

template <>
Matrix analytic_flux_jacobian<1>(
    const tnsr::I<double, 1>& velocity, const double kappa_over_density,
    const double b_times_theta, const double specific_enthalpy,
    const tnsr::i<double, 1>& unit_normal) noexcept {
  const double n_x = get<0>(unit_normal);
  const double u = get<0>(velocity);
  const Matrix a_x{{0., 1., 0.},
                   {b_times_theta - square(u), u * (2. - kappa_over_density),
                    kappa_over_density},
                   {u * (b_times_theta - specific_enthalpy),
                    specific_enthalpy - kappa_over_density * square(u),
                    u * (kappa_over_density + 1.)}};
  return n_x * a_x;
}

template <>
Matrix analytic_flux_jacobian<2>(
    const tnsr::I<double, 2>& velocity, const double kappa_over_density,
    const double b_times_theta, const double specific_enthalpy,
    const tnsr::i<double, 2>& unit_normal) noexcept {
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const Matrix a_x{
      {0., 1., 0., 0.},
      {b_times_theta - square(u), u * (2. - kappa_over_density),
       -v * kappa_over_density, kappa_over_density},
      {-u * v, v, u, 0.},
      {u * (b_times_theta - specific_enthalpy),
       specific_enthalpy - kappa_over_density * square(u),
       -u * v * kappa_over_density, u * (kappa_over_density + 1.)}};
  const Matrix a_y{
      {0., 0., 1., 0.},
      {-u * v, v, u, 0.},
      {b_times_theta - square(v), -u * kappa_over_density,
       v * (2. - kappa_over_density), kappa_over_density},
      {v * (b_times_theta - specific_enthalpy), -u * v * kappa_over_density,
       specific_enthalpy - kappa_over_density * square(v),
       v * (kappa_over_density + 1.)}};
  return n_x * a_x + n_y * a_y;
}

template <>
Matrix analytic_flux_jacobian<3>(
    const tnsr::I<double, 3>& velocity, const double kappa_over_density,
    const double b_times_theta, const double specific_enthalpy,
    const tnsr::i<double, 3>& unit_normal) noexcept {
  const double n_x = get<0>(unit_normal);
  const double n_y = get<1>(unit_normal);
  const double n_z = get<2>(unit_normal);
  const double u = get<0>(velocity);
  const double v = get<1>(velocity);
  const double w = get<2>(velocity);
  const Matrix a_x{
      {0., 1., 0., 0., 0.},
      {b_times_theta - square(u), u * (2. - kappa_over_density),
       -v * kappa_over_density, -w * kappa_over_density, kappa_over_density},
      {-u * v, v, u, 0., 0.},
      {-u * w, w, 0., u, 0.},
      {u * (b_times_theta - specific_enthalpy),
       specific_enthalpy - kappa_over_density * square(u),
       -u * v * kappa_over_density, -u * w * kappa_over_density,
       u * (kappa_over_density + 1.)}};
  const Matrix a_y{
      {0., 0., 1., 0., 0.},
      {-u * v, v, u, 0., 0.},
      {b_times_theta - square(v), -u * kappa_over_density,
       v * (2. - kappa_over_density), -w * kappa_over_density,
       kappa_over_density},
      {-v * w, 0., w, v, 0.},
      {v * (b_times_theta - specific_enthalpy), -u * v * kappa_over_density,
       specific_enthalpy - kappa_over_density * square(v),
       -v * w * kappa_over_density, v * (kappa_over_density + 1.)}};
  const Matrix a_z{{0., 0., 0., 1., 0.},
                   {-u * w, w, 0., u, 0.},
                   {-v * w, 0., w, v, 0.},
                   {b_times_theta - square(w), -u * kappa_over_density,
                    -v * kappa_over_density, w * (2. - kappa_over_density),
                    kappa_over_density},
                   {w * (b_times_theta - specific_enthalpy),
                    -u * w * kappa_over_density, -v * w * kappa_over_density,
                    specific_enthalpy - kappa_over_density * square(w),
                    w * (kappa_over_density + 1.)}};
  return n_x * a_x + n_y * a_y + n_z * a_z;
}

template <size_t Dim, size_t ThermodynamicDim>
void test_left_and_right_eigenvectors_impl(
    const Scalar<double>& density, const tnsr::I<double, Dim>& velocity,
    const Scalar<double>& specific_internal_energy,
    const tnsr::i<double, Dim>& unit_normal,
    const EquationsOfState::EquationOfState<false, ThermodynamicDim>&
        equation_of_state) noexcept {
  Scalar<double> pressure{};
  Scalar<double> kappa_over_density{};
  if constexpr (ThermodynamicDim == 1) {
    pressure = equation_of_state.pressure_from_density(density);
    kappa_over_density = Scalar<double>{
        {{get(equation_of_state.kappa_times_p_over_rho_squared_from_density(
              density)) *
          get(density) / get(pressure)}}};
  } else if constexpr (ThermodynamicDim == 2) {
    pressure = equation_of_state.pressure_from_density_and_energy(
        density, specific_internal_energy);
    kappa_over_density = Scalar<double>{
        {{get(equation_of_state
                  .kappa_times_p_over_rho_squared_from_density_and_energy(
                      density, specific_internal_energy)) *
          get(density) / get(pressure)}}};
  }

  const Scalar<double> v_squared{{{get(dot_product(velocity, velocity))}}};

  const auto sound_speed_squared = NewtonianEuler::sound_speed_squared(
      density, specific_internal_energy, equation_of_state);
  const Scalar<double> energy_density{
      {{get(density) * get(specific_internal_energy) +
        0.5 * get(density) * get(v_squared)}}};
  const Scalar<double> specific_enthalpy{
      {{(get(energy_density) + get(pressure)) / get(density)}}};

  const Matrix right = NewtonianEuler::right_eigenvectors(
      velocity, sound_speed_squared, specific_enthalpy, kappa_over_density,
      unit_normal);
  const Matrix left = NewtonianEuler::left_eigenvectors(
      velocity, sound_speed_squared, specific_enthalpy, kappa_over_density,
      unit_normal);

  // Check that eigenvectors are inverses of each other
  const auto id1 = left * right;
  const auto id2 = right * left;

  // For small values of specific_internal_energy, the relative error can
  // increase from the default level
  Approx local_approx = Approx::custom().epsilon(1e-12).scale(1.0);
  for (size_t i = 0; i < Dim + 2; ++i) {
    for (size_t j = 0; j < Dim + 2; ++j) {
      const double delta_ij = (i == j) ? 1. : 0.;
      CHECK(id1(i, j) == local_approx(delta_ij));
      CHECK(id2(i, j) == local_approx(delta_ij));
    }
  }

  // Check that eigenvectors give correct fluxes
  const double v_n = get(dot_product(unit_normal, velocity));
  Matrix eigenvalues(Dim + 2, Dim + 2, 0.);
  for (size_t i = 0; i < Dim + 2; ++i) {
    eigenvalues(i, i) = v_n;
  }
  eigenvalues(0, 0) -= sqrt(get(sound_speed_squared));
  eigenvalues(Dim + 1, Dim + 1) += sqrt(get(sound_speed_squared));
  const auto flux_jacobian = right * eigenvalues * left;

  const double b_times_theta =
      get(kappa_over_density) * (get(v_squared) - get(specific_enthalpy)) +
      get(sound_speed_squared);
  const auto expected_flux_jacobian =
      analytic_flux_jacobian(velocity, get(kappa_over_density), b_times_theta,
                             get(specific_enthalpy), unit_normal);

  for (size_t i = 0; i < Dim + 2; ++i) {
    for (size_t j = 0; j < Dim + 2; ++j) {
      CHECK(flux_jacobian(i, j) == local_approx(expected_flux_jacobian(i, j)));
    }
  }
}

template <size_t Dim>
void test_right_and_left_eigenvectors() noexcept {
  // This test verifies that the eigenvectors satisfy the conditions by which
  // they are defined:
  // - the right and left eigenvectors are matrix inverses of each other, i.e.,
  //   left * right == right * left == identity
  // - the right and left eigenvectors diagonalize the flux Jacobian, i.e.,
  //   right * eigenvalues * left == flux
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-1., 1.);
  std::uniform_real_distribution<> distribution_positive(1e-3, 1.);

  const auto nn_generator = make_not_null(&generator);
  const auto nn_distribution = make_not_null(&distribution);
  const auto nn_distribution_positive = make_not_null(&distribution_positive);

  const double used_for_size = 0.;
  // This computes a unit normal. It is NOT uniformly distributed in angle,
  // but for this test the angular distribution is not important.
  const auto unit_normal = [&]() noexcept {
    auto result = make_with_random_values<tnsr::i<double, Dim>>(
        nn_generator, nn_distribution, used_for_size);
    double normal_magnitude = get(magnitude(result));
    // Though highly unlikely, the normal could have length nearly 0. If this
    // happens, we edit the normal to make it non-zero.
    if (normal_magnitude < 1e-3) {
      get<0>(result) += 0.9;
      normal_magnitude = get(magnitude(result));
    }
    for (auto& n_i : result) {
      n_i /= normal_magnitude;
    }
    return result;
  }
  ();

  // To check the diagonalization of the Jacobian, we need a self consistent set
  // of primitive and derived-from-primitive variables -- so generate everything
  // from the primitives
  const auto density = make_with_random_values<Scalar<double>>(
      nn_generator, nn_distribution_positive, used_for_size);
  const auto velocity = make_with_random_values<tnsr::I<double, Dim>>(
      nn_generator, nn_distribution, used_for_size);
  const auto specific_internal_energy = make_with_random_values<Scalar<double>>(
      nn_generator, nn_distribution_positive, used_for_size);

  const EquationsOfState::IdealFluid<false> ideal_gas_eos{5. / 3.};
  test_left_and_right_eigenvectors_impl(
      density, velocity, specific_internal_energy, unit_normal, ideal_gas_eos);

  const EquationsOfState::PolytropicFluid<false> polytropic_eos{100., 2.};
  test_left_and_right_eigenvectors_impl(
      density, velocity, specific_internal_energy, unit_normal, polytropic_eos);

  // Test also in cases where unit normal is aligned to coordinate axes
  for (const auto& direction : Direction<Dim>::all_directions()) {
    const auto aligned = euclidean_basis_vector(direction, used_for_size);

    test_left_and_right_eigenvectors_impl(
        density, velocity, specific_internal_energy, aligned, ideal_gas_eos);

    test_left_and_right_eigenvectors_impl(
        density, velocity, specific_internal_energy, aligned, polytropic_eos);
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.NewtonianEuler.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/NewtonianEuler"};

  GENERATE_UNINITIALIZED_DATAVECTOR;
  CHECK_FOR_DATAVECTORS(test_characteristic_speeds, (1, 2, 3))
  CHECK_FOR_DATAVECTORS(test_largest_characteristic_speed, (1, 2, 3))

  test_right_and_left_eigenvectors<1>();
  test_right_and_left_eigenvectors<2>();
  test_right_and_left_eigenvectors<3>();
}
