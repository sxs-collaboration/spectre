// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"        // IWYU pragma: keep
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/Systems/ScalarWave/Characteristics.hpp"
#include "Evolution/Systems/ScalarWave/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/RegularSphericalWave.hpp"
#include "PointwiseFunctions/MathFunctions/Gaussian.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::dt
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
template <size_t Index, size_t Dim>
Scalar<DataVector> speed_with_index(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal) {
  return Scalar<DataVector>{
      ScalarWave::characteristic_speeds<Dim>(normal)[Index]};
}

template <size_t Dim>
void test_characteristic_speeds() noexcept {
  TestHelpers::db::test_compute_tag<
      ScalarWave::Tags::CharacteristicSpeedsCompute<Dim>>(
      "CharacteristicSpeeds");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(speed_with_index<0, Dim>, "Characteristics",
                                    "char_speed_vpsi", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<1, Dim>, "Characteristics",
                                    "char_speed_vzero", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<3, Dim>, "Characteristics",
                                    "char_speed_vminus", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<2, Dim>, "Characteristics",
                                    "char_speed_vplus", {{{-10.0, 10.0}}},
                                    used_for_size);
}

// Test return-by-reference char speeds by comparing to analytic solution
template <size_t Dim>
void test_characteristic_speeds_analytic(
    const size_t grid_size_each_dimension) noexcept {
  // Setup mesh
  Mesh<Dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
  // Get ingredients
  const size_t n_pts = mesh.number_of_grid_points();
  // Outward 3-normal to the surface on which characteristic fields are needed
  const tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form{
      DataVector(n_pts, 1. / sqrt(Dim))};

  // Get characteristic speeds locally
  const auto vpsi_speed_expected =
      make_with_value<Scalar<DataVector>>(unit_normal_one_form, 0.);
  const auto vzero_speed_expected =
      make_with_value<Scalar<DataVector>>(unit_normal_one_form, 0.);
  const auto vplus_speed_expected =
      make_with_value<Scalar<DataVector>>(unit_normal_one_form, 1.);
  const auto vminus_speed_expected =
      make_with_value<Scalar<DataVector>>(unit_normal_one_form, -1.);

  // Check that locally computed fields match returned ones
  std::array<DataVector, 4> char_speeds{};
  ScalarWave::Tags::CharacteristicSpeedsCompute<Dim>::function(
      &char_speeds, unit_normal_one_form);
  const auto& vpsi_speed = char_speeds[0];
  const auto& vzero_speed = char_speeds[1];
  const auto& vplus_speed = char_speeds[2];
  const auto& vminus_speed = char_speeds[3];

  CHECK_ITERABLE_APPROX(vpsi_speed_expected.get(), vpsi_speed);
  CHECK_ITERABLE_APPROX(vzero_speed_expected.get(), vzero_speed);
  CHECK_ITERABLE_APPROX(vplus_speed_expected.get(), vplus_speed);
  CHECK_ITERABLE_APPROX(vminus_speed_expected.get(), vminus_speed);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim>
typename Tag::type field_with_tag(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& psi,
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_one_form) {
  Variables<tmpl::list<ScalarWave::Tags::VPsi, ScalarWave::Tags::VZero<Dim>,
                       ScalarWave::Tags::VPlus, ScalarWave::Tags::VMinus>>
      char_fields{};
  ScalarWave::Tags::CharacteristicFieldsCompute<Dim>::function(
      make_not_null(&char_fields), gamma_2, psi, pi, phi, normal_one_form);
  return get<Tag>(char_fields);
}

template <size_t Dim>
void test_characteristic_fields() noexcept {
  TestHelpers::db::test_compute_tag<
      ScalarWave::Tags::CharacteristicFieldsCompute<Dim>>(
      "CharacteristicFields");
  const DataVector used_for_size(5);
  // VPsi
  pypp::check_with_random_values<1>(field_with_tag<ScalarWave::Tags::VPsi, Dim>,
                                    "Characteristics", "char_field_vpsi",
                                    {{{-100., 100.}}}, used_for_size);
  // VZero
  pypp::check_with_random_values<1>(
      field_with_tag<ScalarWave::Tags::VZero<Dim>, Dim>, "Characteristics",
      "char_field_vzero", {{{-100., 100.}}}, used_for_size, 1.e-11);
  // VPlus
  pypp::check_with_random_values<1>(
      field_with_tag<ScalarWave::Tags::VPlus, Dim>, "Characteristics",
      "char_field_vplus", {{{-100., 100.}}}, used_for_size);
  // VMinus
  pypp::check_with_random_values<1>(
      field_with_tag<ScalarWave::Tags::VMinus, Dim>, "Characteristics",
      "char_field_vminus", {{{-100., 100.}}}, used_for_size);
}

// Test return-by-reference char fields by comparing to analytic solution
template <size_t Dim, typename Solution>
void test_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Set up grid
  Mesh<Dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-dependent solution.
  const double t = 0.;

  // Evaluate analytic solution
  const auto vars = solution.variables(
      x, t,
      tmpl::list<ScalarWave::Pi, ScalarWave::Phi<Dim>, ScalarWave::Psi>{});
  // Get ingredients
  const size_t n_pts = mesh.number_of_grid_points();
  const auto gamma_2 = make_with_value<Scalar<DataVector>>(x, 0.1);
  const auto& psi = get<ScalarWave::Psi>(vars);
  const auto& phi = get<ScalarWave::Phi<Dim>>(vars);
  const auto& pi = get<ScalarWave::Pi>(vars);
  // Outward 3-normal to the surface on which characteristic fields are needed
  const auto unit_normal_one_form =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          x, 1. / sqrt(Dim));

  // Compute characteristic fields locally
  const auto phi_dot_normal = dot_product(unit_normal_one_form, phi);

  tnsr::i<DataVector, Dim, Frame::Inertial> phi_dot_projection_tensor{
      DataVector(n_pts)};
  for (size_t i = 0; i < Dim; ++i) {
    phi_dot_projection_tensor.get(i) =
        phi.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  const auto& vpsi_expected = psi;
  const auto& vzero_expected = phi_dot_projection_tensor;
  const Scalar<DataVector> vplus_expected{get(pi) + get(phi_dot_normal) -
                                          get(gamma_2) * get(psi)};
  const Scalar<DataVector> vminus_expected{get(pi) - get(phi_dot_normal) -
                                           get(gamma_2) * get(psi)};

  // Check that locally computed fields match returned ones
  Variables<tmpl::list<ScalarWave::Tags::VPsi, ScalarWave::Tags::VZero<Dim>,
                       ScalarWave::Tags::VPlus, ScalarWave::Tags::VMinus>>
      uvars{};
  ScalarWave::Tags::CharacteristicFieldsCompute<Dim>::function(
      make_not_null(&uvars), gamma_2, psi, pi, phi, unit_normal_one_form);

  const auto& vpsi = get<ScalarWave::Tags::VPsi>(uvars);
  const auto& vzero = get<ScalarWave::Tags::VZero<Dim>>(uvars);
  const auto& vplus = get<ScalarWave::Tags::VPlus>(uvars);
  const auto& vminus = get<ScalarWave::Tags::VMinus>(uvars);

  CHECK_ITERABLE_APPROX(vpsi_expected, vpsi);
  CHECK_ITERABLE_APPROX(vzero_expected, vzero);
  CHECK_ITERABLE_APPROX(vplus_expected, vplus);
  CHECK_ITERABLE_APPROX(vminus_expected, vminus);
}
}  // namespace

namespace {
template <typename Tag, size_t Dim>
typename Tag::type evol_field_with_tag(
    const Scalar<DataVector>& gamma_2, const Scalar<DataVector>& v_psi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<tmpl::list<ScalarWave::Psi, ScalarWave::Pi, ScalarWave::Phi<Dim>>>
      evolved_vars{};
  ScalarWave::Tags::EvolvedFieldsFromCharacteristicFieldsCompute<Dim>::function(
      make_not_null(&evolved_vars), gamma_2, v_psi, v_zero, v_plus, v_minus,
      unit_normal_one_form);
  return get<Tag>(evolved_vars);
}

template <size_t Dim>
void test_evolved_from_characteristic_fields() noexcept {
  TestHelpers::db::test_compute_tag<
      ScalarWave::Tags::EvolvedFieldsFromCharacteristicFieldsCompute<Dim>>(
      "EvolvedFieldsFromCharacteristicFields");
  const DataVector used_for_size(5);
  // Psi
  pypp::check_with_random_values<1>(evol_field_with_tag<ScalarWave::Psi, Dim>,
                                    "Characteristics", "evol_field_psi",
                                    {{{-100., 100.}}}, used_for_size);
  // Pi
  pypp::check_with_random_values<1>(evol_field_with_tag<ScalarWave::Pi, Dim>,
                                    "Characteristics", "evol_field_pi",
                                    {{{-100., 100.}}}, used_for_size);
  // Phi
  pypp::check_with_random_values<1>(
      evol_field_with_tag<ScalarWave::Phi<Dim>, Dim>, "Characteristics",
      "evol_field_phi", {{{-100., 100.}}}, used_for_size);
}

// Test return-by-reference evolved fields by comparing to analytic solution
template <size_t Dim, typename Solution>
void test_evolved_from_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, 3>& lower_bound,
    const std::array<double, 3>& upper_bound) noexcept {
  // Set up grid
  Mesh<Dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-dependent solution.
  const double t = 0.;

  // Evaluate analytic solution
  const auto vars = solution.variables(
      x, t,
      tmpl::list<ScalarWave::Pi, ScalarWave::Phi<Dim>, ScalarWave::Psi>{});
  // Get ingredients
  const size_t n_pts = mesh.number_of_grid_points();
  const auto gamma_2 = make_with_value<Scalar<DataVector>>(x, 0.1);
  const auto& psi_expected = get<ScalarWave::Psi>(vars);
  const auto& phi_expected = get<ScalarWave::Phi<Dim>>(vars);
  const auto& pi_expected = get<ScalarWave::Pi>(vars);
  // Outward 3-normal to the surface on which characteristic fields are needed
  const auto unit_normal_one_form =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          x, 1. / sqrt(Dim));

  // Fundamental fields (psi, pi, phi) have already been computed locally.
  // Now, check that these locally computed fields match returned ones
  // First, compute characteristic fields locally
  const auto phi_dot_normal = dot_product(unit_normal_one_form, phi_expected);

  tnsr::i<DataVector, Dim, Frame::Inertial> phi_dot_projection_tensor{
      DataVector(n_pts)};
  for (size_t i = 0; i < Dim; ++i) {
    phi_dot_projection_tensor.get(i) =
        phi_expected.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  const auto& vpsi = psi_expected;
  const auto& vzero = phi_dot_projection_tensor;
  const Scalar<DataVector> vplus{get(pi_expected) + get(phi_dot_normal) -
                                 get(gamma_2) * get(psi_expected)};
  const Scalar<DataVector> vminus{get(pi_expected) - get(phi_dot_normal) -
                                  get(gamma_2) * get(psi_expected)};
  // Second, obtain evolved fields using compute tag
  {
    Variables<tmpl::list<ScalarWave::Psi, ScalarWave::Pi, ScalarWave::Phi<Dim>>>
        fields{};
    ScalarWave::Tags::EvolvedFieldsFromCharacteristicFieldsCompute<
        Dim>::function(make_not_null(&fields), gamma_2, vpsi, vzero, vplus,
                       vminus, unit_normal_one_form);
    const auto& psi = get<ScalarWave::Psi>(fields);
    const auto& pi = get<ScalarWave::Pi>(fields);
    const auto& phi = get<ScalarWave::Phi<Dim>>(fields);

    CHECK_ITERABLE_APPROX(psi_expected, psi);
    CHECK_ITERABLE_APPROX(pi_expected, pi);
    CHECK_ITERABLE_APPROX(phi_expected, phi);
  }
  // Third, obtain evolved fields using function
  {
    const auto fields = ScalarWave::evolved_fields_from_characteristic_fields(
        gamma_2, vpsi, vzero, vplus, vminus, unit_normal_one_form);
    const auto& psi = get<ScalarWave::Psi>(fields);
    const auto& pi = get<ScalarWave::Pi>(fields);
    const auto& phi = get<ScalarWave::Phi<Dim>>(fields);

    CHECK_ITERABLE_APPROX(psi_expected, psi);
    CHECK_ITERABLE_APPROX(pi_expected, pi);
    CHECK_ITERABLE_APPROX(phi_expected, phi);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.Systems.ScalarWave.Characteristics",
                  "[Unit][Evolution]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/ScalarWave/"};

  test_characteristic_speeds<1>();
  test_characteristic_speeds<2>();
  test_characteristic_speeds<3>();

  test_characteristic_fields<1>();
  test_characteristic_fields<2>();
  test_characteristic_fields<3>();

  test_evolved_from_characteristic_fields<1>();
  test_evolved_from_characteristic_fields<2>();
  test_evolved_from_characteristic_fields<3>();

  // Test characteristics against 3D spherical wave
  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.22, 1.32}};
  const std::array<double, 3> upper_bound{{0.78, 1.18, 1.28}};

  const ScalarWave::Solutions::RegularSphericalWave spherical_wave_solution(
      std::make_unique<MathFunctions::Gaussian>(1., 1., 0.));

  test_characteristic_speeds_analytic<3>(grid_size);
  test_characteristic_fields_analytic<3>(spherical_wave_solution, grid_size,
                                         lower_bound, upper_bound);
  test_evolved_from_characteristic_fields_analytic<3>(
      spherical_wave_solution, grid_size, lower_bound, upper_bound);

  // Test characteristics against 3D plane wave
  const double kx = 1.5;
  const double ky = -7.2;
  const double kz = 2.7;
  const double center_x = 2.4;
  const double center_y = -4.8;
  const double center_z = 8.4;
  const ScalarWave::Solutions::PlaneWave<3> plane_wave_solution(
      {{kx, ky, kz}}, {{center_x, center_y, center_z}},
      std::make_unique<MathFunctions::PowX>(3));

  test_characteristic_speeds_analytic<3>(grid_size);
  test_characteristic_fields_analytic<3>(plane_wave_solution, grid_size,
                                         lower_bound, upper_bound);
  test_evolved_from_characteristic_fields_analytic<3>(
      plane_wave_solution, grid_size, lower_bound, upper_bound);
}
