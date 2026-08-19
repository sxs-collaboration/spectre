// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/SecondOrderScalarWave/Characteristics.hpp"
#include "Evolution/Systems/SecondOrderScalarWave/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/SecondOrderWrapper.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
template <size_t Index, size_t Dim>
Scalar<DataVector> speed_with_index(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal) {
  return Scalar<DataVector>{
      SecondOrderScalarWave::characteristic_speeds<Dim>(normal)[Index]};
}

template <size_t Dim>
void test_characteristic_speeds() {
  TestHelpers::db::test_compute_tag<
      SecondOrderScalarWave::Tags::CharacteristicSpeedsCompute<Dim>>(
      "CharacteristicSpeeds");
  const DataVector used_for_size(5);
  pypp::check_with_random_values<1>(speed_with_index<0, Dim>, "Characteristics",
                                    "char_speed_vzero", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<1, Dim>, "Characteristics",
                                    "char_speed_vplus", {{{-10.0, 10.0}}},
                                    used_for_size);
  pypp::check_with_random_values<1>(speed_with_index<2, Dim>, "Characteristics",
                                    "char_speed_vminus", {{{-10.0, 10.0}}},
                                    used_for_size);
}

// Test return-by-reference char speeds through the compute-tag function. The
// speeds are the exact constants 0, +1, and -1, so no approximate comparison
// is needed.
template <size_t Dim>
void test_characteristic_speeds_constant() {
  const size_t n_pts = 5;
  const tnsr::i<DataVector, Dim, Frame::Inertial> unit_normal_one_form{
      DataVector(n_pts, 1. / sqrt(Dim))};

  std::array<DataVector, 3> char_speeds{};
  SecondOrderScalarWave::Tags::CharacteristicSpeedsCompute<Dim>::function(
      &char_speeds, unit_normal_one_form);
  CHECK(char_speeds[0] == DataVector(n_pts, 0.));   // VZero
  CHECK(char_speeds[1] == DataVector(n_pts, 1.));   // VPlus
  CHECK(char_speeds[2] == DataVector(n_pts, -1.));  // VMinus
}

template <typename Tag, size_t Dim>
typename Tag::type field_with_tag(
    const Scalar<DataVector>& pi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& phi,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& normal_one_form) {
  Variables<tmpl::list<SecondOrderScalarWave::Tags::VZero<Dim>,
                       SecondOrderScalarWave::Tags::VPlus,
                       SecondOrderScalarWave::Tags::VMinus>>
      char_fields{};
  SecondOrderScalarWave::Tags::CharacteristicFieldsCompute<Dim>::function(
      make_not_null(&char_fields), pi, phi, normal_one_form);
  return get<Tag>(char_fields);
}

template <size_t Dim>
void test_characteristic_fields() {
  TestHelpers::db::test_compute_tag<
      SecondOrderScalarWave::Tags::CharacteristicFieldsCompute<Dim>>(
      "CharacteristicFields");
  const DataVector used_for_size(5);
  // VZero
  pypp::check_with_random_values<1>(
      field_with_tag<SecondOrderScalarWave::Tags::VZero<Dim>, Dim>,
      "Characteristics", "char_field_vzero", {{{-10., 10.}}}, used_for_size,
      1.e-11);
  // VPlus
  pypp::check_with_random_values<1>(
      field_with_tag<SecondOrderScalarWave::Tags::VPlus, Dim>,
      "Characteristics", "char_field_vplus", {{{-10., 10.}}}, used_for_size);
  // VMinus
  pypp::check_with_random_values<1>(
      field_with_tag<SecondOrderScalarWave::Tags::VMinus, Dim>,
      "Characteristics", "char_field_vminus", {{{-10., 10.}}}, used_for_size);
}

// Test return-by-reference char fields by comparing to analytic solution
template <size_t Dim, typename Solution>
void test_characteristic_fields_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, Dim>& lower_bound,
    const std::array<double, Dim>& upper_bound) {
  // Set up grid
  const Mesh<Dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const double t = 0.;

  // Evaluate analytic solution (Psi is retrieved but not used here)
  const auto vars =
      solution.variables(x, t,
                         tmpl::list<SecondOrderScalarWave::Tags::Psi,
                                    SecondOrderScalarWave::Tags::Pi,
                                    SecondOrderScalarWave::Tags::Phi<Dim>>{});

  const size_t n_pts = mesh.number_of_grid_points();
  const auto& pi = get<SecondOrderScalarWave::Tags::Pi>(vars);
  const auto& phi = get<SecondOrderScalarWave::Tags::Phi<Dim>>(vars);
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

  const auto& vzero_expected = phi_dot_projection_tensor;
  const Scalar<DataVector> vplus_expected{get(pi) + get(phi_dot_normal)};
  const Scalar<DataVector> vminus_expected{get(pi) - get(phi_dot_normal)};

  // Check that locally computed fields match returned ones
  Variables<tmpl::list<SecondOrderScalarWave::Tags::VZero<Dim>,
                       SecondOrderScalarWave::Tags::VPlus,
                       SecondOrderScalarWave::Tags::VMinus>>
      uvars{};
  SecondOrderScalarWave::Tags::CharacteristicFieldsCompute<Dim>::function(
      make_not_null(&uvars), pi, phi, unit_normal_one_form);

  const auto& vzero = get<SecondOrderScalarWave::Tags::VZero<Dim>>(uvars);
  const auto& vplus = get<SecondOrderScalarWave::Tags::VPlus>(uvars);
  const auto& vminus = get<SecondOrderScalarWave::Tags::VMinus>(uvars);

  CHECK_ITERABLE_APPROX(vzero_expected, vzero);
  CHECK_ITERABLE_APPROX(vplus_expected, vplus);
  CHECK_ITERABLE_APPROX(vminus_expected, vminus);
}

template <typename Tag, size_t Dim>
typename Tag::type inverse_field_with_tag(
    const tnsr::i<DataVector, Dim, Frame::Inertial>& v_zero,
    const Scalar<DataVector>& v_plus, const Scalar<DataVector>& v_minus,
    const tnsr::i<DataVector, Dim, Frame::Inertial>& unit_normal_one_form) {
  Variables<tmpl::list<SecondOrderScalarWave::Tags::Pi,
                       SecondOrderScalarWave::Tags::Phi<Dim>>>
      evolved_vars{};
  SecondOrderScalarWave::Tags::FieldsFromInverseCharacteristicTransformCompute<
      Dim>::function(make_not_null(&evolved_vars), v_zero, v_plus, v_minus,
                     unit_normal_one_form);
  return get<Tag>(evolved_vars);
}

template <size_t Dim>
void test_fields_from_inverse_characteristic_transform() {
  TestHelpers::db::test_compute_tag<
      SecondOrderScalarWave::Tags::
          FieldsFromInverseCharacteristicTransformCompute<Dim>>(
      "FieldsFromInverseCharacteristicTransform");
  const DataVector used_for_size(5);
  // Pi
  pypp::check_with_random_values<1>(
      inverse_field_with_tag<SecondOrderScalarWave::Tags::Pi, Dim>,
      "Characteristics", "inverse_field_pi", {{{-10., 10.}}}, used_for_size);
  // Phi
  pypp::check_with_random_values<1>(
      inverse_field_with_tag<SecondOrderScalarWave::Tags::Phi<Dim>, Dim>,
      "Characteristics", "inverse_field_phi", {{{-10., 10.}}}, used_for_size);
}

// Test return-by-reference inverse-transform fields by comparing to analytic
// solution
template <size_t Dim, typename Solution>
void test_fields_from_inverse_characteristic_transform_analytic(
    const Solution& solution, const size_t grid_size_each_dimension,
    const std::array<double, Dim>& lower_bound,
    const std::array<double, Dim>& upper_bound) {
  // Set up grid
  const Mesh<Dim> mesh{grid_size_each_dimension, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
          Affine3D{
              Affine{-1., 1., lower_bound[0], upper_bound[0]},
              Affine{-1., 1., lower_bound[1], upper_bound[1]},
              Affine{-1., 1., lower_bound[2], upper_bound[2]},
          });

  // Set up coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  const double t = 0.;

  // Evaluate analytic solution (Psi is retrieved but not used here)
  const auto vars =
      solution.variables(x, t,
                         tmpl::list<SecondOrderScalarWave::Tags::Psi,
                                    SecondOrderScalarWave::Tags::Pi,
                                    SecondOrderScalarWave::Tags::Phi<Dim>>{});

  const size_t n_pts = mesh.number_of_grid_points();
  const auto& pi_expected = get<SecondOrderScalarWave::Tags::Pi>(vars);
  const auto& phi_expected = get<SecondOrderScalarWave::Tags::Phi<Dim>>(vars);
  const auto unit_normal_one_form =
      make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
          x, 1. / sqrt(Dim));

  // Compute characteristic fields locally
  const auto phi_dot_normal = dot_product(unit_normal_one_form, phi_expected);

  tnsr::i<DataVector, Dim, Frame::Inertial> phi_dot_projection_tensor{
      DataVector(n_pts)};
  for (size_t i = 0; i < Dim; ++i) {
    phi_dot_projection_tensor.get(i) =
        phi_expected.get(i) - unit_normal_one_form.get(i) * get(phi_dot_normal);
  }

  const auto& vzero = phi_dot_projection_tensor;
  const Scalar<DataVector> vplus{get(pi_expected) + get(phi_dot_normal)};
  const Scalar<DataVector> vminus{get(pi_expected) - get(phi_dot_normal)};
  // Obtain reconstructed fields using compute tag
  {
    Variables<tmpl::list<SecondOrderScalarWave::Tags::Pi,
                         SecondOrderScalarWave::Tags::Phi<Dim>>>
        fields{};
    SecondOrderScalarWave::Tags::
        FieldsFromInverseCharacteristicTransformCompute<Dim>::function(
            make_not_null(&fields), vzero, vplus, vminus, unit_normal_one_form);
    const auto& pi = get<SecondOrderScalarWave::Tags::Pi>(fields);
    const auto& phi = get<SecondOrderScalarWave::Tags::Phi<Dim>>(fields);

    CHECK_ITERABLE_APPROX(pi_expected, pi);
    CHECK_ITERABLE_APPROX(phi_expected, phi);
  }
  // Obtain reconstructed fields using function
  {
    const auto fields =
        SecondOrderScalarWave::fields_from_inverse_characteristic_transform(
            vzero, vplus, vminus, unit_normal_one_form);
    const auto& pi = get<SecondOrderScalarWave::Tags::Pi>(fields);
    const auto& phi = get<SecondOrderScalarWave::Tags::Phi<Dim>>(fields);

    CHECK_ITERABLE_APPROX(pi_expected, pi);
    CHECK_ITERABLE_APPROX(phi_expected, phi);
  }
}

// Test that characteristic_fields followed by
// fields_from_inverse_characteristic_transform is the identity on (pi, phi),
// using random data and a random unit normal.
template <size_t Dim>
void test_roundtrip() {
  CAPTURE(Dim);
  MAKE_GENERATOR(gen);
  std::uniform_real_distribution<> dist(-10., 10.);
  const size_t n_pts = 5;

  // Random evolved fields
  auto pi = make_with_value<Scalar<DataVector>>(DataVector(n_pts), 0.);
  auto phi = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
      DataVector(n_pts), 0.);
  fill_with_random_values(make_not_null(&pi), make_not_null(&gen),
                          make_not_null(&dist));
  fill_with_random_values(make_not_null(&phi), make_not_null(&gen),
                          make_not_null(&dist));

  // Random unit normal: generate random direction, then normalize
  auto normal = make_with_value<tnsr::i<DataVector, Dim, Frame::Inertial>>(
      DataVector(n_pts), 0.);
  fill_with_random_values(make_not_null(&normal), make_not_null(&gen),
                          make_not_null(&dist));
  const auto mag = magnitude(normal);
  for (size_t i = 0; i < Dim; ++i) {
    normal.get(i) /= get(mag);
  }

  // Forward: evolved -> characteristic
  const auto char_fields =
      SecondOrderScalarWave::characteristic_fields(pi, phi, normal);
  const auto& v_zero =
      get<SecondOrderScalarWave::Tags::VZero<Dim>>(char_fields);
  const auto& v_plus = get<SecondOrderScalarWave::Tags::VPlus>(char_fields);
  const auto& v_minus = get<SecondOrderScalarWave::Tags::VMinus>(char_fields);

  // Inverse: characteristic -> evolved
  const auto recovered =
      SecondOrderScalarWave::fields_from_inverse_characteristic_transform(
          v_zero, v_plus, v_minus, normal);

  CHECK_ITERABLE_APPROX(pi, get<SecondOrderScalarWave::Tags::Pi>(recovered));
  CHECK_ITERABLE_APPROX(phi,
                        get<SecondOrderScalarWave::Tags::Phi<Dim>>(recovered));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.SecondOrderScalarWave.Characteristics",
    "[Unit][Evolution]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "Evolution/Systems/SecondOrderScalarWave/"};

  test_characteristic_speeds<1>();
  test_characteristic_speeds<2>();
  test_characteristic_speeds<3>();

  test_characteristic_fields<1>();
  test_characteristic_fields<2>();
  test_characteristic_fields<3>();

  test_fields_from_inverse_characteristic_transform<1>();
  test_fields_from_inverse_characteristic_transform<2>();
  test_fields_from_inverse_characteristic_transform<3>();

  test_roundtrip<1>();
  test_roundtrip<2>();
  test_roundtrip<3>();

  test_characteristic_speeds_constant<1>();
  test_characteristic_speeds_constant<2>();
  test_characteristic_speeds_constant<3>();

  // Test characteristics against 3D plane wave
  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.22, 1.32}};
  const std::array<double, 3> upper_bound{{0.78, 1.18, 1.28}};

  const double kx = 1.5;
  const double ky = -7.2;
  const double kz = 2.7;
  const double center_x = 2.4;
  const double center_y = -4.8;
  const double center_z = 8.4;
  const SecondOrderScalarWave::Solutions::SecondOrderWrapper
      plane_wave_solution(ScalarWave::Solutions::PlaneWave<3>(
          {{kx, ky, kz}}, {{center_x, center_y, center_z}},
          std::make_unique<MathFunctions::PowX<1, Frame::Inertial>>(3)));

  test_characteristic_fields_analytic<3>(plane_wave_solution, grid_size,
                                         lower_bound, upper_bound);
  test_fields_from_inverse_characteristic_transform_analytic<3>(
      plane_wave_solution, grid_size, lower_bound, upper_bound);

  double largest_characteristic_speed =
      std::numeric_limits<double>::signaling_NaN();
  SecondOrderScalarWave::Tags::ComputeLargestCharacteristicSpeed::function(
      make_not_null(&largest_characteristic_speed));
  CHECK(largest_characteristic_speed == 1.0);
}
