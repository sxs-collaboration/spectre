// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"

class DataVector;
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace {
template <size_t Dim>
Matrix analytic_rotation_matrix(double rotation_angle);
template <size_t Dim>
Matrix analytic_rotation_matrix_deriv(double rotation_angle,
                                      double angular_velocity);

template <>
Matrix analytic_rotation_matrix<2>(const double rotation_angle) {
  return Matrix{{cos(rotation_angle), -sin(rotation_angle)},
                {sin(rotation_angle), cos(rotation_angle)}};
}

template <>
Matrix analytic_rotation_matrix_deriv<2>(const double rotation_angle,
                                         const double angular_velocity) {
  return Matrix{{-angular_velocity * sin(rotation_angle),
                 -angular_velocity * cos(rotation_angle)},
                {angular_velocity * cos(rotation_angle),
                 -angular_velocity * sin(rotation_angle)}};
}

template <>
Matrix analytic_rotation_matrix<3>(const double rotation_angle) {
  const double c = cos(rotation_angle);
  const double s = sin(rotation_angle);
  const double vx = 1.0 / sqrt(3.0);
  const double vy = -vx;
  const double vz = vx;
  // rotate about (1, -1, 1) /root(3)
  return Matrix{{c + vx * vx * (1.0 - c), vx * vy * (1.0 - c) - vz * s,
                 vx * vz * (1.0 - c) + vy * s},
                {vy * vx * (1.0 - c) + vz * s, c + vy * vy * (1.0 - c),
                 vy * vz * (1.0 - c) - vx * s},
                {vz * vx * (1.0 - c) - vy * s, vz * vy * (1.0 - c) + vx * s,
                 c + vz * vz * (1.0 - c)}};
}

template <>
Matrix analytic_rotation_matrix_deriv<3>(const double rotation_angle,
                                         const double angular_velocity) {
  const double c = cos(rotation_angle);
  const double s = sin(rotation_angle);
  const double vx = 1.0 / sqrt(3.0);
  const double vy = -vx;
  const double vz = vx;
  // rotate about (1.0, -1.0, 1.0) /root(3)
  return Matrix{{angular_velocity * (vx * vx - 1.0) * s,
                 angular_velocity * (vx * vy * s - vz * c),
                 angular_velocity * (vx * vz * s + vy * c)},
                {angular_velocity * (vx * vy * s + vz * c),
                 angular_velocity * (vy * vy - 1.0) * s,
                 angular_velocity * (vy * vz * s - vx * c)},
                {angular_velocity * (vx * vz * s - vy * c),
                 angular_velocity * (vy * vz * s + vx * c),
                 angular_velocity * (vz * vz - 1.0) * s}};
}

template <size_t Dim>
std::array<double, Dim> expected_mapped_point(
    const std::array<double, Dim> initial_unmapped_point, const double time) {
  const double rotation_angle = square(time);
  const Matrix rotation_matrix = analytic_rotation_matrix<Dim>(rotation_angle);

  std::array<double, Dim> result{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) = rotation_matrix(i, 0) * initial_unmapped_point[0];
    for (size_t j = 1; j < Dim; j++) {
      gsl::at(result, i) +=
          rotation_matrix(i, j) * gsl::at(initial_unmapped_point, j);
    }
  }

  return result;
}

template <size_t Dim>
std::array<double, Dim> expected_frame_velocity(
    const std::array<double, Dim> initial_unmapped_point, const double time) {
  const double rotation_angle = square(time);
  const double angular_velocity = 2.0 * time;

  const Matrix rotation_matrix_deriv =
      analytic_rotation_matrix_deriv<Dim>(rotation_angle, angular_velocity);

  std::array<double, Dim> result{};
  for (size_t i = 0; i < Dim; i++) {
    gsl::at(result, i) =
        rotation_matrix_deriv(i, 0) * initial_unmapped_point[0];
    for (size_t j = 1; j < Dim; j++) {
      gsl::at(result, i) +=
          rotation_matrix_deriv(i, j) * gsl::at(initial_unmapped_point, j);
    }
  }

  return result;
}

template <size_t Dim>
void test_rotation_map() {
  MAKE_GENERATOR(generator);
  double t{-1.0};
  const double dt{0.6};
  const double final_time{4.0};
  constexpr size_t deriv_order{3};

  INFO("Spatial Dim = " + get_output(Dim));
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using QuatFoT =
      domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;

  const std::string f_of_t_name{"rotation_angle"};
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  std::array<DataVector, deriv_order + 1> init_func{};
  // Just need a placeholder for now. The correct epsilons will be set below
  Approx custom_approx = Approx::custom().epsilon(1.0).scale(1.0);
  // angle(t) = t^2
  // omega(t) = 2 * t
  // dtomega(t) = 2
  const double angle = 1.0;
  const double omega = -2.0;
  const double dtomega = 2.0;
  const double d2tomega = 0.0;
  if constexpr (Dim == 2) {
    init_func = {{{angle}, {omega}, {dtomega}, {d2tomega}}};
    f_of_t_list[f_of_t_name] =
        std::make_unique<Polynomial>(t, init_func, final_time + dt);
    custom_approx = Approx::custom().epsilon(5.0e-13).scale(1.0);
  } else {
    // Axis of rotation nhat = (1.0, -1.0, 1.0) / sqrt(3.0)
    DataVector axis{{1.0, -1.0, 1.0}};
    axis /= sqrt(3.0);
    init_func = {axis * angle, axis * omega, axis * dtomega, axis * d2tomega};
    // initial quaternion is (cos(angle/2), nhat*sin(angle/2))
    const std::array<DataVector, 1> init_quat{
        DataVector{{cos(angle / 2.0), axis[0] * sin(angle / 2.0),
                    axis[1] * sin(angle / 2.0), axis[2] * sin(angle / 2.0)}}};
    f_of_t_list[f_of_t_name] =
        std::make_unique<QuatFoT>(t, init_quat, init_func, final_time + dt);
    custom_approx = Approx::custom().epsilon(5.0e-11).scale(1.0);
  }

  const domain::CoordinateMaps::TimeDependent::Rotation<Dim> rotation_map{
      f_of_t_name};
  const auto rotation_map_deserialized =
      serialize_and_deserialize(rotation_map);

  const std::uniform_real_distribution<double> dist{0.0, 5.0};
  auto initial_unmapped_point =
      make_with_random_values<std::array<double, Dim>>(
          make_not_null(&generator), dist, std::array<double, Dim>{});

  while (t < final_time) {
    const std::array<double, Dim> expected{
        expected_mapped_point(initial_unmapped_point, t)};
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map(initial_unmapped_point, t, f_of_t_list), expected,
        custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map.inverse(expected, t, f_of_t_list).value(),
        initial_unmapped_point, custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map.frame_velocity(initial_unmapped_point, t, f_of_t_list),
        expected_frame_velocity(initial_unmapped_point, t), custom_approx);

    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map_deserialized(initial_unmapped_point, t, f_of_t_list),
        expected, custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map_deserialized.inverse(expected, t, f_of_t_list).value(),
        initial_unmapped_point, custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map_deserialized.frame_velocity(initial_unmapped_point, t,
                                                 f_of_t_list),
        expected_frame_velocity(initial_unmapped_point, t), custom_approx);

    const auto jac{
        rotation_map.jacobian(initial_unmapped_point, t, f_of_t_list)};
    const auto inv_jac{
        rotation_map.inv_jacobian(initial_unmapped_point, t, f_of_t_list)};
    const auto jac_deserialized{rotation_map_deserialized.jacobian(
        initial_unmapped_point, t, f_of_t_list)};
    const auto inv_jac_deserialized{rotation_map_deserialized.inv_jacobian(
        initial_unmapped_point, t, f_of_t_list)};

    const double rotation_angle = square(t);
    const Matrix analytic_jac = analytic_rotation_matrix<Dim>(rotation_angle);

    for (size_t i = 0; i < Dim; i++) {
      for (size_t j = 0; j < Dim; j++) {
        // Jacobian is same as rotation matrix
        CHECK(jac.get(i, j) == custom_approx(analytic_jac(i, j)));
        CHECK(jac_deserialized.get(i, j) == custom_approx(analytic_jac(i, j)));
        // Inv Jacobian is same as inv of rotation matrix
        CHECK(inv_jac.get(i, j) == custom_approx(analytic_jac(j, i)));
        CHECK(inv_jac_deserialized.get(i, j) ==
              custom_approx(analytic_jac(j, i)));
      }
    }
    t += dt;
  }

  // Check inequivalence operator
  CHECK_FALSE(rotation_map != rotation_map);
  CHECK_FALSE(rotation_map_deserialized != rotation_map_deserialized);

  // Check serialization
  CHECK(rotation_map == rotation_map_deserialized);
  CHECK_FALSE(rotation_map != rotation_map_deserialized);

  test_coordinate_map_argument_types(rotation_map, initial_unmapped_point, t,
                                     f_of_t_list);
  CHECK(
      not domain::CoordinateMaps::TimeDependent::Rotation<Dim>{}.is_identity());
}

void compare_rotation_maps() {
  INFO("Compare rotation maps");
  double t{-1.0};
  const double final_time{4.0};
  constexpr size_t deriv_order{3};

  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using QuatFoT =
      domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;

  const std::string quat_name{"QuaternionRotation"};
  const std::string angle_name{"RotationAngle"};
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  std::array<DataVector, deriv_order + 1> init_func{};

  // angle(t) = t^2
  // omega(t) = 2 * t
  // dtomega(t) = 2
  const double angle = 1.0;
  const double omega = -2.0;
  const double dtomega = 2.0;
  const double d2tomega = 0.0;
  init_func = {{{angle}, {omega}, {dtomega}, {d2tomega}}};
  f_of_t_list[angle_name] =
      std::make_unique<Polynomial>(t, init_func, final_time);

  // Axis of rotation nhat = (0.0, 0.0, 1.0)
  DataVector axis{{0.0, 0.0, 1.0}};
  init_func = {axis * angle, axis * omega, axis * dtomega, axis * d2tomega};
  // initial quaternion is (cos(angle/2), nhat*sin(angle/2))
  const std::array<DataVector, 1> init_quat{
      DataVector{{cos(angle / 2.0), axis[0] * sin(angle / 2.0),
                  axis[1] * sin(angle / 2.0), axis[2] * sin(angle / 2.0)}}};
  f_of_t_list[quat_name] =
      std::make_unique<QuatFoT>(t, init_quat, init_func, final_time);

  using RotationMap2D = domain::CoordinateMaps::TimeDependent::Rotation<2>;
  using IdentityMap = domain::CoordinateMaps::Identity<1>;
  using RotationMap3D = domain::CoordinateMaps::TimeDependent::Rotation<3>;
  using ProductRotationMap2D =
      domain::CoordinateMaps::TimeDependent::ProductOf2Maps<RotationMap2D,
                                                            IdentityMap>;

  const RotationMap2D rotation_map_2d{angle_name};
  const IdentityMap identity_map{};
  const ProductRotationMap2D rotation_map_product{rotation_map_2d,
                                                  identity_map};
  const RotationMap3D rotation_map_3d{quat_name};

  const domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                              ProductRotationMap2D>
      rotation_map_from_2d{rotation_map_product};
  const domain::CoordinateMap<Frame::Grid, Frame::Inertial, RotationMap3D>
      rotation_map_from_3d{rotation_map_3d};

  Approx custom_approx = Approx::custom().epsilon(5.0e-13).scale(1.0);
  check_if_maps_are_equal(rotation_map_from_2d, rotation_map_from_3d,
                          final_time, f_of_t_list, custom_approx);
}

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.TimeDependent.Rotation",
                  "[Domain][Unit]") {
  test_rotation_map<2>();
  test_rotation_map<3>();

  // The 3D map should be equal to the identity * 2D map if the rotation is
  // about the z axis
  compare_rotation_maps();
}
}  // namespace
