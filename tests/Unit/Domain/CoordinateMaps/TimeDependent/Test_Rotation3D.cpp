// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <optional>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation3D.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"

namespace {
std::array<double, 3> mapped_point(const std::array<double, 3> ipt,
                                   const double t) {
  const double c = cos(square(t));  // rotation angle = t^2
  const double s = sin(square(t));  // rotation angle = t^2
  const double vx = 1 / sqrt(3);
  const double vy = -vx;
  const double vz = vx;
  // rotate about (1, -1, 1) /root(3)
  return {{(c + vx * vx * (1 - c)) * ipt[0] +
               (vx * vy * (1 - c) - vz * s) * ipt[1] +
               (vx * vz * (1 - c) + vy * s) * ipt[2],
           (vy * vx * (1 - c) + vz * s) * ipt[0] +
               (c + vy * vy * (1 - c)) * ipt[1] +
               (vy * vz * (1 - c) - vx * s) * ipt[2],
           (vz * vx * (1 - c) - vy * s) * ipt[0] +
               (vz * vy * (1 - c) + vx * s) * ipt[1] +
               (c + vz * vz * (1 - c)) * ipt[2]}};
}

std::array<double, 3> frame_vel(const std::array<double, 3> ipt,
                                const double t) {
  const double c = cos(square(t));  // rotation angle = t^2
  const double s = sin(square(t));  // rotation angle = t^2
  const double omega = 2 * t;
  const double vx = 1 / sqrt(3);
  const double vy = -vx;
  const double vz = vx;
  // rotate about (1, -1, 1) /root(3)
  return {{((vz * vz - 1) * s * ipt[0] + (vx * vy * s - vz * c) * ipt[1] +
            (vx * vz * s + vy * c) * ipt[2]) *
               omega,
           ((vx * vy * s + vz * c) * ipt[0] + (vy * vy - 1) * s * ipt[1] +
            (vy * vz * s - vx * c) * ipt[2]) *
               omega,
           ((vx * vz * s - vy * c) * ipt[0] + (vy * vz * s + vx * c) * ipt[1] +
            (vz * vz - 1) * s * ipt[2]) *
               omega}};
}
}  // namespace

namespace domain {
SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.TimeDependent.Rotation3D",
                  "[Domain][Unit]") {
  double t{0.0};
  double dt{0.6};
  double final_time{4.0};
  constexpr size_t deriv_order{2};

  using QuatFoft =
      domain::FunctionsOfTime::QuaternionFunctionOfTime<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;

  const std::string f_of_t_name{"test_quaternion"};
  // rotation about (1,-1,1)/root(3) by angle theta = t^2
  const std::array<DataVector, deriv_order + 1> init_omega{
      DataVector{3, 0.0},
      DataVector{{2.0 / sqrt(3), -2.0 / sqrt(3), 2.0 / sqrt(3)}},
      DataVector{3, 0.0}};
  const std::array<DataVector, 1> init_quat{DataVector{{1.0, 0.0, 0.0, 0.0}}};

  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  f_of_t_list[f_of_t_name] =
      std::make_unique<QuatFoft>(t, init_quat, init_omega, final_time + dt);

  const CoordinateMaps::TimeDependent::Rotation3D rotation_map{f_of_t_name};
  const auto rotation_map_deserialized =
      serialize_and_deserialize(rotation_map);

  const std::array<double, 3> initial_unmapped_point{{8.1, 5.9, 2.3}};
  Approx custom_approx1 = Approx::custom().epsilon(3e-11).scale(1.0);

  while (t < final_time) {
    const std::array<double, 3> expected{
        mapped_point(initial_unmapped_point, t)};
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map(initial_unmapped_point, t, f_of_t_list), expected,
        custom_approx1);
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map.inverse(expected, t, f_of_t_list).value(),
        initial_unmapped_point, custom_approx1);
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map.frame_velocity(initial_unmapped_point, t, f_of_t_list),
        frame_vel(initial_unmapped_point, t), custom_approx1);

    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map_deserialized(initial_unmapped_point, t, f_of_t_list),
        expected, custom_approx1);
    CHECK_ITERABLE_CUSTOM_APPROX(
        rotation_map_deserialized.inverse(expected, t, f_of_t_list).value(),
        initial_unmapped_point, custom_approx1);
    CHECK_ITERABLE_CUSTOM_APPROX(rotation_map_deserialized.frame_velocity(
                                     initial_unmapped_point, t, f_of_t_list),
                                 frame_vel(initial_unmapped_point, t),
                                 custom_approx1);

    const auto jac{
        rotation_map.jacobian(initial_unmapped_point, t, f_of_t_list)};
    const auto inv_jac{
        rotation_map.inv_jacobian(initial_unmapped_point, t, f_of_t_list)};
    const double c = cos(square(t));  // rotation angle = t^2
    const double s = sin(square(t));  // rotation angle = t^2
    const double vx = 1 / sqrt(3);
    const double vy = -vx;
    const double vz = vx;
    // rotate about (1, -1, 1) /root(3)
    Approx custom_approx2 = Approx::custom().epsilon(1e-12).scale(1.0);

    CHECK(get<0, 0>(jac) == custom_approx2(c + vx * vx * (1 - c)));
    CHECK(get<0, 1>(jac) == custom_approx2(vx * vy * (1 - c) - vz * s));
    CHECK(get<0, 2>(jac) == custom_approx2(vx * vz * (1 - c) + vy * s));
    CHECK(get<1, 0>(jac) == custom_approx2(vy * vx * (1 - c) + vz * s));
    CHECK(get<1, 1>(jac) == custom_approx2(c + vy * vy * (1 - c)));
    CHECK(get<1, 2>(jac) == custom_approx2(vy * vz * (1 - c) - vx * s));
    CHECK(get<2, 0>(jac) == custom_approx2(vz * vx * (1 - c) - vy * s));
    CHECK(get<2, 1>(jac) == custom_approx2(vz * vy * (1 - c) + vx * s));
    CHECK(get<2, 2>(jac) == custom_approx2(c + vz * vz * (1 - c)));

    CHECK(get<0, 0>(inv_jac) == custom_approx2(c + vx * vx * (1 - c)));
    CHECK(get<0, 1>(inv_jac) == custom_approx2(vx * vy * (1 - c) + vz * s));
    CHECK(get<0, 2>(inv_jac) == custom_approx2(vx * vz * (1 - c) - vy * s));
    CHECK(get<1, 0>(inv_jac) == custom_approx2(vy * vx * (1 - c) - vz * s));
    CHECK(get<1, 1>(inv_jac) == custom_approx2(c + vy * vy * (1 - c)));
    CHECK(get<1, 2>(inv_jac) == custom_approx2(vy * vz * (1 - c) + vx * s));
    CHECK(get<2, 0>(inv_jac) == custom_approx2(vz * vx * (1 - c) + vy * s));
    CHECK(get<2, 1>(inv_jac) == custom_approx2(vz * vy * (1 - c) - vx * s));
    CHECK(get<2, 2>(inv_jac) == custom_approx2(c + vz * vz * (1 - c)));

    const auto jac_deserialized{rotation_map_deserialized.jacobian(
        initial_unmapped_point, t, f_of_t_list)};
    const auto inv_jac_deserialized{rotation_map_deserialized.inv_jacobian(
        initial_unmapped_point, t, f_of_t_list)};
    CHECK(get<0, 0>(jac_deserialized) == custom_approx2(c + vx * vx * (1 - c)));
    CHECK(get<0, 1>(jac_deserialized) ==
          custom_approx2(vx * vy * (1 - c) - vz * s));
    CHECK(get<0, 2>(jac_deserialized) ==
          custom_approx2(vx * vz * (1 - c) + vy * s));
    CHECK(get<1, 0>(jac_deserialized) ==
          custom_approx2(vy * vx * (1 - c) + vz * s));
    CHECK(get<1, 1>(jac_deserialized) == custom_approx2(c + vy * vy * (1 - c)));
    CHECK(get<1, 2>(jac_deserialized) ==
          custom_approx2(vy * vz * (1 - c) - vx * s));
    CHECK(get<2, 0>(jac_deserialized) ==
          custom_approx2(vz * vx * (1 - c) - vy * s));
    CHECK(get<2, 1>(jac_deserialized) ==
          custom_approx2(vz * vy * (1 - c) + vx * s));
    CHECK(get<2, 2>(jac_deserialized) == custom_approx2(c + vz * vz * (1 - c)));

    CHECK(get<0, 0>(inv_jac_deserialized) ==
          custom_approx2(c + vx * vx * (1 - c)));
    CHECK(get<0, 1>(inv_jac_deserialized) ==
          custom_approx2(vx * vy * (1 - c) + vz * s));
    CHECK(get<0, 2>(inv_jac_deserialized) ==
          custom_approx2(vx * vz * (1 - c) - vy * s));
    CHECK(get<1, 0>(inv_jac_deserialized) ==
          custom_approx2(vy * vx * (1 - c) - vz * s));
    CHECK(get<1, 1>(inv_jac_deserialized) ==
          custom_approx2(c + vy * vy * (1 - c)));
    CHECK(get<1, 2>(inv_jac_deserialized) ==
          custom_approx2(vy * vz * (1 - c) + vx * s));
    CHECK(get<2, 0>(inv_jac_deserialized) ==
          custom_approx2(vz * vx * (1 - c) + vy * s));
    CHECK(get<2, 1>(inv_jac_deserialized) ==
          custom_approx2(vz * vy * (1 - c) - vx * s));
    CHECK(get<2, 2>(inv_jac_deserialized) ==
          custom_approx2(c + vz * vz * (1 - c)));

    t += dt;
  }

  const std::string f_of_t_name_2{"Different name"};
  const CoordinateMaps::TimeDependent::Rotation3D rotation_map_2{f_of_t_name_2};
  // Check inequivalence operator
  CHECK_FALSE(rotation_map != rotation_map);
  CHECK_FALSE(rotation_map_deserialized != rotation_map_deserialized);
  CHECK(rotation_map != rotation_map_2);
  CHECK_FALSE(rotation_map == rotation_map_2);

  // Check serialization
  CHECK(rotation_map == rotation_map_deserialized);
  CHECK_FALSE(rotation_map != rotation_map_deserialized);

  test_coordinate_map_argument_types(rotation_map, initial_unmapped_point, t,
                                     f_of_t_list);
  CHECK(not CoordinateMaps::TimeDependent::Rotation3D{}.is_identity());
}
}  // namespace domain
