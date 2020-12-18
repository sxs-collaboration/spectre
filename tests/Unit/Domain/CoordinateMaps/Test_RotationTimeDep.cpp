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

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Rotation.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"

class DataVector;
namespace domain::FunctionsOfTime {
class FunctionOfTime;
}  // namespace domain::FunctionsOfTime

namespace {
std::array<double, 2> expected_mapped_point(
    const std::array<double, 2> initial_unmapped_point,
    const double time) noexcept {
  const double rotation_angle = square(time);
  return {{initial_unmapped_point[0] * cos(rotation_angle) -
               initial_unmapped_point[1] * sin(rotation_angle),
           initial_unmapped_point[0] * sin(rotation_angle) +
               initial_unmapped_point[1] * cos(rotation_angle)}};
}

std::array<double, 2> expected_frame_velocity(
    const std::array<double, 2> initial_unmapped_point,
    const double time) noexcept {
  const double rotation_angle = square(time);
  const double angular_velocity = 2.0 * time;
  return {
      {initial_unmapped_point[0] * -sin(rotation_angle) * angular_velocity -
           initial_unmapped_point[1] * cos(rotation_angle) * angular_velocity,
       initial_unmapped_point[0] * cos(rotation_angle) * angular_velocity +
           initial_unmapped_point[1] * -sin(rotation_angle) *
               angular_velocity}};
}
}  // namespace

namespace domain {
SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.RotationTimeDep",
                  "[Domain][Unit]") {
  double t{-1.0};
  const double dt{0.6};
  const double final_time{4.0};
  constexpr size_t deriv_order{3};
  constexpr size_t spatial_dim{2};

  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;

  const std::string f_of_t_name{"rotation_angle"};
  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0}, {-2.0}, {2.0}, {0.0}}};
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  f_of_t_list[f_of_t_name] =
      std::make_unique<Polynomial>(t, init_func, final_time + dt);

  const CoordinateMaps::TimeDependent::Rotation<spatial_dim> rotation_map{
      f_of_t_name};
  const auto rotation_map_deserialized =
      serialize_and_deserialize(rotation_map);

  const std::array<double, 2> initial_unmapped_point{{3.2, 4.5}};

  while (t < final_time) {
    const std::array<double, 2> expected{
        expected_mapped_point(initial_unmapped_point, t)};
    CHECK_ITERABLE_APPROX(rotation_map(initial_unmapped_point, t, f_of_t_list),
                          expected);
    CHECK_ITERABLE_APPROX(
        rotation_map.inverse(expected, t, f_of_t_list).value(),
        initial_unmapped_point);
    CHECK_ITERABLE_APPROX(
        rotation_map.frame_velocity(initial_unmapped_point, t, f_of_t_list),
        expected_frame_velocity(initial_unmapped_point, t));

    CHECK_ITERABLE_APPROX(
        rotation_map_deserialized(initial_unmapped_point, t, f_of_t_list),
        expected);
    CHECK_ITERABLE_APPROX(
        rotation_map_deserialized.inverse(expected, t, f_of_t_list).value(),
        initial_unmapped_point);
    CHECK_ITERABLE_APPROX(rotation_map_deserialized.frame_velocity(
                              initial_unmapped_point, t, f_of_t_list),
                          expected_frame_velocity(initial_unmapped_point, t));

    const auto jac{
        rotation_map.jacobian(initial_unmapped_point, t, f_of_t_list)};
    const auto inv_jac{
        rotation_map.inv_jacobian(initial_unmapped_point, t, f_of_t_list)};
    const double cos_t_squared{cos(square(t))};
    const double sin_t_squared{sin(square(t))};

    CHECK(get<0, 0>(jac) == approx(cos_t_squared));
    CHECK(get<0, 1>(jac) == approx(-sin_t_squared));
    CHECK(get<1, 0>(jac) == approx(sin_t_squared));
    CHECK(get<1, 1>(jac) == approx(cos_t_squared));

    CHECK(get<0, 0>(inv_jac) == approx(cos_t_squared));
    CHECK(get<0, 1>(inv_jac) == approx(sin_t_squared));
    CHECK(get<1, 0>(inv_jac) == approx(-sin_t_squared));
    CHECK(get<1, 1>(inv_jac) == approx(cos_t_squared));

    const auto jac_deserialized{rotation_map_deserialized.jacobian(
        initial_unmapped_point, t, f_of_t_list)};
    const auto inv_jac_deserialized{rotation_map_deserialized.inv_jacobian(
        initial_unmapped_point, t, f_of_t_list)};
    CHECK(get<0, 0>(jac_deserialized) == approx(cos_t_squared));
    CHECK(get<0, 1>(jac_deserialized) == approx(-sin_t_squared));
    CHECK(get<1, 0>(jac_deserialized) == approx(sin_t_squared));
    CHECK(get<1, 1>(jac_deserialized) == approx(cos_t_squared));

    CHECK(get<0, 0>(inv_jac_deserialized) == approx(cos_t_squared));
    CHECK(get<0, 1>(inv_jac_deserialized) == approx(sin_t_squared));
    CHECK(get<1, 0>(inv_jac_deserialized) == approx(-sin_t_squared));
    CHECK(get<1, 1>(inv_jac_deserialized) == approx(cos_t_squared));

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
      not CoordinateMaps::TimeDependent::Rotation<spatial_dim>{}.is_identity());
}
}  // namespace domain
