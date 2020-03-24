// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/ProductMapsTimeDep.hpp"
#include "Domain/CoordinateMaps/ProductMapsTimeDep.tpp"
#include "Domain/CoordinateMaps/Translation.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"

namespace domain {
namespace {
template <typename Map1, typename Map2>
void test_product_of_2_maps_time_dep(
    const CoordMapsTimeDependent::ProductOf2Maps<Map1, Map2>& map2d,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const double x_source_a, const double x_source_b, const double x_target_a,
    const double x_target_b, const double xi, const double x, const double eta,
    const double y, const double y_source_a, const double y_source_b,
    const double y_target_a, const double y_target_b,
    const std::array<double, 2>& expected_frame_velocity) noexcept {
  using AffineMap = CoordinateMaps::Affine;
  using TranslationMap = CoordMapsTimeDependent::Translation;
  static_assert(cpp17::is_same_v<Map1, AffineMap> or
                    cpp17::is_same_v<Map1, TranslationMap>,
                "Map1 must be either an affine map or a translation map");
  static_assert(cpp17::is_same_v<Map2, AffineMap> or
                    cpp17::is_same_v<Map2, TranslationMap>,
                "Map2 must be either an affine map or a translation map");

  const std::array<double, 2> point_source_a{{x_source_a, y_source_a}};
  const std::array<double, 2> point_source_b{{x_source_b, y_source_b}};
  const std::array<double, 2> point_xi{{xi, eta}};
  const std::array<double, 2> point_target_a{{x_target_a, y_target_a}};
  const std::array<double, 2> point_target_b{{x_target_b, y_target_b}};
  const std::array<double, 2> point_x{{x, y}};

  CHECK(map2d(point_source_a, time, functions_of_time) == point_target_a);
  CHECK(map2d(point_source_b, time, functions_of_time) == point_target_b);
  CHECK(map2d(point_xi, time, functions_of_time) == point_x);

  CHECK(map2d.inverse(point_target_a, time, functions_of_time).get() ==
        point_source_a);
  CHECK(map2d.inverse(point_target_b, time, functions_of_time).get() ==
        point_source_b);
  CHECK_ITERABLE_APPROX(map2d.inverse(point_x, time, functions_of_time).get(),
                        point_xi);

  const double inv_jacobian_00 =
      (x_source_b - x_source_a) / (x_target_b - x_target_a);
  const double inv_jacobian_11 =
      (y_source_b - y_source_a) / (y_target_b - y_target_a);
  const auto inv_jac_A =
      map2d.inv_jacobian(point_source_a, time, functions_of_time);
  const auto inv_jac_B =
      map2d.inv_jacobian(point_source_b, time, functions_of_time);
  const auto inv_jac_xi = map2d.inv_jacobian(point_xi, time, functions_of_time);

  const auto check_jac = [](const auto& jac, const auto& expected_jac_00,
                            const auto& expected_jac_11) noexcept {
    CHECK(get<0, 0>(jac) == expected_jac_00);
    CHECK(get<0, 1>(jac) == 0.0);
    CHECK(get<1, 0>(jac) == 0.0);
    CHECK(get<1, 1>(jac) == expected_jac_11);
  };

  check_jac(inv_jac_A, inv_jacobian_00, inv_jacobian_11);
  check_jac(inv_jac_B, inv_jacobian_00, inv_jacobian_11);
  check_jac(inv_jac_xi, inv_jacobian_00, inv_jacobian_11);

  const double jacobian_00 =
      (x_target_b - x_target_a) / (x_source_b - x_source_a);
  const double jacobian_11 =
      (y_target_b - y_target_a) / (y_source_b - y_source_a);
  const auto jac_A = map2d.jacobian(point_source_a, time, functions_of_time);
  const auto jac_B = map2d.jacobian(point_source_b, time, functions_of_time);
  const auto jac_xi = map2d.jacobian(point_xi, time, functions_of_time);

  check_jac(jac_A, jacobian_00, jacobian_11);
  check_jac(jac_B, jacobian_00, jacobian_11);
  check_jac(jac_xi, jacobian_00, jacobian_11);

  CHECK(map2d.frame_velocity(point_source_a, time, functions_of_time) ==
        expected_frame_velocity);
  CHECK(map2d.frame_velocity(point_source_b, time, functions_of_time) ==
        expected_frame_velocity);
  CHECK(map2d.frame_velocity(point_xi, time, functions_of_time) ==
        expected_frame_velocity);

  // Check Jacobians for DataVectors
  const Mesh<2> mesh{8, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto tensor_logical_coords = logical_coordinates(mesh);
  const std::array<DataVector, 2> logical_coords{
      {get<0>(tensor_logical_coords), get<1>(tensor_logical_coords)}};
  const auto volume_inv_jac =
      map2d.inv_jacobian(logical_coords, time, functions_of_time);
  const auto volume_jac =
      map2d.jacobian(logical_coords, time, functions_of_time);
  for (size_t i = 0; i < 2; ++i) {
    for (size_t j = 0; j < 2; ++j) {
      if (i == j) {
        CHECK(volume_inv_jac.get(i, j) ==
              DataVector(logical_coords[0].size(),
                         i == 0 ? inv_jacobian_00 : inv_jacobian_11));
        CHECK(volume_jac.get(i, j) ==
              DataVector(logical_coords[0].size(),
                         i == 0 ? jacobian_00 : jacobian_11));
      } else {
        CHECK(volume_inv_jac.get(i, j) ==
              DataVector(logical_coords[0].size(), 0.0));
        CHECK(volume_jac.get(i, j) ==
              DataVector(logical_coords[0].size(), 0.0));
      }
    }
  }

  // Check frame velocity with DataVectors
  std::array<DataVector, 2> expected_frame_velocity_datavector{
      {DataVector(logical_coords[0].size(), expected_frame_velocity[0]),
       DataVector(logical_coords[1].size(), expected_frame_velocity[1])}};

  CHECK(map2d.frame_velocity(logical_coords, time, functions_of_time) ==
        expected_frame_velocity_datavector);

  CHECK(map2d == map2d);
  CHECK_FALSE(map2d != map2d);
}

void test_product_of_2_maps_time_dep() noexcept {
  INFO("Product of two maps with time dependence");
  constexpr size_t deriv_order = 3;
  using affine_map = CoordinateMaps::Affine;
  using translation_map = CoordMapsTimeDependent::Translation;

  const std::string f_of_t_name{"translation"};
  const double time = 2.0;
  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0}, {-2.0}, {2.0}, {0.0}}};
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  std::unordered_map<std::string, FoftPtr> functions_of_time{};
  functions_of_time[f_of_t_name] = std::make_unique<Polynomial>(0.0, init_func);

  {
    // Test one time-dependent and one time-independent map case.
    const double x_source_a = -1.0;
    const double x_source_b = 1.0;
    const double x_target_a = -2.0;
    const double x_target_b = 2.0;

    const double xi = 0.5 * (x_source_a + x_source_b);
    const double x =
        x_target_b * (xi - x_source_a) / (x_source_b - x_source_a) +
        x_target_a * (x_source_b - xi) / (x_source_b - x_source_a);
    const double eta = 0.7;
    const double y = eta + functions_of_time.at(f_of_t_name)->func(time)[0][0];

    const double y_source_a = -1.0;
    const double y_source_b = 1.0;
    const double y_target_a =
        -1.0 + functions_of_time.at(f_of_t_name)->func(time)[0][0];
    const double y_target_b =
        1.0 + functions_of_time.at(f_of_t_name)->func(time)[0][0];

    affine_map affine_map_x(x_source_a, x_source_b, x_target_a, x_target_b);
    translation_map translation_map_y{f_of_t_name};

    using Map2d =
        CoordMapsTimeDependent::ProductOf2Maps<affine_map, translation_map>;
    Map2d map2d(affine_map_x, translation_map_y);

    test_product_of_2_maps_time_dep(
        map2d, time, functions_of_time, x_source_a, x_source_b, x_target_a,
        x_target_b, xi, x, eta, y, y_source_a, y_source_b, y_target_a,
        y_target_b,
        {{0.0, functions_of_time.at(f_of_t_name)->func_and_deriv(time)[1][0]}});
    test_product_of_2_maps_time_dep(
        serialize_and_deserialize(map2d), time, functions_of_time, x_source_a,
        x_source_b, x_target_a, x_target_b, xi, x, eta, y, y_source_a,
        y_source_b, y_target_a, y_target_b,
        {{0.0, functions_of_time.at(f_of_t_name)->func_and_deriv(time)[1][0]}});

    using Map2d_b =
        CoordMapsTimeDependent::ProductOf2Maps<translation_map, affine_map>;
    Map2d_b map2d_b(translation_map_y, affine_map_x);

    test_product_of_2_maps_time_dep(
        map2d_b, time, functions_of_time, y_source_a, y_source_b, y_target_a,
        y_target_b, eta, y, xi, x, x_source_a, x_source_b, x_target_a,
        x_target_b,
        {{functions_of_time.at(f_of_t_name)->func_and_deriv(time)[1][0], 0.0}});
    test_product_of_2_maps_time_dep(
        serialize_and_deserialize(map2d_b), time, functions_of_time, y_source_a,
        y_source_b, y_target_a, y_target_b, eta, y, xi, x, x_source_a,
        x_source_b, x_target_a, x_target_b,
        {{functions_of_time.at(f_of_t_name)->func_and_deriv(time)[1][0], 0.0}});
  }

  {
    using Map2d = CoordMapsTimeDependent::ProductOf2Maps<translation_map,
                                                         translation_map>;

    const std::string f_of_t_name_x{"translation_x"};
    const std::string f_of_t_name_y{"translation_y"};

    translation_map translation_map_x{f_of_t_name_x};
    translation_map translation_map_y{f_of_t_name_y};
    Map2d map2d(translation_map_x, translation_map_y);

    functions_of_time[f_of_t_name_x] = std::make_unique<Polynomial>(
        0.0,
        std::array<DataVector, deriv_order + 1>{{{1.0}, {-3.0}, {2.0}, {0.0}}});
    functions_of_time[f_of_t_name_y] = std::make_unique<Polynomial>(
        0.0,
        std::array<DataVector, deriv_order + 1>{{{1.0}, {2.4}, {2.0}, {0.0}}});

    const double x_source_a = -1.0;
    const double x_source_b = 1.0;
    const double x_target_a =
        -1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
    const double x_target_b =
        1.0 + +functions_of_time.at(f_of_t_name_x)->func(time)[0][0];

    const double xi = 0.5;
    const double x = xi + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
    const double eta = 0.7;
    const double y =
        eta + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];

    const double y_source_a = -1.0;
    const double y_source_b = 1.0;
    const double y_target_a =
        -1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
    const double y_target_b =
        1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];

    test_product_of_2_maps_time_dep(
        map2d, time, functions_of_time, x_source_a, x_source_b, x_target_a,
        x_target_b, xi, x, eta, y, y_source_a, y_source_b, y_target_a,
        y_target_b,
        {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
          functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0]}});
    test_product_of_2_maps_time_dep(
        serialize_and_deserialize(map2d), time, functions_of_time, x_source_a,
        x_source_b, x_target_a, x_target_b, xi, x, eta, y, y_source_a,
        y_source_b, y_target_a, y_target_b,
        {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
          functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0]}});
  }
}

template <typename Map1, typename Map2, typename Map3>
void test_product_of_3_maps_time_dep(
    const CoordMapsTimeDependent::ProductOf3Maps<Map1, Map2, Map3>& map3d,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time,
    const double x_source_a, const double x_source_b, const double x_target_a,
    const double x_target_b, const double xi, const double x, const double eta,
    const double y, const double zeta, const double z, const double y_source_a,
    const double y_source_b, const double y_target_a, const double y_target_b,
    const double z_source_a, const double z_source_b, const double z_target_a,
    const double z_target_b,
    const std::array<double, 3>& expected_frame_velocity) noexcept {
  using AffineMap = CoordinateMaps::Affine;
  using TranslationMap = CoordMapsTimeDependent::Translation;
  static_assert(cpp17::is_same_v<Map1, AffineMap> or
                    cpp17::is_same_v<Map1, TranslationMap>,
                "Map1 must be either an affine map or a translation map");
  static_assert(cpp17::is_same_v<Map2, AffineMap> or
                    cpp17::is_same_v<Map2, TranslationMap>,
                "Map2 must be either an affine map or a translation map");
  static_assert(cpp17::is_same_v<Map3, AffineMap> or
                    cpp17::is_same_v<Map3, TranslationMap>,
                "Map3 must be either an affine map or a translation map");

  const std::array<double, 3> point_source_a{
      {x_source_a, y_source_a, z_source_a}};
  const std::array<double, 3> point_source_b{
      {x_source_b, y_source_b, z_source_b}};
  const std::array<double, 3> point_xi{{xi, eta, zeta}};
  const std::array<double, 3> point_target_a{
      {x_target_a, y_target_a, z_target_a}};
  const std::array<double, 3> point_target_b{
      {x_target_b, y_target_b, z_target_b}};
  const std::array<double, 3> point_x{{x, y, z}};

  CHECK(map3d(point_source_a, time, functions_of_time) == point_target_a);
  CHECK(map3d(point_source_b, time, functions_of_time) == point_target_b);
  CHECK(map3d(point_xi, time, functions_of_time) == point_x);

  CHECK(map3d.inverse(point_target_a, time, functions_of_time).get() ==
        point_source_a);
  CHECK(map3d.inverse(point_target_b, time, functions_of_time).get() ==
        point_source_b);
  CHECK_ITERABLE_APPROX(map3d.inverse(point_x, time, functions_of_time).get(),
                        point_xi);

  const double inv_jacobian_00 =
      (x_source_b - x_source_a) / (x_target_b - x_target_a);
  const double inv_jacobian_11 =
      (y_source_b - y_source_a) / (y_target_b - y_target_a);
  const double inv_jacobian_22 =
      (z_source_b - z_source_a) / (z_target_b - z_target_a);
  const auto inv_jac_A =
      map3d.inv_jacobian(point_source_a, time, functions_of_time);
  const auto inv_jac_B =
      map3d.inv_jacobian(point_source_b, time, functions_of_time);
  const auto inv_jac_xi = map3d.inv_jacobian(point_xi, time, functions_of_time);

  const auto check_jac = [](const auto& jac, const auto& expected_jac_00,
                            const auto& expected_jac_11,
                            const auto& expected_jac_22) noexcept {
    CHECK(get<0, 0>(jac) == expected_jac_00);
    CHECK(get<1, 1>(jac) == expected_jac_11);
    CHECK(get<2, 2>(jac) == expected_jac_22);
    CHECK(get<0, 1>(jac) == 0.0);
    CHECK(get<0, 2>(jac) == 0.0);
    CHECK(get<1, 0>(jac) == 0.0);
    CHECK(get<2, 0>(jac) == 0.0);
    CHECK(get<1, 2>(jac) == 0.0);
    CHECK(get<2, 1>(jac) == 0.0);
  };

  check_jac(inv_jac_A, inv_jacobian_00, inv_jacobian_11, inv_jacobian_22);
  check_jac(inv_jac_B, inv_jacobian_00, inv_jacobian_11, inv_jacobian_22);
  check_jac(inv_jac_xi, inv_jacobian_00, inv_jacobian_11, inv_jacobian_22);

  const double jacobian_00 =
      (x_target_b - x_target_a) / (x_source_b - x_source_a);
  const double jacobian_11 =
      (y_target_b - y_target_a) / (y_source_b - y_source_a);
  const double jacobian_22 =
      (z_target_b - z_target_a) / (z_source_b - z_source_a);
  const auto jac_A = map3d.jacobian(point_source_a, time, functions_of_time);
  const auto jac_B = map3d.jacobian(point_source_b, time, functions_of_time);
  const auto jac_xi = map3d.jacobian(point_xi, time, functions_of_time);

  check_jac(jac_A, jacobian_00, jacobian_11, jacobian_22);
  check_jac(jac_B, jacobian_00, jacobian_11, jacobian_22);
  check_jac(jac_xi, jacobian_00, jacobian_11, jacobian_22);

  CHECK(map3d == map3d);
  CHECK_FALSE(map3d != map3d);

  CHECK(map3d.frame_velocity(point_source_a, time, functions_of_time) ==
        expected_frame_velocity);
  CHECK(map3d.frame_velocity(point_source_b, time, functions_of_time) ==
        expected_frame_velocity);
  CHECK(map3d.frame_velocity(point_xi, time, functions_of_time) ==
        expected_frame_velocity);

  // Check Jacobians for DataVectors
  const Mesh<3> mesh{8, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto tensor_logical_coords = logical_coordinates(mesh);
  const std::array<DataVector, 3> logical_coords{
      {get<0>(tensor_logical_coords), get<1>(tensor_logical_coords),
       get<2>(tensor_logical_coords)}};
  const auto volume_inv_jac =
      map3d.inv_jacobian(logical_coords, time, functions_of_time);
  const auto volume_jac =
      map3d.jacobian(logical_coords, time, functions_of_time);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      if (i == j) {
        CHECK(volume_inv_jac.get(i, j) ==
              DataVector(logical_coords[0].size(),
                         i == 0 ? inv_jacobian_00
                                : i == 1 ? inv_jacobian_11 : inv_jacobian_22));
        CHECK(volume_jac.get(i, j) ==
              DataVector(
                  logical_coords[0].size(),
                  i == 0 ? jacobian_00 : i == 1 ? jacobian_11 : jacobian_22));
      } else {
        CHECK(volume_inv_jac.get(i, j) ==
              DataVector(logical_coords[0].size(), 0.0));
        CHECK(volume_jac.get(i, j) ==
              DataVector(logical_coords[0].size(), 0.0));
      }
    }
  }

  // Check frame velocity with DataVectors
  std::array<DataVector, 3> expected_frame_velocity_datavector{
      {DataVector(logical_coords[0].size(), expected_frame_velocity[0]),
       DataVector(logical_coords[1].size(), expected_frame_velocity[1]),
       DataVector(logical_coords[2].size(), expected_frame_velocity[2])}};

  CHECK(map3d.frame_velocity(logical_coords, time, functions_of_time) ==
        expected_frame_velocity_datavector);

  CHECK(map3d == map3d);
  CHECK_FALSE(map3d != map3d);
}

void test_product_of_3_maps() noexcept {
  INFO("Product of 3 maps");
  constexpr size_t deriv_order = 3;
  using affine_map = CoordinateMaps::Affine;
  using translation_map = CoordMapsTimeDependent::Translation;

  const std::string f_of_t_name_x{"translation_x"};
  const std::string f_of_t_name_y{"translation_y"};
  const std::string f_of_t_name_z{"translation_z"};

  const double time = 2.0;
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  std::unordered_map<std::string, FoftPtr> functions_of_time{};
  functions_of_time[f_of_t_name_x] = std::make_unique<Polynomial>(
      0.0,
      std::array<DataVector, deriv_order + 1>{{{1.0}, {-3.0}, {1.3}, {0.0}}});
  functions_of_time[f_of_t_name_y] = std::make_unique<Polynomial>(
      0.0,
      std::array<DataVector, deriv_order + 1>{{{1.0}, {2.4}, {2.0}, {0.0}}});
  functions_of_time[f_of_t_name_z] = std::make_unique<Polynomial>(
      0.0,
      std::array<DataVector, deriv_order + 1>{{{1.0}, {0.4}, {4.0}, {0.0}}});

  {
    // Test one time-dependent and two time-independent map case.
    const double x_source_a = -1.0;
    const double x_source_b = 1.0;
    const double x_target_a = -2.0;
    const double x_target_b = 2.0;
    affine_map affine_map_x(x_source_a, x_source_b, x_target_a, x_target_b);

    const double y_source_a = -2.0;
    const double y_source_b = 3.0;
    const double y_target_a = 5.0;
    const double y_target_b = -2.0;
    affine_map affine_map_y(y_source_a, y_source_b, y_target_a, y_target_b);

    const double xi = 0.5 * (x_source_a + x_source_b);
    const double x =
        x_target_b * (xi - x_source_a) / (x_source_b - x_source_a) +
        x_target_a * (x_source_b - xi) / (x_source_b - x_source_a);
    const double eta = 0.5 * (y_source_a + y_source_b);
    const double y =
        y_target_b * (eta - y_source_a) / (y_source_b - y_source_a) +
        y_target_a * (y_source_b - eta) / (y_source_b - y_source_a);

    {
      INFO("Check with z-direction map time-dependent");
      const double z_source_a = -1.0;
      const double z_source_b = 1.0;
      const double z_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];
      const double z_target_b =
          1.0 + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];

      const double zeta = 0.7;
      const double z =
          zeta + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];

      translation_map translation_map_z{f_of_t_name_z};
      using Map3d =
          CoordMapsTimeDependent::ProductOf3Maps<affine_map, affine_map,
                                                 translation_map>;
      Map3d map3d{affine_map_x, affine_map_y, translation_map_z};

      test_product_of_3_maps_time_dep(
          map3d, time, functions_of_time, x_source_a, x_source_b, x_target_a,
          x_target_b, xi, x, eta, y, zeta, z, y_source_a, y_source_b,
          y_target_a, y_target_b, z_source_a, z_source_b, z_target_a,
          z_target_b,
          {{0.0, 0.0,
            functions_of_time.at(f_of_t_name_z)->func_and_deriv(time)[1][0]}});
      test_product_of_3_maps_time_dep(
          serialize_and_deserialize(map3d), time, functions_of_time, x_source_a,
          x_source_b, x_target_a, x_target_b, xi, x, eta, y, zeta, z,
          y_source_a, y_source_b, y_target_a, y_target_b, z_source_a,
          z_source_b, z_target_a, z_target_b,
          {{0.0, 0.0,
            functions_of_time.at(f_of_t_name_z)->func_and_deriv(time)[1][0]}});
    }
    {
      INFO("Check with y-direction map time-dependent");
      const double z_source_a = -1.0;
      const double z_source_b = 1.0;
      const double z_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
      const double z_target_b =
          1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];

      const double zeta = 0.7;
      const double z =
          zeta + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];

      translation_map translation_map_z{f_of_t_name_y};
      using Map3d =
          CoordMapsTimeDependent::ProductOf3Maps<affine_map, translation_map,
                                                 affine_map>;
      Map3d map3d{affine_map_x, translation_map_z, affine_map_y};

      test_product_of_3_maps_time_dep(
          map3d, time, functions_of_time, x_source_a, x_source_b, x_target_a,
          x_target_b, xi, x, zeta, z, eta, y, z_source_a, z_source_b,
          z_target_a, z_target_b, y_source_a, y_source_b, y_target_a,
          y_target_b,
          {{0.0,
            functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0],
            0.0}});
      test_product_of_3_maps_time_dep(
          serialize_and_deserialize(map3d), time, functions_of_time, x_source_a,
          x_source_b, x_target_a, x_target_b, xi, x, zeta, z, eta, y,
          z_source_a, z_source_b, z_target_a, z_target_b, y_source_a,
          y_source_b, y_target_a, y_target_b,
          {{0.0,
            functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0],
            0.0}});
    }
    {
      INFO("Check with x-direction map time-dependent");
      const double z_source_a = -1.0;
      const double z_source_b = 1.0;
      const double z_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
      const double z_target_b =
          1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];

      const double zeta = 0.7;
      const double z =
          zeta + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];

      translation_map translation_map_z{f_of_t_name_x};
      using Map3d =
          CoordMapsTimeDependent::ProductOf3Maps<translation_map, affine_map,
                                                 affine_map>;
      Map3d map3d{translation_map_z, affine_map_y, affine_map_x};

      test_product_of_3_maps_time_dep(
          map3d, time, functions_of_time, z_source_a, z_source_b, z_target_a,
          z_target_b, zeta, z, eta, y, xi, x, y_source_a, y_source_b,
          y_target_a, y_target_b, x_source_a, x_source_b, x_target_a,
          x_target_b,
          {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
            0.0, 0.0}});
      test_product_of_3_maps_time_dep(
          serialize_and_deserialize(map3d), time, functions_of_time, z_source_a,
          z_source_b, z_target_a, z_target_b, zeta, z, eta, y, xi, x,
          y_source_a, y_source_b, y_target_a, y_target_b, x_source_a,
          x_source_b, x_target_a, x_target_b,
          {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
            0.0, 0.0}});
    }
  }

  {
    // Test two time-dependent and one time-independent map case.
    const double x_source_a = -1.0;
    const double x_source_b = 1.0;
    const double x_target_a = -2.0;
    const double x_target_b = 2.0;
    affine_map affine_map_x(x_source_a, x_source_b, x_target_a, x_target_b);

    const double xi = 0.5 * (x_source_a + x_source_b);
    const double x =
        x_target_b * (xi - x_source_a) / (x_source_b - x_source_a) +
        x_target_a * (x_source_b - xi) / (x_source_b - x_source_a);
    {
      INFO("Check with x-direction map time-independent");
      const double y_source_a = -1.0;
      const double y_source_b = 1.0;
      const double y_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
      const double y_target_b =
          1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
      const double z_source_a = -1.0;
      const double z_source_b = 1.0;
      const double z_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];
      const double z_target_b =
          1.0 + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];

      const double eta = 0.7;
      const double y =
          eta + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
      const double zeta = 0.7;
      const double z =
          zeta + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];

      translation_map translation_map_y{f_of_t_name_y};
      translation_map translation_map_z{f_of_t_name_z};
      using Map3d =
          CoordMapsTimeDependent::ProductOf3Maps<affine_map, translation_map,
                                                 translation_map>;
      Map3d map3d{affine_map_x, translation_map_y, translation_map_z};

      test_product_of_3_maps_time_dep(
          map3d, time, functions_of_time, x_source_a, x_source_b, x_target_a,
          x_target_b, xi, x, eta, y, zeta, z, y_source_a, y_source_b,
          y_target_a, y_target_b, z_source_a, z_source_b, z_target_a,
          z_target_b,
          {{0.0,
            functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0],
            functions_of_time.at(f_of_t_name_z)->func_and_deriv(time)[1][0]}});
      test_product_of_3_maps_time_dep(
          serialize_and_deserialize(map3d), time, functions_of_time, x_source_a,
          x_source_b, x_target_a, x_target_b, xi, x, eta, y, zeta, z,
          y_source_a, y_source_b, y_target_a, y_target_b, z_source_a,
          z_source_b, z_target_a, z_target_b,
          {{0.0,
            functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0],
            functions_of_time.at(f_of_t_name_z)->func_and_deriv(time)[1][0]}});
    }
    {
      INFO("Check with y-direction map time-independent");
      const double y_source_a = -1.0;
      const double y_source_b = 1.0;
      const double y_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
      const double y_target_b =
          1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
      const double z_source_a = -1.0;
      const double z_source_b = 1.0;
      const double z_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];
      const double z_target_b =
          1.0 + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];

      const double eta = 0.7;
      const double y =
          eta + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
      const double zeta = 0.7;
      const double z =
          zeta + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];

      translation_map translation_map_y{f_of_t_name_x};
      translation_map translation_map_z{f_of_t_name_z};
      using Map3d =
          CoordMapsTimeDependent::ProductOf3Maps<translation_map, affine_map,
                                                 translation_map>;
      Map3d map3d{translation_map_y, affine_map_x, translation_map_z};

      test_product_of_3_maps_time_dep(
          map3d, time, functions_of_time, y_source_a, y_source_b, y_target_a,
          y_target_b, eta, y, xi, x, zeta, z, x_source_a, x_source_b,
          x_target_a, x_target_b, z_source_a, z_source_b, z_target_a,
          z_target_b,
          {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
            0.0,
            functions_of_time.at(f_of_t_name_z)->func_and_deriv(time)[1][0]}});
      test_product_of_3_maps_time_dep(
          serialize_and_deserialize(map3d), time, functions_of_time, y_source_a,
          y_source_b, y_target_a, y_target_b, eta, y, xi, x, zeta, z,
          x_source_a, x_source_b, x_target_a, x_target_b, z_source_a,
          z_source_b, z_target_a, z_target_b,
          {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
            0.0,
            functions_of_time.at(f_of_t_name_z)->func_and_deriv(time)[1][0]}});
    }
    {
      INFO("Check with z-direction map time-independent");
      const double y_source_a = -1.0;
      const double y_source_b = 1.0;
      const double y_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
      const double y_target_b =
          1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
      const double z_source_a = -1.0;
      const double z_source_b = 1.0;
      const double z_target_a =
          -1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
      const double z_target_b =
          1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];

      const double eta = 0.7;
      const double y =
          eta + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
      const double zeta = 0.7;
      const double z =
          zeta + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];

      translation_map translation_map_y{f_of_t_name_y};
      translation_map translation_map_z{f_of_t_name_x};
      using Map3d =
          CoordMapsTimeDependent::ProductOf3Maps<translation_map,
                                                 translation_map, affine_map>;
      Map3d map3d{translation_map_z, translation_map_y, affine_map_x};

      test_product_of_3_maps_time_dep(
          map3d, time, functions_of_time, z_source_a, z_source_b, z_target_a,
          z_target_b, zeta, z, eta, y, xi, x, y_source_a, y_source_b,
          y_target_a, y_target_b, x_source_a, x_source_b, x_target_a,
          x_target_b,
          {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
            functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0],
            0.0}});
      test_product_of_3_maps_time_dep(
          serialize_and_deserialize(map3d), time, functions_of_time, z_source_a,
          z_source_b, z_target_a, z_target_b, zeta, z, eta, y, xi, x,
          y_source_a, y_source_b, y_target_a, y_target_b, x_source_a,
          x_source_b, x_target_a, x_target_b,
          {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
            functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0],
            0.0}});
    }
  }
  {
    INFO("Check only time-dependent maps");
    const double x_source_a = -1.0;
    const double x_source_b = 1.0;
    const double x_target_a =
        -1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
    const double x_target_b =
        1.0 + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
    const double y_source_a = -1.0;
    const double y_source_b = 1.0;
    const double y_target_a =
        -1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
    const double y_target_b =
        1.0 + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
    const double z_source_a = -1.0;
    const double z_source_b = 1.0;
    const double z_target_a =
        -1.0 + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];
    const double z_target_b =
        1.0 + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];

    const double xi = 0.3;
    const double x = xi + functions_of_time.at(f_of_t_name_x)->func(time)[0][0];
    const double eta = 0.7;
    const double y =
        eta + functions_of_time.at(f_of_t_name_y)->func(time)[0][0];
    const double zeta = 0.7;
    const double z =
        zeta + functions_of_time.at(f_of_t_name_z)->func(time)[0][0];

    translation_map translation_map_x{f_of_t_name_x};
    translation_map translation_map_y{f_of_t_name_y};
    translation_map translation_map_z{f_of_t_name_z};
    using Map3d =
        CoordMapsTimeDependent::ProductOf3Maps<translation_map, translation_map,
                                               translation_map>;
    Map3d map3d{translation_map_x, translation_map_y, translation_map_z};

    test_product_of_3_maps_time_dep(
        map3d, time, functions_of_time, x_source_a, x_source_b, x_target_a,
        x_target_b, xi, x, eta, y, zeta, z, y_source_a, y_source_b, y_target_a,
        y_target_b, z_source_a, z_source_b, z_target_a, z_target_b,
        {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
          functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0],
          functions_of_time.at(f_of_t_name_z)->func_and_deriv(time)[1][0]}});
    test_product_of_3_maps_time_dep(
        serialize_and_deserialize(map3d), time, functions_of_time, x_source_a,
        x_source_b, x_target_a, x_target_b, xi, x, eta, y, zeta, z, y_source_a,
        y_source_b, y_target_a, y_target_b, z_source_a, z_source_b, z_target_a,
        z_target_b,
        {{functions_of_time.at(f_of_t_name_x)->func_and_deriv(time)[1][0],
          functions_of_time.at(f_of_t_name_y)->func_and_deriv(time)[1][0],
          functions_of_time.at(f_of_t_name_z)->func_and_deriv(time)[1][0]}});
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordMapsTimeDependent.ProductMaps",
                  "[Domain][Unit]") {
  test_product_of_2_maps_time_dep();
  test_product_of_3_maps();
}
}  // namespace domain
