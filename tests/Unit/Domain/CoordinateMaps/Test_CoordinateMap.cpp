// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>
#include <optional>
#include <pup.h>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/SpecialMobius.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace domain {
namespace {
template <typename Map1, typename Map2, typename DataType, size_t Dim>
auto compose_jacobians(const Map1& map1, const Map2& map2,
                       const std::array<DataType, Dim>& point) {
  const auto jac1 = map1.jacobian(point);
  const auto jac2 = map2.jacobian(map1(point));

  auto result =
      make_with_value<Jacobian<DataType, Dim, Frame::Logical, Frame::Grid>>(
          point[0], 0.);
  for (size_t target = 0; target < Dim; ++target) {
    for (size_t source = 0; source < Dim; ++source) {
      for (size_t dummy = 0; dummy < Dim; ++dummy) {
        result.get(target, source) +=
            jac2.get(target, dummy) * jac1.get(dummy, source);
      }
    }
  }
  return result;
}

template <typename Map1, typename Map2, typename DataType, size_t Dim>
auto compose_inv_jacobians(const Map1& map1, const Map2& map2,
                           const std::array<DataType, Dim>& point) {
  const auto inv_jac1 = map1.inv_jacobian(point);
  const auto inv_jac2 = map2.inv_jacobian(map1(point));

  auto result = make_with_value<
      InverseJacobian<DataType, Dim, Frame::Logical, Frame::Grid>>(point[0],
                                                                   0.);
  for (size_t target = 0; target < Dim; ++target) {
    for (size_t source = 0; source < Dim; ++source) {
      for (size_t dummy = 0; dummy < Dim; ++dummy) {
        result.get(source, target) +=
            inv_jac1.get(source, dummy) * inv_jac2.get(dummy, target);
      }
    }
  }
  return result;
}

void test_single_coordinate_map() {
  INFO("Single coordinate map");
  using affine_map1d = CoordinateMaps::Affine;

  const auto affine1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map1d{-1.0, 1.0, 2.0, 8.0});
  const auto affine1d_base_inertial =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          affine_map1d{-1.0, 1.0, 2.0, 8.0});
  const auto affine1d_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(
          affine_map1d{-1.0, 1.0, 2.0, 8.0});
  const auto affine1d_base_from_inertial =
      affine1d_base_inertial->get_to_grid_frame();
  const auto first_affine1d = affine_map1d{-1.0, 1.0, 2.0, 8.0};

  CHECK(affine1d == *affine1d_base);
  CHECK(*affine1d_base == affine1d);

  std::array<std::array<double, 1>, 4> coords1d{
      {{{0.1}}, {{-8.2}}, {{5.7}}, {{2.9}}}};

  const auto check_map_ptr = [](const auto& first_map, const auto& map,
                                const auto& map_base, const auto& local_coord,
                                const auto& local_source_points) {
    // coord is a std::array, and tuple_size<array> gets the size
    constexpr size_t dim =
        std::tuple_size<std::decay_t<decltype(local_coord)>>::value;
    CHECK((make_array<double, dim>((*map_base)(local_source_points))) ==
          first_map(local_coord));
    CHECK((make_array<double, dim>(
              map_base->inverse(tnsr::I<double, dim, Frame::Grid>{local_coord})
                  .value())) == first_map.inverse(local_coord).value());

    const auto expected_jac_no_frame = first_map.jacobian(local_coord);
    Jacobian<double, dim, Frame::Logical, Frame::Grid> local_expected_jac{};
    REQUIRE(expected_jac_no_frame.size() == local_expected_jac.size());
    for (size_t i = 0; i < local_expected_jac.size(); ++i) {
      local_expected_jac[i] = expected_jac_no_frame[i];
    }
    CHECK_ITERABLE_APPROX(map_base->jacobian(local_source_points),
                          local_expected_jac);

    const auto expected_inv_jac_no_frame = first_map.inv_jacobian(local_coord);
    InverseJacobian<double, dim, Frame::Logical, Frame::Grid>
        local_expected_inv_jac{};
    REQUIRE(expected_inv_jac_no_frame.size() == local_expected_inv_jac.size());
    for (size_t i = 0; i < local_expected_jac.size(); ++i) {
      local_expected_inv_jac[i] = expected_inv_jac_no_frame[i];
    }
    CHECK_ITERABLE_APPROX(map_base->inv_jacobian(local_source_points),
                          local_expected_inv_jac);

    const auto coords_jacs_velocity =
        map_base->coords_frame_velocity_jacobians(local_source_points);
    CHECK(std::get<0>(coords_jacs_velocity) == map(local_source_points));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          map.inv_jacobian(local_source_points));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          map.jacobian(local_source_points));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<double, dim, Frame::Grid>{0.0});
  };

  for (const auto& coord : coords1d) {
    const tnsr::I<double, 1, Frame::Logical> source_points{{{coord[0]}}};
    CHECK((make_array<double, 1>(affine1d(tnsr::I<double, 1, Frame::Logical>{
              {{coord[0]}}}))) == first_affine1d(coord));
    CHECK((make_array<double, 1>(
              affine1d.inverse(tnsr::I<double, 1, Frame::Grid>{{{coord[0]}}})
                  .value())) == first_affine1d.inverse(coord).value());

    CHECK(affine1d.jacobian(source_points).get(0, 0) ==
          first_affine1d.jacobian(coord).get(0, 0));

    CHECK(affine1d.inv_jacobian(source_points).get(0, 0) ==
          first_affine1d.inv_jacobian(coord).get(0, 0));

    check_map_ptr(first_affine1d, affine1d, affine1d_base, coord,
                  source_points);
    check_map_ptr(first_affine1d, affine1d, affine1d_base_from_inertial, coord,
                  source_points);
  }

  CHECK_FALSE(affine1d.is_identity());
  CHECK_FALSE(affine1d_base->is_identity());
  CHECK_FALSE(affine1d_base_from_inertial->is_identity());

  CHECK_FALSE(affine1d.inv_jacobian_is_time_dependent());
  CHECK_FALSE(affine1d.jacobian_is_time_dependent());
  CHECK_FALSE(affine1d_base->jacobian_is_time_dependent());
  CHECK_FALSE(affine1d_base->inv_jacobian_is_time_dependent());

  using rotate2d = CoordinateMaps::Rotation<2>;

  const auto first_rotated2d = rotate2d{M_PI_4};
  const auto rotated2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(first_rotated2d);
  const auto rotated2d_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(first_rotated2d);
  const auto rotated2d_base_inertial =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(first_rotated2d);
  const auto rotated2d_base_from_inertial =
      rotated2d_base_inertial->get_to_grid_frame();

  CHECK(rotated2d == *rotated2d_base);
  CHECK(*rotated2d_base == rotated2d);

  std::array<std::array<double, 2>, 4> coords2d{
      {{{0.1, 2.8}}, {{-8.2, 2.8}}, {{5.7, -4.9}}, {{2.9, 3.4}}}};

  for (const auto& coord : coords2d) {
    const tnsr::I<double, 2, Frame::Logical> source_points{coord};

    CHECK((make_array<double, 2>(rotated2d(source_points))) ==
          first_rotated2d(coord));
    CHECK((make_array<double, 2>(
              rotated2d.inverse(tnsr::I<double, 2, Frame::Grid>{coord})
                  .value())) == first_rotated2d.inverse(coord).value());

    const auto expected_jac_inertial = first_rotated2d.jacobian(coord);
    Jacobian<double, 2, Frame::Logical, Frame::Grid> expected_jac_grid{};
    REQUIRE(decltype(expected_jac_inertial)::size() ==
            decltype(expected_jac_grid)::size());
    for (size_t i = 0; i < decltype(expected_jac_inertial)::size(); ++i) {
      expected_jac_grid[i] = expected_jac_inertial[i];
    }
    CHECK(rotated2d.jacobian(source_points) == expected_jac_grid);

    const auto expected_inv_jac_inertial = first_rotated2d.inv_jacobian(coord);
    InverseJacobian<double, 2, Frame::Logical, Frame::Grid>
        expected_inv_jac_grid{};
    REQUIRE(decltype(expected_inv_jac_inertial)::size() ==
            decltype(expected_inv_jac_grid)::size());
    for (size_t i = 0; i < decltype(expected_inv_jac_inertial)::size(); ++i) {
      expected_inv_jac_grid[i] = expected_inv_jac_inertial[i];
    }
    CHECK(rotated2d.inv_jacobian(source_points) == expected_inv_jac_grid);

    check_map_ptr(first_rotated2d, rotated2d, rotated2d_base, coord,
                  source_points);
    check_map_ptr(first_rotated2d, rotated2d, rotated2d_base_from_inertial,
                  coord, source_points);
  }

  CHECK_FALSE(rotated2d.is_identity());
  CHECK_FALSE(rotated2d_base->is_identity());
  CHECK_FALSE(rotated2d_base_from_inertial->is_identity());

  CHECK_FALSE(rotated2d.inv_jacobian_is_time_dependent());
  CHECK_FALSE(rotated2d.jacobian_is_time_dependent());
  CHECK_FALSE(rotated2d_base->jacobian_is_time_dependent());
  CHECK_FALSE(rotated2d_base->inv_jacobian_is_time_dependent());

  using rotate3d = CoordinateMaps::Rotation<3>;

  const auto first_rotated3d = rotate3d{M_PI_4, M_PI_4, M_PI_2};
  const auto rotated3d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(first_rotated3d);
  const auto rotated3d_base_inertial =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          first_rotated3d);
  const auto rotated3d_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(first_rotated3d);
  const auto rotated3d_base_from_inertial =
      rotated3d_base_inertial->get_to_grid_frame();

  CHECK(rotated3d == *rotated3d_base);
  CHECK(*rotated3d_base == rotated3d);

  std::array<std::array<double, 3>, 4> coords3d{{{{0.1, 2.8, 9.3}},
                                                 {{-8.2, 2.8, -9.7}},
                                                 {{5.7, -4.9, 8.1}},
                                                 {{2.9, 3.4, -7.8}}}};

  for (const auto& coord : coords3d) {
    const tnsr::I<double, 3, Frame::Logical> source_points{
        {{coord[0], coord[1], coord[2]}}};

    CHECK((make_array<double, 3>(rotated3d(source_points))) ==
          first_rotated3d(coord));
    CHECK((make_array<double, 3>(rotated3d
                                     .inverse(tnsr::I<double, 3, Frame::Grid>{
                                         {{coord[0], coord[1], coord[2]}}})
                                     .value())) ==
          first_rotated3d.inverse(coord).value());

    const auto expected_jac_inertial = first_rotated3d.jacobian(coord);
    Jacobian<double, 3, Frame::Logical, Frame::Grid> expected_jac_grid{};
    REQUIRE(decltype(expected_jac_inertial)::size() ==
            decltype(expected_jac_grid)::size());
    for (size_t i = 0; i < decltype(expected_jac_inertial)::size(); ++i) {
      expected_jac_grid[i] = expected_jac_inertial[i];
    }
    CHECK(rotated3d.jacobian(source_points) == expected_jac_grid);

    const auto expected_inv_jac_inertial = first_rotated3d.inv_jacobian(coord);
    InverseJacobian<double, 3, Frame::Logical, Frame::Grid>
        expected_inv_jac_grid{};
    REQUIRE(decltype(expected_inv_jac_inertial)::size() ==
            decltype(expected_inv_jac_grid)::size());
    for (size_t i = 0; i < decltype(expected_inv_jac_inertial)::size(); ++i) {
      expected_inv_jac_grid[i] = expected_inv_jac_inertial[i];
    }
    CHECK(rotated3d.inv_jacobian(source_points) == expected_inv_jac_grid);

    check_map_ptr(first_rotated3d, rotated3d, rotated3d_base, coord,
                  source_points);
    check_map_ptr(first_rotated3d, rotated3d, rotated3d_base_from_inertial,
                  coord, source_points);
  }

  CHECK_FALSE(rotated3d.is_identity());
  CHECK_FALSE(rotated3d_base->is_identity());
  CHECK_FALSE(rotated3d_base_from_inertial->is_identity());

  CHECK_FALSE(rotated3d.inv_jacobian_is_time_dependent());
  CHECK_FALSE(rotated3d.jacobian_is_time_dependent());
  CHECK_FALSE(rotated3d_base->jacobian_is_time_dependent());
  CHECK_FALSE(rotated3d_base->inv_jacobian_is_time_dependent());
}

void test_coordinate_map_with_affine_map() {
  INFO("Coordinate map with affine map");
  using affine_map = CoordinateMaps::Affine;
  using affine_map_2d = CoordinateMaps::ProductOf2Maps<affine_map, affine_map>;
  using affine_map_3d =
      CoordinateMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;

  constexpr size_t number_of_points_checked = 10;

  // Test 1D
  const auto map = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map{-1.0, 1.0, 0.0, 2.3}, affine_map{0.0, 2.3, -0.5, 0.5});
  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    const tnsr::I<double, 1, Frame::Logical> source_points{2.0 / i + -1.0};
    CHECK((tnsr::I<double, 1, Frame::Grid>(1.0 / i + -0.5))[0] ==
          approx(map(source_points)[0]));
    CHECK((tnsr::I<double, 1, Frame::Logical>(2.0 / i + -1.0))[0] ==
          approx(map.inverse(tnsr::I<double, 1, Frame::Grid>{1.0 / i + -0.5})
                     .value()[0]));

    CHECK(approx(map.inv_jacobian(source_points).get(0, 0)) == 2.0);
    CHECK(approx(map.jacobian(source_points).get(0, 0)) == 0.5);

    const auto coords_jacs_velocity =
        map.coords_frame_velocity_jacobians(source_points);
    CHECK(std::get<0>(coords_jacs_velocity) == map(source_points));
    CHECK(std::get<1>(coords_jacs_velocity) == map.inv_jacobian(source_points));
    CHECK(std::get<2>(coords_jacs_velocity) == map.jacobian(source_points));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<double, 1, Frame::Grid>{0.0});
  }

  // Test 2D
  const auto prod_map2d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map_2d{affine_map{-1.0, 1.0, 0.0, 2.0},
                    affine_map{0.0, 2.0, -0.5, 0.5}},
      affine_map_2d{affine_map{0.0, 2.0, 2.0, 6.0},
                    affine_map{-0.5, 0.5, 0.0, 8.0}});
  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    const tnsr::I<double, 2, Frame::Logical> source_points{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}};
    const auto mapped_point = prod_map2d(source_points);
    const auto expected_mapped_point =
        tnsr::I<double, 2, Frame::Grid>{{{4.0 / i + 2.0, 8.0 / i + 0.0}}};
    CHECK(get<0>(expected_mapped_point) == approx(get<0>(mapped_point)));
    CHECK(get<1>(expected_mapped_point) == approx(get<1>(mapped_point)));

    const auto inv_mapped_point = prod_map2d
                                      .inverse(tnsr::I<double, 2, Frame::Grid>{
                                          {{4.0 / i + 2.0, 8.0 / i + 0.0}}})
                                      .value();
    const auto expected_inv_mapped_point =
        tnsr::I<double, 2, Frame::Grid>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}};
    CHECK(get<0>(expected_inv_mapped_point) ==
          approx(get<0>(inv_mapped_point)));
    CHECK(get<1>(expected_inv_mapped_point) ==
          approx(get<1>(inv_mapped_point)));

    const auto inv_jac = prod_map2d.inv_jacobian(source_points);
    CHECK(0.5 == approx(get<0, 0>(inv_jac)));
    CHECK(0.0 == approx(get<1, 0>(inv_jac)));
    CHECK(0.0 == approx(get<0, 1>(inv_jac)));
    CHECK(0.25 == approx(get<1, 1>(inv_jac)));

    const auto jac = prod_map2d.jacobian(source_points);
    CHECK(2.0 == approx(get<0, 0>(jac)));
    CHECK(0.0 == approx(get<1, 0>(jac)));
    CHECK(0.0 == approx(get<0, 1>(jac)));
    CHECK(4.0 == approx(get<1, 1>(jac)));

    const auto coords_jacs_velocity =
        prod_map2d.coords_frame_velocity_jacobians(source_points);
    CHECK(std::get<0>(coords_jacs_velocity) == prod_map2d(source_points));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          prod_map2d.inv_jacobian(source_points));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          prod_map2d.jacobian(source_points));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<double, 2, Frame::Grid>{0.0});
  }

  // Test 3D
  const auto prod_map3d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map_3d{affine_map{-1.0, 1.0, 0.0, 2.0},
                    affine_map{0.0, 2.0, -0.5, 0.5},
                    affine_map{5.0, 7.0, -7.0, 7.0}},
      affine_map_3d{affine_map{0.0, 2.0, 2.0, 6.0},
                    affine_map{-0.5, 0.5, 0.0, 8.0},
                    affine_map{-7.0, 7.0, 3.0, 23.0}});

  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    const tnsr::I<double, 3, Frame::Logical> source_points{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}};
    const auto mapped_point = prod_map3d(source_points);
    const auto expected_mapped_point = tnsr::I<double, 3, Frame::Grid>{
        {{4.0 / i + 2.0, 8.0 / i + 0.0, 3.0 + 20.0 / i}}};
    CHECK(get<0>(expected_mapped_point) == approx(get<0>(mapped_point)));
    CHECK(get<1>(expected_mapped_point) == approx(get<1>(mapped_point)));
    CHECK(get<2>(expected_mapped_point) == approx(get<2>(mapped_point)));

    const auto inv_mapped_point =
        prod_map3d
            .inverse(tnsr::I<double, 3, Frame::Grid>{
                {{4.0 / i + 2.0, 8.0 / i + 0.0, 3.0 + 20.0 / i}}})
            .value();
    const auto expected_inv_mapped_point = tnsr::I<double, 3, Frame::Grid>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}};
    CHECK(get<0>(expected_inv_mapped_point) ==
          approx(get<0>(inv_mapped_point)));
    CHECK(get<1>(expected_inv_mapped_point) ==
          approx(get<1>(inv_mapped_point)));
    CHECK(get<2>(expected_inv_mapped_point) ==
          approx(get<2>(inv_mapped_point)));

    const auto inv_jac = prod_map3d.inv_jacobian(source_points);
    CHECK(0.5 == approx(get<0, 0>(inv_jac)));
    CHECK(0.0 == approx(get<1, 0>(inv_jac)));
    CHECK(0.0 == approx(get<0, 1>(inv_jac)));
    CHECK(0.25 == approx(get<1, 1>(inv_jac)));
    CHECK(0.0 == approx(get<0, 2>(inv_jac)));
    CHECK(0.0 == approx(get<1, 2>(inv_jac)));
    CHECK(0.0 == approx(get<2, 0>(inv_jac)));
    CHECK(0.0 == approx(get<2, 1>(inv_jac)));
    CHECK(0.1 == approx(get<2, 2>(inv_jac)));

    const auto jac = prod_map3d.jacobian(source_points);
    CHECK(2.0 == approx(get<0, 0>(jac)));
    CHECK(0.0 == approx(get<1, 0>(jac)));
    CHECK(0.0 == approx(get<0, 1>(jac)));
    CHECK(4.0 == approx(get<1, 1>(jac)));
    CHECK(0.0 == approx(get<0, 2>(jac)));
    CHECK(0.0 == approx(get<1, 2>(jac)));
    CHECK(0.0 == approx(get<2, 0>(jac)));
    CHECK(0.0 == approx(get<2, 1>(jac)));
    CHECK(10.0 == approx(get<2, 2>(jac)));

    const auto coords_jacs_velocity =
        prod_map3d.coords_frame_velocity_jacobians(source_points);
    CHECK(std::get<0>(coords_jacs_velocity) == prod_map3d(source_points));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          prod_map3d.inv_jacobian(source_points));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          prod_map3d.jacobian(source_points));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<double, 3, Frame::Grid>{0.0});
  }
}

void test_coordinate_map_with_rotation_map() {
  INFO("Coordinate map with rotation map");
  using rotate2d = CoordinateMaps::Rotation<2>;
  using rotate3d = CoordinateMaps::Rotation<3>;

  // No 1D test because it would just the be affine map test

  // Test 2D
  const auto double_rotated2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotate2d{M_PI_4},
                                                       rotate2d{M_PI_2});
  const auto first_rotated2d = rotate2d{M_PI_4};
  const auto second_rotated2d = rotate2d{M_PI_2};

  std::array<std::array<double, 2>, 4> coords2d{
      {{{0.1, 2.8}}, {{-8.2, 2.8}}, {{5.7, -4.9}}, {{2.9, 3.4}}}};

  for (size_t i = 0; i < coords2d.size(); ++i) {
    INFO(i);
    const auto coord = gsl::at(coords2d, i);
    const tnsr::I<double, 2, Frame::Logical> source_points{
        {{coord[0], coord[1]}}};

    CHECK((make_array<double, 2>(double_rotated2d(source_points))) ==
          second_rotated2d(first_rotated2d(coord)));
    CHECK((make_array<double, 2>(double_rotated2d
                                     .inverse(tnsr::I<double, 2, Frame::Grid>{
                                         {{coord[0], coord[1]}}})
                                     .value())) ==
          first_rotated2d.inverse(second_rotated2d.inverse(coord).value())
              .value());

    const auto jac = double_rotated2d.jacobian(source_points);
    const auto expected_jac = compose_jacobians(
        first_rotated2d, second_rotated2d, gsl::at(coords2d, i));
    CHECK_ITERABLE_APPROX(jac, expected_jac);

    const auto inv_jac = double_rotated2d.inv_jacobian(source_points);
    const auto expected_inv_jac = compose_inv_jacobians(
        first_rotated2d, second_rotated2d, gsl::at(coords2d, i));
    CHECK_ITERABLE_APPROX(inv_jac, expected_inv_jac);

    const auto coords_jacs_velocity =
        double_rotated2d.coords_frame_velocity_jacobians(source_points);
    CHECK(std::get<0>(coords_jacs_velocity) == double_rotated2d(source_points));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          double_rotated2d.inv_jacobian(source_points));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          double_rotated2d.jacobian(source_points));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<double, 2, Frame::Grid>{0.0});
  }

  // Test 3D
  const auto double_rotated3d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(
          rotate3d{M_PI_4, M_PI_4, M_PI_2}, rotate3d{M_PI_2, M_PI_4, M_PI_4});
  const auto first_rotated3d = rotate3d{M_PI_4, M_PI_4, M_PI_2};
  const auto second_rotated3d = rotate3d{M_PI_2, M_PI_4, M_PI_4};

  std::array<std::array<double, 3>, 4> coords3d{{{{0.1, 2.8, 9.3}},
                                                 {{-8.2, 2.8, -9.7}},
                                                 {{5.7, -4.9, 8.1}},
                                                 {{2.9, 3.4, -7.8}}}};

  for (size_t i = 0; i < coords3d.size(); ++i) {
    INFO(i);
    const auto coord = gsl::at(coords3d, i);
    const tnsr::I<double, 3, Frame::Logical> source_points{
        {{coord[0], coord[1], coord[2]}}};

    CHECK((make_array<double, 3>(double_rotated3d(source_points))) ==
          second_rotated3d(first_rotated3d(coord)));
    CHECK((make_array<double, 3>(double_rotated3d
                                     .inverse(tnsr::I<double, 3, Frame::Grid>{
                                         {{coord[0], coord[1], coord[2]}}})
                                     .value())) ==
          first_rotated3d.inverse(second_rotated3d.inverse(coord).value())
              .value());

    const auto jac = double_rotated3d.jacobian(source_points);
    const auto expected_jac = compose_jacobians(
        first_rotated3d, second_rotated3d, gsl::at(coords3d, i));
    CHECK_ITERABLE_APPROX(jac, expected_jac);

    const auto inv_jac = double_rotated3d.inv_jacobian(source_points);
    const auto expected_inv_jac = compose_inv_jacobians(
        first_rotated3d, second_rotated3d, gsl::at(coords3d, i));
    CHECK_ITERABLE_APPROX(inv_jac, expected_inv_jac);

    const auto coords_jacs_velocity =
        double_rotated3d.coords_frame_velocity_jacobians(source_points);
    CHECK(std::get<0>(coords_jacs_velocity) == double_rotated3d(source_points));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          double_rotated3d.inv_jacobian(source_points));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          double_rotated3d.jacobian(source_points));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<double, 3, Frame::Grid>{0.0});
  }

  // Check inequivalence operator
  CHECK_FALSE(double_rotated3d != double_rotated3d);
  test_serialization(double_rotated3d);
}

void test_coordinate_map_with_rotation_map_datavector() {
  INFO("Coordinate map with rotation map datavector");
  using rotate2d = CoordinateMaps::Rotation<2>;
  using rotate3d = CoordinateMaps::Rotation<3>;

  // No 1D test because it would just the be affine map test

  // Test 2D
  {
    const auto double_rotated2d =
        make_coordinate_map<Frame::Logical, Frame::Grid>(rotate2d{M_PI_4},
                                                         rotate2d{M_PI_2});
    const auto first_rotated2d = rotate2d{M_PI_4};
    const auto second_rotated2d = rotate2d{M_PI_2};

    const tnsr::I<DataVector, 2, Frame::Logical> coords2d{
        {{DataVector{0.1, -8.2, 5.7, 2.9}, DataVector{2.8, 2.8, -4.9, 3.4}}}};
    const tnsr::I<DataVector, 2, Frame::Grid> coords2d_grid{
        {{DataVector{0.1, -8.2, 5.7, 2.9}, DataVector{2.8, 2.8, -4.9, 3.4}}}};
    const auto coords2d_array = make_array<DataVector, 2>(coords2d);

    CHECK((make_array<DataVector, 2>(double_rotated2d(coords2d))) ==
          second_rotated2d(first_rotated2d(coords2d_array)));

    const auto jac = double_rotated2d.jacobian(coords2d);
    const auto expected_jac =
        compose_jacobians(first_rotated2d, second_rotated2d, coords2d_array);
    CHECK_ITERABLE_APPROX(jac, expected_jac);

    const auto inv_jac = double_rotated2d.inv_jacobian(coords2d);
    const auto expected_inv_jac = compose_inv_jacobians(
        first_rotated2d, second_rotated2d, coords2d_array);
    CHECK_ITERABLE_APPROX(inv_jac, expected_inv_jac);

    const auto coords_jacs_velocity =
        double_rotated2d.coords_frame_velocity_jacobians(coords2d);
    CHECK(std::get<0>(coords_jacs_velocity) == double_rotated2d(coords2d));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          double_rotated2d.inv_jacobian(coords2d));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          double_rotated2d.jacobian(coords2d));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<DataVector, 2, Frame::Grid>{
              DataVector{coords2d.get(0).size(), 0.0}});
  }

  // Test 3D
  {
    const auto first_rotated3d = rotate3d{M_PI_4, M_PI_4, M_PI_2};
    const auto second_rotated3d = rotate3d{M_PI_2, M_PI_4, M_PI_4};
    const auto double_rotated3d_full =
        make_coordinate_map<Frame::Logical, Frame::Grid>(first_rotated3d,
                                                         second_rotated3d);
    const auto double_rotated3d_base =
        make_coordinate_map_base<Frame::Logical, Frame::Grid>(first_rotated3d,
                                                              second_rotated3d);
    const auto& double_rotated3d = *double_rotated3d_base;

    CHECK(double_rotated3d_full == double_rotated3d);

    const auto different_rotated3d_base =
        make_coordinate_map_base<Frame::Logical, Frame::Grid>(second_rotated3d,
                                                              first_rotated3d);
    CHECK(*different_rotated3d_base == *different_rotated3d_base);
    CHECK(*different_rotated3d_base != double_rotated3d);

    const tnsr::I<DataVector, 3, Frame::Logical> coords3d{
        {{DataVector{0.1, -8.2, 5.7, 2.9}, DataVector{2.8, 2.8, -4.9, 3.4},
          DataVector{9.3, -9.7, 8.1, -7.8}}}};
    const tnsr::I<DataVector, 3, Frame::Grid> coords3d_grid{
        {{DataVector{0.1, -8.2, 5.7, 2.9}, DataVector{2.8, 2.8, -4.9, 3.4},
          DataVector{9.3, -9.7, 8.1, -7.8}}}};
    const auto coords3d_array = make_array<DataVector, 3>(coords3d);

    CHECK((make_array<DataVector, 3>(double_rotated3d(coords3d))) ==
          second_rotated3d(first_rotated3d(coords3d_array)));

    const auto jac = double_rotated3d.jacobian(coords3d);
    const auto expected_jac =
        compose_jacobians(first_rotated3d, second_rotated3d, coords3d_array);
    CHECK_ITERABLE_APPROX(jac, expected_jac);

    const auto inv_jac = double_rotated3d.inv_jacobian(coords3d);
    const auto expected_inv_jac = compose_inv_jacobians(
        first_rotated3d, second_rotated3d, coords3d_array);
    CHECK_ITERABLE_APPROX(inv_jac, expected_inv_jac);

    // Check inequivalence operator
    CHECK_FALSE(double_rotated3d_full != double_rotated3d_full);
    test_serialization(double_rotated3d_full);

    const auto coords_jacs_velocity =
        double_rotated3d.coords_frame_velocity_jacobians(coords3d);
    CHECK(std::get<0>(coords_jacs_velocity) == double_rotated3d(coords3d));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          double_rotated3d.inv_jacobian(coords3d));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          double_rotated3d.jacobian(coords3d));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<DataVector, 3, Frame::Grid>{
              DataVector{coords3d.get(0).size(), 0.0}});
  }
}

void test_coordinate_map_with_rotation_wedge() {
  INFO("Coordinate map with rotation wedge");
  using Rotate = CoordinateMaps::Rotation<2>;
  using Wedge2D = CoordinateMaps::Wedge2D;

  const auto first_map = Rotate(2.0);
  const auto second_map =
      Wedge2D(3.0, 7.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
              false);

  const auto composed_map =
      make_coordinate_map<Frame::Logical, Frame::Grid>(first_map, second_map);

  const std::array<double, 2> test_point_array{{0.1, 0.8}};
  const tnsr::I<double, 2, Frame::Logical> test_point_vector(test_point_array);

  const auto mapped_point_array = second_map(first_map(test_point_array));
  const auto mapped_point_vector = composed_map(test_point_vector);
  CHECK((make_array<double, 2>(mapped_point_vector)) == mapped_point_array);

  const auto jac = composed_map.jacobian(test_point_vector);
  const auto expected_jac =
      compose_jacobians(first_map, second_map, test_point_array);
  CHECK_ITERABLE_APPROX(jac, expected_jac);

  const auto inv_jac = composed_map.inv_jacobian(test_point_vector);
  const auto expected_inv_jac =
      compose_inv_jacobians(first_map, second_map, test_point_array);
  CHECK_ITERABLE_APPROX(inv_jac, expected_inv_jac);

  const auto coords_jacs_velocity =
      composed_map.coords_frame_velocity_jacobians(test_point_vector);
  CHECK(std::get<0>(coords_jacs_velocity) == composed_map(test_point_vector));
  CHECK(std::get<1>(coords_jacs_velocity) ==
        composed_map.inv_jacobian(test_point_vector));
  CHECK(std::get<2>(coords_jacs_velocity) ==
        composed_map.jacobian(test_point_vector));
  CHECK(std::get<3>(coords_jacs_velocity) ==
        tnsr::I<double, 2, Frame::Grid>{0.0});
}

void test_make_vector_coordinate_map_base() {
  INFO("Make vector coordinate map base");
  using Affine = CoordinateMaps::Affine;
  using Affine2D = CoordinateMaps::ProductOf2Maps<Affine, Affine>;

  const auto affine1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      Affine{-1.0, 1.0, 2.0, 8.0});
  const auto affine1d_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(
          Affine{-1.0, 1.0, 2.0, 8.0});
  const auto vector_of_affine1d =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Grid>(
          Affine{-1.0, 1.0, 2.0, 8.0});

  CHECK(affine1d == *affine1d_base);
  CHECK(*affine1d_base == affine1d);
  CHECK(affine1d == *(vector_of_affine1d[0]));
  CHECK(*(vector_of_affine1d[0]) == affine1d);

  using Wedge2DMap = CoordinateMaps::Wedge2D;
  const auto upper_xi_wedge =
      Wedge2DMap{1.0,
                 2.0,
                 0.0,
                 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
                 true};
  const auto upper_eta_wedge =
      Wedge2DMap{1.0,
                 2.0,
                 0.0,
                 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
                 true};
  const auto lower_xi_wedge =
      Wedge2DMap{1.0,
                 2.0,
                 0.0,
                 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
                 true};
  const auto lower_eta_wedge =
      Wedge2DMap{1.0,
                 2.0,
                 0.0,
                 1.0,
                 OrientationMap<2>{std::array<Direction<2>, 2>{
                     {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
                 true};
  const auto vector_of_wedges =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
              true},
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
              true},
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
              true},
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
              true});

  CHECK(make_coordinate_map<Frame::Logical, Frame::Inertial>(upper_xi_wedge) ==
        *(vector_of_wedges[0]));
  CHECK(make_coordinate_map<Frame::Logical, Frame::Inertial>(upper_eta_wedge) ==
        *(vector_of_wedges[1]));
  CHECK(make_coordinate_map<Frame::Logical, Frame::Inertial>(lower_xi_wedge) ==
        *(vector_of_wedges[2]));
  CHECK(make_coordinate_map<Frame::Logical, Frame::Inertial>(lower_eta_wedge) ==
        *(vector_of_wedges[3]));
  CHECK(*make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            upper_xi_wedge) == *(vector_of_wedges[0]));
  CHECK(*make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            upper_eta_wedge) == *(vector_of_wedges[1]));
  CHECK(*make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            lower_xi_wedge) == *(vector_of_wedges[2]));
  CHECK(*make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            lower_eta_wedge) == *(vector_of_wedges[3]));

  const auto wedges = std::vector<Wedge2DMap>{upper_xi_wedge, upper_eta_wedge,
                                              lower_xi_wedge, lower_eta_wedge};
  const auto vector_of_wedges2 =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial, 2>(
          wedges);
  CHECK(make_coordinate_map<Frame::Logical, Frame::Inertial>(upper_xi_wedge) ==
        *(vector_of_wedges2[0]));
  CHECK(make_coordinate_map<Frame::Logical, Frame::Inertial>(upper_eta_wedge) ==
        *(vector_of_wedges2[1]));
  CHECK(make_coordinate_map<Frame::Logical, Frame::Inertial>(lower_xi_wedge) ==
        *(vector_of_wedges2[2]));
  CHECK(make_coordinate_map<Frame::Logical, Frame::Inertial>(lower_eta_wedge) ==
        *(vector_of_wedges2[3]));
  CHECK(*make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            upper_xi_wedge) == *(vector_of_wedges2[0]));
  CHECK(*make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            upper_eta_wedge) == *(vector_of_wedges2[1]));
  CHECK(*make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            lower_xi_wedge) == *(vector_of_wedges2[2]));
  CHECK(*make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            lower_eta_wedge) == *(vector_of_wedges2[3]));

  const auto translation =
      Affine2D{Affine{-1.0, 1.0, -1.0, 1.0}, Affine{-1.0, 1.0, 0.0, 2.0}};
  const auto vector_of_translated_wedges =
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial, 2>(
          wedges, translation);

  const auto translated_upper_xi_wedge =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
              true},
          translation);
  const auto translated_upper_eta_wedge =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
              true},
          translation);
  const auto translated_lower_xi_wedge =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
              true},
          translation);
  const auto translated_lower_eta_wedge =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
              true},
          translation);
  const auto translated_upper_xi_wedge_base =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_xi(), Direction<2>::upper_eta()}}},
              true},
          translation);
  const auto translated_upper_eta_wedge_base =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}},
              true},
          translation);
  const auto translated_lower_xi_wedge_base =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::lower_xi(), Direction<2>::lower_eta()}}},
              true},
          translation);
  const auto translated_lower_eta_wedge_base =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Wedge2DMap{
              1.0, 2.0, 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::lower_eta(), Direction<2>::upper_xi()}}},
              true},
          translation);

  CHECK(translated_upper_xi_wedge == *(vector_of_translated_wedges[0]));
  CHECK(translated_upper_eta_wedge == *(vector_of_translated_wedges[1]));
  CHECK(translated_lower_xi_wedge == *(vector_of_translated_wedges[2]));
  CHECK(translated_lower_eta_wedge == *(vector_of_translated_wedges[3]));
  CHECK(*translated_upper_xi_wedge_base == *(vector_of_translated_wedges[0]));
  CHECK(*translated_upper_eta_wedge_base == *(vector_of_translated_wedges[1]));
  CHECK(*translated_lower_xi_wedge_base == *(vector_of_translated_wedges[2]));
  CHECK(*translated_lower_eta_wedge_base == *(vector_of_translated_wedges[3]));
}

void test_coordinate_maps_are_identity() {
  INFO("Coordinate maps are identity");
  const auto giant_identity_map =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::Identity<3>{},
          CoordinateMaps::BulgedCube{sqrt(3.0), 0.0, false},
          CoordinateMaps::DiscreteRotation<3>{OrientationMap<3>{}},
          CoordinateMaps::EquatorialCompression{1.0},
          CoordinateMaps::Frustum{
              std::array<std::array<double, 2>, 4>{
                  {{{-1.0, -1.0}}, {{1.0, 1.0}}, {{-1.0, -1.0}}, {{1.0, 1.0}}}},
              -1.0, 1.0, OrientationMap<3>{}, false},
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>{
              CoordinateMaps::Affine{-1.0, 1.0, -1.0, 1.0},
              CoordinateMaps::Affine{-1.0, 1.0, -1.0, 1.0},
              CoordinateMaps::Affine{-1.0, 1.0, -1.0, 1.0}},
          CoordinateMaps::Rotation<3>{0.0, 0.0, 0.0},
          CoordinateMaps::SpecialMobius{0.0});
  const std::unique_ptr<CoordinateMapBase<Frame::Logical, Frame::Inertial, 3>>
      giant_identity_map_base =
          std::make_unique<std::decay_t<decltype(giant_identity_map)>>(
              giant_identity_map);
  test_serialization(giant_identity_map);

  CHECK(giant_identity_map.is_identity());
  CHECK(giant_identity_map_base->is_identity());

  CHECK_FALSE(giant_identity_map.inv_jacobian_is_time_dependent());
  CHECK_FALSE(giant_identity_map.jacobian_is_time_dependent());
  CHECK_FALSE(giant_identity_map_base->inv_jacobian_is_time_dependent());
  CHECK_FALSE(giant_identity_map_base->jacobian_is_time_dependent());

  const auto wedge = make_coordinate_map<Frame::Logical, Frame::Inertial>(
      CoordinateMaps::Wedge<3>(0.2, 4.0, OrientationMap<3>{}, 0.0, 1.0, true));
  const auto wedge_composed_with_giant_identity =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::Wedge<3>(0.2, 4.0, OrientationMap<3>{}, 0.0, 1.0,
                                   true),
          CoordinateMaps::Identity<3>{},
          CoordinateMaps::BulgedCube{sqrt(3.0), 0.0, false},
          CoordinateMaps::DiscreteRotation<3>{OrientationMap<3>{}},
          CoordinateMaps::EquatorialCompression{1.0},
          CoordinateMaps::Frustum{
              std::array<std::array<double, 2>, 4>{
                  {{{-1.0, -1.0}}, {{1.0, 1.0}}, {{-1.0, -1.0}}, {{1.0, 1.0}}}},
              -1.0, 1.0, OrientationMap<3>{}, false},
          CoordinateMaps::ProductOf3Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>{
              CoordinateMaps::Affine{-1.0, 1.0, -1.0, 1.0},
              CoordinateMaps::Affine{-1.0, 1.0, -1.0, 1.0},
              CoordinateMaps::Affine{-1.0, 1.0, -1.0, 1.0}},
          CoordinateMaps::Rotation<3>{0.0, 0.0, 0.0},
          CoordinateMaps::SpecialMobius{0.0});

  CHECK_FALSE(wedge.is_identity());
  CHECK_FALSE(wedge_composed_with_giant_identity.is_identity());

  CHECK_FALSE(wedge.inv_jacobian_is_time_dependent());
  CHECK_FALSE(wedge.jacobian_is_time_dependent());
  CHECK_FALSE(
      wedge_composed_with_giant_identity.inv_jacobian_is_time_dependent());
  CHECK_FALSE(wedge_composed_with_giant_identity.jacobian_is_time_dependent());

  for (size_t i = 1; i < 11; ++i) {
    const auto source_point = tnsr::I<double, 3, Frame::Logical>{
        {{-1.0 + 2.0 / i, -1.0 + 2.0 / i, -1.0 + 2.0 / i}}};
    const auto mapped_point = tnsr::I<double, 3, Frame::Inertial>{
        {{-1.0 + 2.0 / i, -1.0 + 2.0 / i, -1.0 + 2.0 / i}}};
    CHECK(get<0>(mapped_point) == get<0>(giant_identity_map(source_point)));
    CHECK(get<1>(mapped_point) == get<1>(giant_identity_map(source_point)));
    CHECK(get<2>(mapped_point) == get<2>(giant_identity_map(source_point)));
    CHECK(get<0>(source_point) ==
          get<0>(giant_identity_map.inverse(mapped_point).value()));
    CHECK(get<1>(source_point) ==
          get<1>(giant_identity_map.inverse(mapped_point).value()));
    CHECK(get<2>(source_point) ==
          get<2>(giant_identity_map.inverse(mapped_point).value()));
    const auto wedge_mapped_point = wedge(source_point);
    CHECK(get<0>(wedge_mapped_point) ==
          get<0>(wedge_composed_with_giant_identity(source_point)));
    CHECK(get<1>(wedge_mapped_point) ==
          get<1>(wedge_composed_with_giant_identity(source_point)));
    CHECK(get<2>(wedge_mapped_point) ==
          get<2>(wedge_composed_with_giant_identity(source_point)));
    CHECK(get<0>(wedge.inverse(wedge_mapped_point).value()) ==
          get<0>(wedge_composed_with_giant_identity.inverse(wedge_mapped_point)
                     .value()));
    CHECK(get<1>(wedge.inverse(wedge_mapped_point).value()) ==
          get<1>(wedge_composed_with_giant_identity.inverse(wedge_mapped_point)
                     .value()));
    CHECK(get<2>(wedge.inverse(wedge_mapped_point).value()) ==
          get<2>(wedge_composed_with_giant_identity.inverse(wedge_mapped_point)
                     .value()));

    const auto inv_jac = giant_identity_map.inv_jacobian(source_point);
    CHECK(1.0 == approx(get<0, 0>(inv_jac)));
    CHECK(0.0 == approx(get<1, 0>(inv_jac)));
    CHECK(0.0 == approx(get<2, 0>(inv_jac)));
    CHECK(0.0 == approx(get<0, 1>(inv_jac)));
    CHECK(1.0 == approx(get<1, 1>(inv_jac)));
    CHECK(0.0 == approx(get<2, 1>(inv_jac)));
    CHECK(0.0 == approx(get<0, 2>(inv_jac)));
    CHECK(0.0 == approx(get<1, 2>(inv_jac)));
    CHECK(1.0 == approx(get<2, 2>(inv_jac)));
    const auto jac = giant_identity_map.jacobian(source_point);
    CHECK(1.0 == approx(get<0, 0>(jac)));
    CHECK(0.0 == approx(get<1, 0>(jac)));
    CHECK(0.0 == approx(get<2, 0>(jac)));
    CHECK(0.0 == approx(get<0, 1>(jac)));
    CHECK(1.0 == approx(get<1, 1>(jac)));
    CHECK(0.0 == approx(get<2, 1>(jac)));
    CHECK(0.0 == approx(get<0, 2>(jac)));
    CHECK(0.0 == approx(get<1, 2>(jac)));
    CHECK(1.0 == approx(get<2, 2>(jac)));

    const auto wedge_1_jac = wedge.jacobian(source_point);
    const auto wedge_2_jac =
        wedge_composed_with_giant_identity.jacobian(source_point);

    CHECK(get<0, 0>(wedge_1_jac) == get<0, 0>(wedge_2_jac));
    CHECK(get<1, 0>(wedge_1_jac) == get<1, 0>(wedge_2_jac));
    CHECK(get<2, 0>(wedge_1_jac) == get<2, 0>(wedge_2_jac));
    CHECK(get<0, 0>(wedge_1_jac) == get<0, 0>(wedge_2_jac));
    CHECK(get<1, 0>(wedge_1_jac) == get<1, 0>(wedge_2_jac));
    CHECK(get<2, 0>(wedge_1_jac) == get<2, 0>(wedge_2_jac));
    CHECK(get<0, 0>(wedge_1_jac) == get<0, 0>(wedge_2_jac));
    CHECK(get<1, 0>(wedge_1_jac) == get<1, 0>(wedge_2_jac));
    CHECK(get<2, 0>(wedge_1_jac) == get<2, 0>(wedge_2_jac));

    const auto wedge_1_jac_inv = wedge.inv_jacobian(source_point);
    const auto wedge_2_jac_inv =
        wedge_composed_with_giant_identity.inv_jacobian(source_point);
    CHECK(get<0, 0>(wedge_1_jac_inv) == get<0, 0>(wedge_2_jac_inv));
    CHECK(get<1, 0>(wedge_1_jac_inv) == get<1, 0>(wedge_2_jac_inv));
    CHECK(get<2, 0>(wedge_1_jac_inv) == get<2, 0>(wedge_2_jac_inv));
    CHECK(get<0, 0>(wedge_1_jac_inv) == get<0, 0>(wedge_2_jac_inv));
    CHECK(get<1, 0>(wedge_1_jac_inv) == get<1, 0>(wedge_2_jac_inv));
    CHECK(get<2, 0>(wedge_1_jac_inv) == get<2, 0>(wedge_2_jac_inv));
    CHECK(get<0, 0>(wedge_1_jac_inv) == get<0, 0>(wedge_2_jac_inv));
    CHECK(get<1, 0>(wedge_1_jac_inv) == get<1, 0>(wedge_2_jac_inv));
    CHECK(get<2, 0>(wedge_1_jac_inv) == get<2, 0>(wedge_2_jac_inv));

    const auto coords_jacs_velocity =
        wedge_composed_with_giant_identity.coords_frame_velocity_jacobians(
            source_point);
    CHECK(std::get<0>(coords_jacs_velocity) ==
          wedge_composed_with_giant_identity(source_point));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          wedge_composed_with_giant_identity.inv_jacobian(source_point));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          wedge_composed_with_giant_identity.jacobian(source_point));
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<double, 3, Frame::Inertial>{0.0});
  }
}

void test_time_dependent_map() {
  INFO("Time dependent CoordinateMap")
  // define vars for FunctionOfTime::PiecewisePolynomial f(t) = t**2.
  const double initial_time = -1.;
  const double final_time = 4.4;
  constexpr size_t deriv_order = 3;

  const std::array<DataVector, deriv_order + 1> init_func{
      {{1.0}, {-2.0}, {2.0}, {0.0}}};
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["Translation"] =
      std::make_unique<Polynomial>(initial_time, init_func, final_time);

  const CoordinateMaps::TimeDependent::Translation trans_map{"Translation"};

  // affine(x) = 1.5 * x + 5.5
  domain::CoordinateMaps::Affine affine_map{-1., 1., 4., 7.};

  const auto time_dependent_map_first =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(trans_map,
                                                           affine_map);
  const auto time_dependent_map_second =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(affine_map,
                                                           trans_map);

  CHECK_FALSE(time_dependent_map_first.inv_jacobian_is_time_dependent());
  CHECK_FALSE(time_dependent_map_first.jacobian_is_time_dependent());
  CHECK_FALSE(time_dependent_map_second.inv_jacobian_is_time_dependent());
  CHECK_FALSE(time_dependent_map_second.jacobian_is_time_dependent());

  const tnsr::I<double, 1, Frame::Logical> tnsr_double_logical{{{3.2}}};
  const tnsr::I<DataVector, 1, Frame::Logical> tnsr_datavector_logical{
      DataVector{{-4.3, 10.1, -3.5}}};

  const tnsr::I<double, 1, Frame::Inertial> tnsr_double_inertial_1{{{39.34}}};
  const tnsr::I<double, 1, Frame::Inertial> tnsr_double_inertial_2{{{29.66}}};

  const tnsr::I<DataVector, 1, Frame::Inertial> tnsr_datavector_inertial_1{
      DataVector{28.09, 49.69, 29.29}};
  const tnsr::I<DataVector, 1, Frame::Inertial> tnsr_datavector_inertial_2{
      DataVector{18.41, 40.01, 19.61}};

  CHECK_ITERABLE_APPROX(time_dependent_map_first(tnsr_double_logical,
                                                 final_time, functions_of_time),
                        tnsr_double_inertial_1);

  CHECK_ITERABLE_APPROX(time_dependent_map_second(
                            tnsr_double_logical, final_time, functions_of_time),
                        tnsr_double_inertial_2);

  CHECK_ITERABLE_APPROX(time_dependent_map_first(tnsr_datavector_logical,
                                                 final_time, functions_of_time),
                        tnsr_datavector_inertial_1);
  CHECK_ITERABLE_APPROX(
      time_dependent_map_second(tnsr_datavector_logical, final_time,
                                functions_of_time),
      tnsr_datavector_inertial_2);

  CHECK_ITERABLE_APPROX(
      *(time_dependent_map_first.inverse(tnsr_double_inertial_1, final_time,
                                         functions_of_time)),
      tnsr_double_logical);
  CHECK_ITERABLE_APPROX(
      *(time_dependent_map_second.inverse(tnsr_double_inertial_2, final_time,
                                          functions_of_time)),
      tnsr_double_logical);

  CHECK(time_dependent_map_first
            .jacobian(tnsr_double_logical, final_time, functions_of_time)
            .get(0, 0) == 1.5);
  CHECK(time_dependent_map_first
            .inv_jacobian(tnsr_double_logical, final_time, functions_of_time)
            .get(0, 0) == 2. / 3.);
  CHECK_ITERABLE_APPROX(
      time_dependent_map_first
          .jacobian(tnsr_datavector_logical, final_time, functions_of_time)
          .get(0, 0),
      (DataVector{1.5, 1.5, 1.5}));
  CHECK_ITERABLE_APPROX(
      time_dependent_map_first
          .inv_jacobian(tnsr_datavector_logical, final_time, functions_of_time)
          .get(0, 0),
      (DataVector{2. / 3., 2. / 3., 2. / 3.}));

  test_serialization(time_dependent_map_first);

  const auto serialized_map =
      serialize_and_deserialize(time_dependent_map_first);

  CHECK_ITERABLE_APPROX(
      serialized_map(tnsr_double_logical, final_time, functions_of_time),
      tnsr_double_inertial_1);

  CHECK_ITERABLE_APPROX(
      serialized_map(tnsr_datavector_logical, final_time, functions_of_time),
      tnsr_datavector_inertial_1);

  CHECK_ITERABLE_APPROX(
      *(serialized_map.inverse(tnsr_double_inertial_1, final_time,
                               functions_of_time)),
      tnsr_double_logical);

  CHECK(serialized_map
            .jacobian(tnsr_double_logical, final_time, functions_of_time)
            .get(0, 0) == 1.5);
  CHECK(serialized_map
            .inv_jacobian(tnsr_double_logical, final_time, functions_of_time)
            .get(0, 0) == 2. / 3.);
  CHECK_ITERABLE_APPROX(
      serialized_map
          .jacobian(tnsr_datavector_logical, final_time, functions_of_time)
          .get(0, 0),
      (DataVector{1.5, 1.5, 1.5}));
  CHECK_ITERABLE_APPROX(
      serialized_map
          .inv_jacobian(tnsr_datavector_logical, final_time, functions_of_time)
          .get(0, 0),
      (DataVector{2. / 3., 2. / 3., 2. / 3.}));

  {
    const auto coords_jacs_velocity =
        serialized_map.coords_frame_velocity_jacobians(
            tnsr_double_logical, final_time, functions_of_time);
    CHECK(std::get<0>(coords_jacs_velocity) ==
          serialized_map(tnsr_double_logical, final_time, functions_of_time));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          serialized_map.inv_jacobian(tnsr_double_logical, final_time,
                                      functions_of_time));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          serialized_map.jacobian(tnsr_double_logical, final_time,
                                  functions_of_time));
    const auto velocity =
        functions_of_time.at("Translation")->func_and_deriv(final_time)[1];
    // The 1.5 factor comes from the Jacobian
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<double, 1, Frame::Inertial>{1.5 *
                                              velocity[velocity.size() - 1]});
  }
  {
    const auto coords_jacs_velocity =
        serialized_map.coords_frame_velocity_jacobians(
            tnsr_datavector_logical, final_time, functions_of_time);
    CHECK(
        std::get<0>(coords_jacs_velocity) ==
        serialized_map(tnsr_datavector_logical, final_time, functions_of_time));
    CHECK(std::get<1>(coords_jacs_velocity) ==
          serialized_map.inv_jacobian(tnsr_datavector_logical, final_time,
                                      functions_of_time));
    CHECK(std::get<2>(coords_jacs_velocity) ==
          serialized_map.jacobian(tnsr_datavector_logical, final_time,
                                  functions_of_time));
    const auto velocity =
        functions_of_time.at("Translation")->func_and_deriv(final_time)[1];
    // The 1.5 factor comes from the Jacobian
    CHECK(std::get<3>(coords_jacs_velocity) ==
          tnsr::I<DataVector, 1, Frame::Inertial>{
              DataVector{tnsr_datavector_logical.get(0).size(),
                         1.5 * velocity[velocity.size() - 1]}});
  }
}

void test_push_back() {
  INFO("Coordinate map with affine map");
  using affine_map = CoordinateMaps::Affine;
  using affine_map_2d = CoordinateMaps::ProductOf2Maps<affine_map, affine_map>;
  using affine_map_3d =
      CoordinateMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;

  const auto check_coord_map_push = [](const auto& first_map,
                                       const auto& second_map) noexcept {
    const auto first_coord_map =
        make_coordinate_map<Frame::Logical, Frame::Grid>(first_map);
    const auto second_coord_map =
        make_coordinate_map<Frame::Logical, Frame::Grid>(second_map);
    const auto composed_map =
        make_coordinate_map<Frame::Logical, Frame::Grid>(first_map, second_map);
    const auto composed_push_back_map = push_back(first_coord_map, second_map);
    const auto composed_push_front_map =
        push_front(second_coord_map, first_map);
    CHECK(composed_map == composed_push_back_map);
    CHECK(composed_map == composed_push_front_map);

    CHECK(composed_map == push_back(first_coord_map, second_coord_map));
    CHECK(composed_map == push_front(second_coord_map, first_coord_map));
  };

  // Test 1d
  check_coord_map_push(affine_map{-1.0, 1.0, 0.0, 2.3},
                       affine_map{0.0, 2.3, -0.5, 0.5});
  // Test 2d
  check_coord_map_push(affine_map_2d{affine_map{-1.0, 1.0, 0.0, 2.0},
                                     affine_map{0.0, 2.0, -0.5, 0.5}},
                       affine_map_2d{affine_map{0.0, 2.0, 2.0, 6.0},
                                     affine_map{-0.5, 0.5, 0.0, 8.0}});
  // Test 3d
  check_coord_map_push(affine_map_3d{affine_map{-1.0, 1.0, 0.0, 2.0},
                                     affine_map{0.0, 2.0, -0.5, 0.5},
                                     affine_map{5.0, 7.0, -7.0, 7.0}},
                       affine_map_3d{affine_map{0.0, 2.0, 2.0, 6.0},
                                     affine_map{-0.5, 0.5, 0.0, 8.0},
                                     affine_map{-7.0, 7.0, 3.0, 23.0}});
}

void test_jacobian_is_time_dependent() noexcept {
  using affine_map = CoordinateMaps::Affine;
  using cubic_scale_map = CoordinateMaps::TimeDependent::CubicScale<1>;
  using map_2d = CoordinateMaps::TimeDependent::ProductOf2Maps<affine_map,
                                                               cubic_scale_map>;
  using map_3d =
      CoordinateMaps::TimeDependent::ProductOf3Maps<affine_map, affine_map,
                                                    cubic_scale_map>;

  const auto coord_map_1 = make_coordinate_map<Frame::Logical, Frame::Grid>(
      cubic_scale_map(10.0, "ExpansionA", "ExpansionB"));
  const auto coord_map_1_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(
          cubic_scale_map(10.0, "ExpansionA", "ExpansionB"));

  const auto coord_map_2 = make_coordinate_map<Frame::Logical, Frame::Grid>(
      map_2d(affine_map(-1.0, 1.0, 2.0, 3.0),
             cubic_scale_map(10.0, "ExpansionA", "ExpansionB")));
  const auto coord_map_2_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(
          map_2d(affine_map(-1.0, 1.0, 2.0, 3.0),
                 cubic_scale_map(10.0, "ExpansionA", "ExpansionB")));

  const auto coord_map_3 = make_coordinate_map<Frame::Logical, Frame::Grid>(
      map_3d(affine_map(-1.0, 1.0, 2.0, 3.0), affine_map(-1.0, 1.0, 2.0, 3.0),
             cubic_scale_map(10.0, "ExpansionA", "ExpansionB")));
  const auto coord_map_3_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(map_3d(
          affine_map(-1.0, 1.0, 2.0, 3.0), affine_map(-1.0, 1.0, 2.0, 3.0),
          cubic_scale_map(10.0, "ExpansionA", "ExpansionB")));

  CHECK(coord_map_1.inv_jacobian_is_time_dependent());
  CHECK(coord_map_1.jacobian_is_time_dependent());
  CHECK(coord_map_1_base->inv_jacobian_is_time_dependent());
  CHECK(coord_map_1_base->jacobian_is_time_dependent());

  CHECK(coord_map_2.inv_jacobian_is_time_dependent());
  CHECK(coord_map_2.jacobian_is_time_dependent());
  CHECK(coord_map_2_base->inv_jacobian_is_time_dependent());
  CHECK(coord_map_2_base->jacobian_is_time_dependent());

  CHECK(coord_map_3.inv_jacobian_is_time_dependent());
  CHECK(coord_map_3.jacobian_is_time_dependent());
  CHECK(coord_map_3_base->inv_jacobian_is_time_dependent());
  CHECK(coord_map_3_base->jacobian_is_time_dependent());
}

void test_coords_frame_velocity_jacobians() noexcept {
  using affine_map = CoordinateMaps::Affine;
  using trans_map = CoordinateMaps::TimeDependent::Translation;
  using affine_map_2d = CoordinateMaps::ProductOf2Maps<affine_map, affine_map>;
  using trans_map_2d =
      CoordinateMaps::TimeDependent::ProductOf2Maps<trans_map, trans_map>;
  using affine_map_3d =
      CoordinateMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;
  using trans_map_3d =
      CoordinateMaps::TimeDependent::ProductOf3Maps<trans_map, trans_map,
                                                    trans_map>;

  const double initial_time = 0.0;
  const double final_time   = 2.0;
  const double time = 2.0;
  constexpr size_t deriv_order = 3;

  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time["trans_x"] = std::make_unique<Polynomial>(
      initial_time,
      std::array<DataVector, deriv_order + 1>{{{1.0}, {-2.0}, {0.0}, {0.0}}},
      final_time);
  functions_of_time["trans_y"] = std::make_unique<Polynomial>(
      initial_time,
      std::array<DataVector, deriv_order + 1>{{{1.0}, {3.0}, {0.0}, {0.0}}},
      final_time);
  functions_of_time["trans_z"] = std::make_unique<Polynomial>(
      initial_time,
      std::array<DataVector, deriv_order + 1>{{{1.0}, {4.5}, {0.0}, {0.0}}},
      final_time);
  functions_of_time["ExpansionA"] = std::make_unique<Polynomial>(
      initial_time,
      std::array<DataVector, deriv_order + 1>{{{1.0}, {-0.01}, {0.0}, {0.0}}},
      final_time);
  functions_of_time["ExpansionB"] = std::make_unique<Polynomial>(
      initial_time,
      std::array<DataVector, deriv_order + 1>{{{1.0}, {0.0}, {0.0}, {0.0}}},
      final_time);

  const auto composed_map_1d =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          trans_map{"trans_x"}, affine_map{-1.0, 1.0, 0.0, 2.3},
          trans_map{"trans_x"});
  CHECK(
      std::get<3>(composed_map_1d.coords_frame_velocity_jacobians(
          tnsr::I<double, 1, Frame::Logical>{0.5}, time, functions_of_time)) ==
      tnsr::I<double, 1, Frame::Inertial>{-2.0 + 1.15 * -2.0});
  CHECK(std::get<3>(composed_map_1d.coords_frame_velocity_jacobians(
            tnsr::I<DataVector, 1, Frame::Logical>{DataVector{-0.5, 0.0, 0.5}},
            time, functions_of_time)) ==
        tnsr::I<DataVector, 1, Frame::Inertial>{3_st, -2.0 + 1.15 * -2.0});

  const auto composed_map_2d =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          trans_map_2d{trans_map{"trans_x"}, trans_map{"trans_y"}},
          affine_map_2d{affine_map{-1.0, 1.0, 0.0, 2.3},
                        affine_map{-1.0, 1.0, 1.0, 7.2}},
          trans_map_2d{trans_map{"trans_x"}, trans_map{"trans_y"}});
  CHECK(
      std::get<3>(composed_map_2d.coords_frame_velocity_jacobians(
          tnsr::I<double, 2, Frame::Logical>{0.5}, time, functions_of_time)) ==
      tnsr::I<double, 2, Frame::Inertial>{
          {{-2.0 + 1.15 * -2.0, 3.0 + 3.1 * 3.0}}});
  CHECK(std::get<3>(composed_map_2d.coords_frame_velocity_jacobians(
            tnsr::I<DataVector, 2, Frame::Logical>{DataVector{-0.5, 0.0, 0.5}},
            time, functions_of_time)) ==
        tnsr::I<DataVector, 2, Frame::Inertial>{
            {{DataVector{3_st, -2.0 + 1.15 * -2.0},
              DataVector{3_st, 3.0 + 3.1 * 3.0}}}});

  const trans_map_3d translation3d{trans_map{"trans_x"}, trans_map{"trans_y"},
                                   trans_map{"trans_z"}};
  const affine_map_3d affine3d{affine_map{-1.0, 1.0, 0.0, 2.3},
                               affine_map{-1.0, 1.0, 1.0, 7.2},
                               affine_map{-1.0, 1.0, -10.0, 7.2}};
  const CoordinateMaps::TimeDependent::CubicScale<3> cubic_scale{
      20.0, "ExpansionA", "ExpansionB"};
  const auto composed_map_3d =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          translation3d, affine3d, cubic_scale);
  {
    const std::array<double, 3> source_pt{{0.5, 0.25, -0.34}};
    const std::array<double, 3> source_pt_cubic_scale =
        affine3d(translation3d(source_pt, time, functions_of_time));
    const std::array<double, 3> velocity_affine_map_frame{
        {1.15 * -2.0, 3.1 * 3.0, 8.6 * 4.5}};
    const auto cubic_scale_jac =
        cubic_scale.jacobian(source_pt_cubic_scale, time, functions_of_time);
    const auto cubic_scale_velocity = cubic_scale.frame_velocity(
        source_pt_cubic_scale, time, functions_of_time);
    const tnsr::I<double, 3, Frame::Inertial> expected_velocity{
        {{cubic_scale_velocity[0] +
              cubic_scale_jac.get(0, 0) * velocity_affine_map_frame[0] +
              cubic_scale_jac.get(0, 1) * velocity_affine_map_frame[1] +
              cubic_scale_jac.get(0, 2) * velocity_affine_map_frame[2],
          cubic_scale_velocity[1] +
              cubic_scale_jac.get(1, 0) * velocity_affine_map_frame[0] +
              cubic_scale_jac.get(1, 1) * velocity_affine_map_frame[1] +
              cubic_scale_jac.get(1, 2) * velocity_affine_map_frame[2],
          cubic_scale_velocity[2] +
              cubic_scale_jac.get(2, 0) * velocity_affine_map_frame[0] +
              cubic_scale_jac.get(2, 1) * velocity_affine_map_frame[1] +
              cubic_scale_jac.get(2, 2) * velocity_affine_map_frame[2]}}};

    CHECK(std::get<3>(composed_map_3d.coords_frame_velocity_jacobians(
              tnsr::I<double, 3, Frame::Logical>{source_pt}, time,
              functions_of_time)) == expected_velocity);
  }
  {
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist(-1.0, 1.0);
    const auto source_pt = make_with_random_values<std::array<DataVector, 3>>(
        make_not_null(&generator), make_not_null(&dist), DataVector{5});
    const std::array<DataVector, 3> source_pt_cubic_scale =
        affine3d(translation3d(source_pt, time, functions_of_time));
    const std::array<DataVector, 3> velocity_affine_map_frame{
        {DataVector{5, 1.15} * -2.0, DataVector{5, 3.1} * 3.0,
         DataVector{5, 8.6} * 4.5}};
    const auto cubic_scale_jac =
        cubic_scale.jacobian(source_pt_cubic_scale, time, functions_of_time);
    const auto cubic_scale_velocity = cubic_scale.frame_velocity(
        source_pt_cubic_scale, time, functions_of_time);
    const tnsr::I<DataVector, 3, Frame::Inertial> expected_velocity{
        {{cubic_scale_velocity[0] +
              cubic_scale_jac.get(0, 0) * velocity_affine_map_frame[0] +
              cubic_scale_jac.get(0, 1) * velocity_affine_map_frame[1] +
              cubic_scale_jac.get(0, 2) * velocity_affine_map_frame[2],
          cubic_scale_velocity[1] +
              cubic_scale_jac.get(1, 0) * velocity_affine_map_frame[0] +
              cubic_scale_jac.get(1, 1) * velocity_affine_map_frame[1] +
              cubic_scale_jac.get(1, 2) * velocity_affine_map_frame[2],
          cubic_scale_velocity[2] +
              cubic_scale_jac.get(2, 0) * velocity_affine_map_frame[0] +
              cubic_scale_jac.get(2, 1) * velocity_affine_map_frame[1] +
              cubic_scale_jac.get(2, 2) * velocity_affine_map_frame[2]}}};
    CHECK(std::get<3>(composed_map_3d.coords_frame_velocity_jacobians(
              tnsr::I<DataVector, 3, Frame::Logical>{source_pt}, time,
              functions_of_time)) == expected_velocity);
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMap", "[Domain][Unit]") {
  test_single_coordinate_map();
  test_coordinate_map_with_affine_map();
  test_coordinate_map_with_rotation_map();
  test_coordinate_map_with_rotation_map_datavector();
  test_coordinate_map_with_rotation_wedge();
  test_make_vector_coordinate_map_base();
  test_coordinate_maps_are_identity();
  test_time_dependent_map();
  test_push_back();
  test_jacobian_is_time_dependent();
  test_coords_frame_velocity_jacobians();
}
}  // namespace domain
