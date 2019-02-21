// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cmath>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/BulgedCube.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
#include "Domain/CoordinateMaps/Frustum.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/SpecialMobius.hpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/Direction.hpp"
#include "Domain/OrientationMap.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tensor

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
  const auto affine1d_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(
          affine_map1d{-1.0, 1.0, 2.0, 8.0});
  const auto first_affine1d = affine_map1d{-1.0, 1.0, 2.0, 8.0};

  CHECK(affine1d == *affine1d_base);
  CHECK(*affine1d_base == affine1d);

  std::array<std::array<double, 1>, 4> coords1d{
      {{{0.1}}, {{-8.2}}, {{5.7}}, {{2.9}}}};

  for (const auto& coord : coords1d) {
    CHECK((make_array<double, 1>((*affine1d_base)(
              tnsr::I<double, 1, Frame::Logical>{{{coord[0]}}}))) ==
          first_affine1d(coord));
    CHECK((make_array<double, 1>(
              affine1d_base
                  ->inverse(tnsr::I<double, 1, Frame::Grid>{{{coord[0]}}})
                  .get())) == first_affine1d.inverse(coord).get());

    CHECK((make_array<double, 1>(affine1d(tnsr::I<double, 1, Frame::Logical>{
              {{coord[0]}}}))) == first_affine1d(coord));
    CHECK((make_array<double, 1>(
              affine1d.inverse(tnsr::I<double, 1, Frame::Grid>{{{coord[0]}}})
                  .get())) == first_affine1d.inverse(coord).get());

    const auto jac =
        affine1d.jacobian(tnsr::I<double, 1, Frame::Logical>{{{coord[0]}}});
    const auto expected_jac = first_affine1d.jacobian(coord);
    CHECK(affine1d_base
              ->jacobian(tnsr::I<double, 1, Frame::Logical>{{{coord[0]}}})
              .get(0, 0) == expected_jac.get(0, 0));
    CHECK(jac.get(0, 0) == expected_jac.get(0, 0));

    const auto inv_jac =
        affine1d.inv_jacobian(tnsr::I<double, 1, Frame::Logical>{{{coord[0]}}});
    const auto expected_inv_jac = first_affine1d.inv_jacobian(coord);
    CHECK(affine1d_base
              ->inv_jacobian(tnsr::I<double, 1, Frame::Logical>{{{coord[0]}}})
              .get(0, 0) == expected_inv_jac.get(0, 0));
    CHECK(inv_jac.get(0, 0) == expected_inv_jac.get(0, 0));
  }

  using rotate2d = CoordinateMaps::Rotation<2>;

  const auto first_rotated2d = rotate2d{M_PI_4};
  const auto rotated2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(first_rotated2d);
  const auto rotated2d_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(first_rotated2d);

  CHECK(rotated2d == *rotated2d_base);
  CHECK(*rotated2d_base == rotated2d);

  std::array<std::array<double, 2>, 4> coords2d{
      {{{0.1, 2.8}}, {{-8.2, 2.8}}, {{5.7, -4.9}}, {{2.9, 3.4}}}};

  for (const auto& coord : coords2d) {
    CHECK((make_array<double, 2>((*rotated2d_base)(
              tnsr::I<double, 2, Frame::Logical>{{{coord[0], coord[1]}}}))) ==
          first_rotated2d(coord));
    CHECK((make_array<double, 2>(rotated2d_base
                                     ->inverse(tnsr::I<double, 2, Frame::Grid>{
                                         {{coord[0], coord[1]}}})
                                     .get())) ==
          first_rotated2d.inverse(coord).get());

    CHECK((make_array<double, 2>(rotated2d(tnsr::I<double, 2, Frame::Logical>{
              {{coord[0], coord[1]}}}))) == first_rotated2d(coord));
    CHECK((make_array<double, 2>(rotated2d
                                     .inverse(tnsr::I<double, 2, Frame::Grid>{
                                         {{coord[0], coord[1]}}})
                                     .get())) ==
          first_rotated2d.inverse(coord).get());

    const auto jac = rotated2d.jacobian(
        tnsr::I<double, 2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto jac2 = rotated2d_base->jacobian(
        tnsr::I<double, 2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto expected_jac = first_rotated2d.jacobian(coord);
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
        CHECK(jac2.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = rotated2d.inv_jacobian(
        tnsr::I<double, 2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto inv_jac2 = rotated2d_base->inv_jacobian(
        tnsr::I<double, 2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto expected_inv_jac = first_rotated2d.inv_jacobian(coord);
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
        CHECK(inv_jac2.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }

  using rotate3d = CoordinateMaps::Rotation<3>;

  const auto first_rotated3d = rotate3d{M_PI_4, M_PI_4, M_PI_2};
  const auto rotated3d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(first_rotated3d);
  const auto rotated3d_base =
      make_coordinate_map_base<Frame::Logical, Frame::Grid>(first_rotated3d);

  CHECK(rotated3d == *rotated3d_base);
  CHECK(*rotated3d_base == rotated3d);

  std::array<std::array<double, 3>, 4> coords3d{{{{0.1, 2.8, 9.3}},
                                                 {{-8.2, 2.8, -9.7}},
                                                 {{5.7, -4.9, 8.1}},
                                                 {{2.9, 3.4, -7.8}}}};

  for (const auto& coord : coords3d) {
    CHECK((make_array<double, 3>((
              *rotated3d_base)(tnsr::I<double, 3, Frame::Logical>{
              {{coord[0], coord[1], coord[2]}}}))) == first_rotated3d(coord));
    CHECK((make_array<double, 3>(rotated3d_base
                                     ->inverse(tnsr::I<double, 3, Frame::Grid>{
                                         {{coord[0], coord[1], coord[2]}}})
                                     .get())) ==
          first_rotated3d.inverse(coord).get());

    CHECK((make_array<double, 3>(rotated3d(tnsr::I<double, 3, Frame::Logical>{
              {{coord[0], coord[1], coord[2]}}}))) == first_rotated3d(coord));
    CHECK((make_array<double, 3>(rotated3d
                                     .inverse(tnsr::I<double, 3, Frame::Grid>{
                                         {{coord[0], coord[1], coord[2]}}})
                                     .get())) ==
          first_rotated3d.inverse(coord).get());

    const auto jac = rotated3d.jacobian(
        tnsr::I<double, 3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto jac2 = rotated3d_base->jacobian(
        tnsr::I<double, 3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto expected_jac = first_rotated3d.jacobian(coord);
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
        CHECK(jac2.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = rotated3d.inv_jacobian(
        tnsr::I<double, 3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto inv_jac2 = rotated3d_base->inv_jacobian(
        tnsr::I<double, 3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto expected_inv_jac = first_rotated3d.inv_jacobian(coord);
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
        CHECK(inv_jac2.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }
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
    CHECK((tnsr::I<double, 1, Frame::Grid>(1.0 / i + -0.5))[0] ==
          approx(map(tnsr::I<double, 1, Frame::Logical>{2.0 / i + -1.0})[0]));
    CHECK((tnsr::I<double, 1, Frame::Logical>(2.0 / i + -1.0))[0] ==
          approx(map.inverse(tnsr::I<double, 1, Frame::Grid>{1.0 / i + -0.5})
                     .get()[0]));

    CHECK(approx(map.inv_jacobian(
                        tnsr::I<double, 1, Frame::Logical>{2.0 / i + -1.0})
                     .get(0, 0)) == 2.0);
    CHECK(
        approx(map.jacobian(tnsr::I<double, 1, Frame::Logical>{2.0 / i + -1.0})
                   .get(0, 0)) == 0.5);
  }

  // Test 2D
  const auto prod_map2d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map_2d{affine_map{-1.0, 1.0, 0.0, 2.0},
                    affine_map{0.0, 2.0, -0.5, 0.5}},
      affine_map_2d{affine_map{0.0, 2.0, 2.0, 6.0},
                    affine_map{-0.5, 0.5, 0.0, 8.0}});
  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    const auto mapped_point = prod_map2d(
        tnsr::I<double, 2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    const auto expected_mapped_point =
        tnsr::I<double, 2, Frame::Grid>{{{4.0 / i + 2.0, 8.0 / i + 0.0}}};
    CHECK(get<0>(expected_mapped_point) == approx(get<0>(mapped_point)));
    CHECK(get<1>(expected_mapped_point) == approx(get<1>(mapped_point)));

    const auto inv_mapped_point = prod_map2d
                                      .inverse(tnsr::I<double, 2, Frame::Grid>{
                                          {{4.0 / i + 2.0, 8.0 / i + 0.0}}})
                                      .get();
    const auto expected_inv_mapped_point =
        tnsr::I<double, 2, Frame::Grid>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}};
    CHECK(get<0>(expected_inv_mapped_point) ==
          approx(get<0>(inv_mapped_point)));
    CHECK(get<1>(expected_inv_mapped_point) ==
          approx(get<1>(inv_mapped_point)));

    const auto inv_jac = prod_map2d.inv_jacobian(
        tnsr::I<double, 2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    CHECK(0.5 == approx(get<0, 0>(inv_jac)));
    CHECK(0.0 == approx(get<1, 0>(inv_jac)));
    CHECK(0.0 == approx(get<0, 1>(inv_jac)));
    CHECK(0.25 == approx(get<1, 1>(inv_jac)));

    const auto jac = prod_map2d.jacobian(
        tnsr::I<double, 2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    CHECK(2.0 == approx(get<0, 0>(jac)));
    CHECK(0.0 == approx(get<1, 0>(jac)));
    CHECK(0.0 == approx(get<0, 1>(jac)));
    CHECK(4.0 == approx(get<1, 1>(jac)));
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
    const auto mapped_point = prod_map3d(tnsr::I<double, 3, Frame::Logical>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    const auto expected_mapped_point = tnsr::I<double, 3, Frame::Grid>{
        {{4.0 / i + 2.0, 8.0 / i + 0.0, 3.0 + 20.0 / i}}};
    CHECK(get<0>(expected_mapped_point) == approx(get<0>(mapped_point)));
    CHECK(get<1>(expected_mapped_point) == approx(get<1>(mapped_point)));
    CHECK(get<2>(expected_mapped_point) == approx(get<2>(mapped_point)));

    const auto inv_mapped_point =
        prod_map3d
            .inverse(tnsr::I<double, 3, Frame::Grid>{
                {{4.0 / i + 2.0, 8.0 / i + 0.0, 3.0 + 20.0 / i}}})
            .get();
    const auto expected_inv_mapped_point = tnsr::I<double, 3, Frame::Grid>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}};
    CHECK(get<0>(expected_inv_mapped_point) ==
          approx(get<0>(inv_mapped_point)));
    CHECK(get<1>(expected_inv_mapped_point) ==
          approx(get<1>(inv_mapped_point)));
    CHECK(get<2>(expected_inv_mapped_point) ==
          approx(get<2>(inv_mapped_point)));

    const auto inv_jac =
        prod_map3d.inv_jacobian(tnsr::I<double, 3, Frame::Logical>{
            {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    CHECK(0.5 == approx(get<0, 0>(inv_jac)));
    CHECK(0.0 == approx(get<1, 0>(inv_jac)));
    CHECK(0.0 == approx(get<0, 1>(inv_jac)));
    CHECK(0.25 == approx(get<1, 1>(inv_jac)));
    CHECK(0.0 == approx(get<0, 2>(inv_jac)));
    CHECK(0.0 == approx(get<1, 2>(inv_jac)));
    CHECK(0.0 == approx(get<2, 0>(inv_jac)));
    CHECK(0.0 == approx(get<2, 1>(inv_jac)));
    CHECK(0.1 == approx(get<2, 2>(inv_jac)));

    const auto jac = prod_map3d.jacobian(tnsr::I<double, 3, Frame::Logical>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    CHECK(2.0 == approx(get<0, 0>(jac)));
    CHECK(0.0 == approx(get<1, 0>(jac)));
    CHECK(0.0 == approx(get<0, 1>(jac)));
    CHECK(4.0 == approx(get<1, 1>(jac)));
    CHECK(0.0 == approx(get<0, 2>(jac)));
    CHECK(0.0 == approx(get<1, 2>(jac)));
    CHECK(0.0 == approx(get<2, 0>(jac)));
    CHECK(0.0 == approx(get<2, 1>(jac)));
    CHECK(10.0 == approx(get<2, 2>(jac)));
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
    CHECK((make_array<double, 2>(double_rotated2d(
              tnsr::I<double, 2, Frame::Logical>{{{coord[0], coord[1]}}}))) ==
          second_rotated2d(first_rotated2d(coord)));
    CHECK((make_array<double, 2>(double_rotated2d
                                     .inverse(tnsr::I<double, 2, Frame::Grid>{
                                         {{coord[0], coord[1]}}})
                                     .get())) ==
          first_rotated2d.inverse(second_rotated2d.inverse(coord).get()).get());

    const auto jac = double_rotated2d.jacobian(
        tnsr::I<double, 2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto expected_jac = compose_jacobians(
        first_rotated2d, second_rotated2d, gsl::at(coords2d, i));
    CHECK_ITERABLE_APPROX(jac, expected_jac);

    const auto inv_jac = double_rotated2d.inv_jacobian(
        tnsr::I<double, 2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto expected_inv_jac = compose_inv_jacobians(
        first_rotated2d, second_rotated2d, gsl::at(coords2d, i));
    CHECK_ITERABLE_APPROX(inv_jac, expected_inv_jac);
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
    CHECK((make_array<double, 3>(
              double_rotated3d(tnsr::I<double, 3, Frame::Logical>{
                  {{coord[0], coord[1], coord[2]}}}))) ==
          second_rotated3d(first_rotated3d(coord)));
    CHECK((make_array<double, 3>(double_rotated3d
                                     .inverse(tnsr::I<double, 3, Frame::Grid>{
                                         {{coord[0], coord[1], coord[2]}}})
                                     .get())) ==
          first_rotated3d.inverse(second_rotated3d.inverse(coord).get()).get());

    const auto jac = double_rotated3d.jacobian(
        tnsr::I<double, 3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto expected_jac = compose_jacobians(
        first_rotated3d, second_rotated3d, gsl::at(coords3d, i));
    CHECK_ITERABLE_APPROX(jac, expected_jac);

    const auto inv_jac = double_rotated3d.inv_jacobian(
        tnsr::I<double, 3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto expected_inv_jac = compose_inv_jacobians(
        first_rotated3d, second_rotated3d, gsl::at(coords3d, i));
    CHECK_ITERABLE_APPROX(inv_jac, expected_inv_jac);
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
  }
}

void test_coordinate_map_with_rotation_wedge() {
  INFO("Coordinate map with rotation wedge");
  using Rotate = CoordinateMaps::Rotation<2>;
  using Wedge2D = CoordinateMaps::Wedge2D;

  const auto first_map = Rotate(2.);
  const auto second_map =
      Wedge2D(3., 7., 0.0, 1.0,
              OrientationMap<2>{std::array<Direction<2>, 2>{
                  {Direction<2>::lower_eta(), Direction<2>::lower_xi()}}},
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
  test_serialization(giant_identity_map);

  const auto wedge = make_coordinate_map<Frame::Logical, Frame::Inertial>(
      CoordinateMaps::Wedge3D(0.2, 4.0, OrientationMap<3>{}, 0.0, 1.0, true));
  const auto wedge_composed_with_giant_identity =
      make_coordinate_map<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::Wedge3D(0.2, 4.0, OrientationMap<3>{}, 0.0, 1.0,
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

  for (size_t i = 1; i < 11; ++i) {
    const auto source_point = tnsr::I<double, 3, Frame::Logical>{
        {{-1.0 + 2.0 / i, -1.0 + 2.0 / i, -1.0 + 2.0 / i}}};
    const auto mapped_point = tnsr::I<double, 3, Frame::Inertial>{
        {{-1.0 + 2.0 / i, -1.0 + 2.0 / i, -1.0 + 2.0 / i}}};
    CHECK(get<0>(mapped_point) == get<0>(giant_identity_map(source_point)));
    CHECK(get<1>(mapped_point) == get<1>(giant_identity_map(source_point)));
    CHECK(get<2>(mapped_point) == get<2>(giant_identity_map(source_point)));
    CHECK(get<0>(source_point) ==
          get<0>(giant_identity_map.inverse(mapped_point).get()));
    CHECK(get<1>(source_point) ==
          get<1>(giant_identity_map.inverse(mapped_point).get()));
    CHECK(get<2>(source_point) ==
          get<2>(giant_identity_map.inverse(mapped_point).get()));
    const auto wedge_mapped_point = wedge(source_point);
    CHECK(get<0>(wedge_mapped_point) ==
          get<0>(wedge_composed_with_giant_identity(source_point)));
    CHECK(get<1>(wedge_mapped_point) ==
          get<1>(wedge_composed_with_giant_identity(source_point)));
    CHECK(get<2>(wedge_mapped_point) ==
          get<2>(wedge_composed_with_giant_identity(source_point)));
    CHECK(get<0>(wedge.inverse(wedge_mapped_point).get()) ==
          get<0>(wedge_composed_with_giant_identity.inverse(wedge_mapped_point)
                     .get()));
    CHECK(get<1>(wedge.inverse(wedge_mapped_point).get()) ==
          get<1>(wedge_composed_with_giant_identity.inverse(wedge_mapped_point)
                     .get()));
    CHECK(get<2>(wedge.inverse(wedge_mapped_point).get()) ==
          get<2>(wedge_composed_with_giant_identity.inverse(wedge_mapped_point)
                     .get()));

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
}
}  // namespace domain
