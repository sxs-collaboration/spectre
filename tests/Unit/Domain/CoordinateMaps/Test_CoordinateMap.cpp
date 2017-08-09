// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/AffineMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_single_coordinate_map() {
  using affine_map1d = CoordinateMaps::AffineMap;

  const auto affine1d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map1d{-1.0, 1.0, 2.0, 8.0});
  const auto first_affine1d = affine_map1d{-1.0, 1.0, 2.0, 8.0};

  std::array<std::array<double, 1>, 4> coords1d{
      {{{0.1}}, {{-8.2}}, {{5.7}}, {{2.9}}}};

  for (const auto& coord : coords1d) {
    CHECK((make_array<double, 1>(affine1d(Point<1, Frame::Logical>{
              {{coord[0]}}}))) == first_affine1d(coord));
    CHECK((make_array<double, 1>(affine1d.inverse(Point<1, Frame::Grid>{
              {{coord[0]}}}))) == first_affine1d.inverse(coord));

    const auto jac = affine1d.jacobian(Point<1, Frame::Logical>{{{coord[0]}}});
    const auto expected_jac = first_affine1d.jacobian(coord);
    CHECK(jac.get(0, 0) == expected_jac.get(0, 0));

    const auto inv_jac =
        affine1d.inv_jacobian(Point<1, Frame::Logical>{{{coord[0]}}});
    const auto expected_inv_jac = first_affine1d.inv_jacobian(coord);
    CHECK(inv_jac.get(0, 0) == expected_inv_jac.get(0, 0));
  }

  using rotate2d = CoordinateMaps::Rotation<2>;

  const auto rotated2d =
      make_coordinate_map<Frame::Logical, Frame::Grid>(rotate2d{M_PI_4});
  const auto first_rotated2d = rotate2d{M_PI_4};

  std::array<std::array<double, 2>, 4> coords2d{
      {{{0.1, 2.8}}, {{-8.2, 2.8}}, {{5.7, -4.9}}, {{2.9, 3.4}}}};

  for (const auto& coord : coords2d) {
    CHECK((make_array<double, 2>(rotated2d(Point<2, Frame::Logical>{
              {{coord[0], coord[1]}}}))) == first_rotated2d(coord));
    CHECK((make_array<double, 2>(rotated2d.inverse(Point<2, Frame::Grid>{
              {{coord[0], coord[1]}}}))) == first_rotated2d.inverse(coord));

    const auto jac =
        rotated2d.jacobian(Point<2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto expected_jac = first_rotated2d.jacobian(coord);
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = rotated2d.inv_jacobian(
        Point<2, Frame::Logical>{{{coord[0], coord[1]}}});
    const auto expected_inv_jac = first_rotated2d.inv_jacobian(coord);
    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }

  using rotate3d = CoordinateMaps::Rotation<3>;

  const auto rotated3d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      rotate3d{M_PI_4, M_PI_4, M_PI_2});
  const auto first_rotated3d = rotate3d{M_PI_4, M_PI_4, M_PI_2};

  std::array<std::array<double, 3>, 4> coords3d{{{{0.1, 2.8, 9.3}},
                                                 {{-8.2, 2.8, -9.7}},
                                                 {{5.7, -4.9, 8.1}},
                                                 {{2.9, 3.4, -7.8}}}};

  for (const auto& coord : coords3d) {
    CHECK((make_array<double, 3>(rotated3d(Point<3, Frame::Logical>{
              {{coord[0], coord[1], coord[2]}}}))) == first_rotated3d(coord));
    CHECK((make_array<double, 3>(rotated3d.inverse(
              Point<3, Frame::Grid>{{{coord[0], coord[1], coord[2]}}}))) ==
          first_rotated3d.inverse(coord));

    const auto jac = rotated3d.jacobian(
        Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto expected_jac = first_rotated3d.jacobian(coord);
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = rotated3d.inv_jacobian(
        Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});
    const auto expected_inv_jac = first_rotated3d.inv_jacobian(coord);
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }
}

void test_coordinate_map_with_affine_map() {
  using affine_map = CoordinateMaps::AffineMap;
  using affine_map_2d = CoordinateMaps::ProductOf2Maps<affine_map, affine_map>;
  using affine_map_3d =
      CoordinateMaps::ProductOf3Maps<affine_map, affine_map, affine_map>;

  constexpr size_t number_of_points_checked = 10;
  Approx approx = Approx::custom().epsilon(1.0e-14);

  // Test 1D
  const auto map = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map{-1.0, 1.0, 0.0, 2.3}, affine_map{0.0, 2.3, -0.5, 0.5});
  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    CHECK((Point<1, Frame::Grid>(1.0 / i + -0.5))[0] ==
          approx(map(Point<1, Frame::Logical>{2.0 / i + -1.0})[0]));
    CHECK((Point<1, Frame::Logical>(2.0 / i + -1.0))[0] ==
          approx(map.inverse(Point<1, Frame::Grid>{1.0 / i + -0.5})[0]));

    CHECK(approx(map.inv_jacobian(Point<1, Frame::Logical>{2.0 / i + -1.0})
                     .get(0, 0)) == 2.0);
    CHECK(approx(map.jacobian(Point<1, Frame::Logical>{2.0 / i + -1.0})
                     .get(0, 0)) == 0.5);
  }

  // Test 2D
  const auto prod_map2d = make_coordinate_map<Frame::Logical, Frame::Grid>(
      affine_map_2d{affine_map{-1.0, 1.0, 0.0, 2.0},
                    affine_map{0.0, 2.0, -0.5, 0.5}},
      affine_map_2d{affine_map{0.0, 2.0, 2.0, 6.0},
                    affine_map{-0.5, 0.5, 0.0, 8.0}});
  for (size_t i = 1; i < number_of_points_checked + 1; ++i) {
    const auto mapped_point =
        prod_map2d(Point<2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    const auto expected_mapped_point =
        Point<2, Frame::Grid>{{{4.0 / i + 2.0, 8.0 / i + 0.0}}};
    CHECK(expected_mapped_point.template get<0>() ==
          approx(mapped_point.template get<0>()));
    CHECK(expected_mapped_point.template get<1>() ==
          approx(mapped_point.template get<1>()));

    const auto inv_mapped_point = prod_map2d.inverse(
        Point<2, Frame::Grid>{{{4.0 / i + 2.0, 8.0 / i + 0.0}}});
    const auto expected_inv_mapped_point =
        Point<2, Frame::Grid>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}};
    CHECK(expected_inv_mapped_point.template get<0>() ==
          approx(inv_mapped_point.template get<0>()));
    CHECK(expected_inv_mapped_point.template get<1>() ==
          approx(inv_mapped_point.template get<1>()));

    const auto inv_jac = prod_map2d.inv_jacobian(
        Point<2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    CHECK(0.5 == approx(inv_jac.template get<0, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<1, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<0, 1>()));
    CHECK(0.25 == approx(inv_jac.template get<1, 1>()));

    const auto jac = prod_map2d.jacobian(
        Point<2, Frame::Logical>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i}}});
    CHECK(2.0 == approx(jac.template get<0, 0>()));
    CHECK(0.0 == approx(jac.template get<1, 0>()));
    CHECK(0.0 == approx(jac.template get<0, 1>()));
    CHECK(4.0 == approx(jac.template get<1, 1>()));
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
    const auto mapped_point = prod_map3d(Point<3, Frame::Logical>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    const auto expected_mapped_point =
        Point<3, Frame::Grid>{{{4.0 / i + 2.0, 8.0 / i + 0.0, 3.0 + 20.0 / i}}};
    CHECK(expected_mapped_point.template get<0>() ==
          approx(mapped_point.template get<0>()));
    CHECK(expected_mapped_point.template get<1>() ==
          approx(mapped_point.template get<1>()));
    CHECK(expected_mapped_point.template get<2>() ==
          approx(mapped_point.template get<2>()));

    const auto inv_mapped_point = prod_map3d.inverse(Point<3, Frame::Grid>{
        {{4.0 / i + 2.0, 8.0 / i + 0.0, 3.0 + 20.0 / i}}});
    const auto expected_inv_mapped_point =
        Point<3, Frame::Grid>{{{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}};
    CHECK(expected_inv_mapped_point.template get<0>() ==
          approx(inv_mapped_point.template get<0>()));
    CHECK(expected_inv_mapped_point.template get<1>() ==
          approx(inv_mapped_point.template get<1>()));
    CHECK(expected_inv_mapped_point.template get<2>() ==
          approx(inv_mapped_point.template get<2>()));

    const auto inv_jac = prod_map3d.inv_jacobian(Point<3, Frame::Logical>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    CHECK(0.5 == approx(inv_jac.template get<0, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<1, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<0, 1>()));
    CHECK(0.25 == approx(inv_jac.template get<1, 1>()));
    CHECK(0.0 == approx(inv_jac.template get<0, 2>()));
    CHECK(0.0 == approx(inv_jac.template get<1, 2>()));
    CHECK(0.0 == approx(inv_jac.template get<2, 0>()));
    CHECK(0.0 == approx(inv_jac.template get<2, 1>()));
    CHECK(0.1 == approx(inv_jac.template get<2, 2>()));

    const auto jac = prod_map3d.jacobian(Point<3, Frame::Logical>{
        {{-1.0 + 2.0 / i, 0.0 + 2.0 / i, 5.0 + 2.0 / i}}});
    CHECK(2.0 == approx(jac.template get<0, 0>()));
    CHECK(0.0 == approx(jac.template get<1, 0>()));
    CHECK(0.0 == approx(jac.template get<0, 1>()));
    CHECK(4.0 == approx(jac.template get<1, 1>()));
    CHECK(0.0 == approx(jac.template get<0, 2>()));
    CHECK(0.0 == approx(jac.template get<1, 2>()));
    CHECK(0.0 == approx(jac.template get<2, 0>()));
    CHECK(0.0 == approx(jac.template get<2, 1>()));
    CHECK(10.0 == approx(jac.template get<2, 2>()));
  }
}

void test_coordinate_map_with_rotation_map() {
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
    const auto coord = gsl::at(coords2d, i);
    CHECK((make_array<double, 2>(double_rotated2d(
              Point<2, Frame::Logical>{{{coord[0], coord[1]}}}))) ==
          second_rotated2d(first_rotated2d(coord)));
    CHECK((make_array<double, 2>(double_rotated2d.inverse(
              Point<2, Frame::Grid>{{{coord[0], coord[1]}}}))) ==
          first_rotated2d.inverse(second_rotated2d.inverse(coord)));

    const auto jac = double_rotated2d.jacobian(
        Point<2, Frame::Logical>{{{coord[0], coord[1]}}});

    const auto expected_jac = [&first_rotated2d, &second_rotated2d,
                               &coords2d](const size_t ii) {

      const auto first_jac = first_rotated2d.jacobian(gsl::at(coords2d, ii));
      auto second_jac =
          second_rotated2d.jacobian(first_rotated2d(gsl::at(coords2d, ii)));

      std::array<double, 2> temp{};
      for (size_t source = 0; source < 2; ++source) {
        for (size_t target = 0; target < 2; ++target) {
          gsl::at(temp, target) =
              second_jac.get(source, 0) * first_jac.get(0, target);
          for (size_t dummy = 1; dummy < 2; ++dummy) {
            gsl::at(temp, target) +=
                second_jac.get(source, dummy) * first_jac.get(dummy, target);
          }
        }
        for (size_t target = 0; target < 2; ++target) {
          second_jac.get(source, target) = gsl::at(temp, target);
        }
      }
      return second_jac;
    }(i);

    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        INFO(i);
        INFO(j);
        INFO(k);
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = double_rotated2d.inv_jacobian(
        Point<2, Frame::Logical>{{{coord[0], coord[1]}}});

    const auto expected_inv_jac = [&first_rotated2d, &second_rotated2d,
                                   &coords2d](const size_t ii) {
      auto first_inv_jac = first_rotated2d.inv_jacobian(gsl::at(coords2d, ii));

      const auto second_inv_jac =
          second_rotated2d.inv_jacobian(first_rotated2d(gsl::at(coords2d, ii)));

      std::array<double, 2> temp{};
      for (size_t source = 0; source < 2; ++source) {
        for (size_t target = 0; target < 2; ++target) {
          gsl::at(temp, target) =
              first_inv_jac.get(source, 0) * second_inv_jac.get(0, target);
          for (size_t dummy = 1; dummy < 2; ++dummy) {
            gsl::at(temp, target) += first_inv_jac.get(source, dummy) *
                                     second_inv_jac.get(dummy, target);
          }
        }
        for (size_t target = 0; target < 2; ++target) {
          first_inv_jac.get(source, target) = gsl::at(temp, target);
        }
      }

      return first_inv_jac;
    }(i);

    for (size_t j = 0; j < 2; ++j) {
      for (size_t k = 0; k < 2; ++k) {
        INFO(i);
        INFO(j);
        INFO(k);
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
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
    const auto coord = gsl::at(coords3d, i);
    CHECK((make_array<double, 3>(double_rotated3d(
              Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}}))) ==
          second_rotated3d(first_rotated3d(coord)));
    CHECK((make_array<double, 3>(double_rotated3d.inverse(
              Point<3, Frame::Grid>{{{coord[0], coord[1], coord[2]}}}))) ==
          first_rotated3d.inverse(second_rotated3d.inverse(coord)));

    const auto jac = double_rotated3d.jacobian(
        Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});

    const auto expected_jac = [&first_rotated3d, &second_rotated3d,
                               &coords3d](const size_t ii) {

      const auto first_jac = first_rotated3d.jacobian(gsl::at(coords3d, ii));
      auto second_jac =
          second_rotated3d.jacobian(first_rotated3d(gsl::at(coords3d, ii)));

      std::array<double, 3> temp{};
      for (size_t source = 0; source < 3; ++source) {
        for (size_t target = 0; target < 3; ++target) {
          gsl::at(temp, target) =
              second_jac.get(source, 0) * first_jac.get(0, target);
          for (size_t dummy = 1; dummy < 3; ++dummy) {
            gsl::at(temp, target) +=
                second_jac.get(source, dummy) * first_jac.get(dummy, target);
          }
        }
        for (size_t target = 0; target < 3; ++target) {
          second_jac.get(source, target) = gsl::at(temp, target);
        }
      }
      return second_jac;
    }(i);

    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        INFO(i);
        INFO(j);
        INFO(k);
        CHECK(jac.get(j, k) == expected_jac.get(j, k));
      }
    }

    const auto inv_jac = double_rotated3d.inv_jacobian(
        Point<3, Frame::Logical>{{{coord[0], coord[1], coord[2]}}});

    const auto expected_inv_jac = [&first_rotated3d, &second_rotated3d,
                                   &coords3d](const size_t ii) {
      auto first_inv_jac = first_rotated3d.inv_jacobian(gsl::at(coords3d, ii));

      const auto second_inv_jac =
          second_rotated3d.inv_jacobian(first_rotated3d(gsl::at(coords3d, ii)));

      std::array<double, 3> temp{};
      for (size_t source = 0; source < 3; ++source) {
        for (size_t target = 0; target < 3; ++target) {
          gsl::at(temp, target) =
              first_inv_jac.get(source, 0) * second_inv_jac.get(0, target);
          for (size_t dummy = 1; dummy < 3; ++dummy) {
            gsl::at(temp, target) += first_inv_jac.get(source, dummy) *
                                     second_inv_jac.get(dummy, target);
          }
        }
        for (size_t target = 0; target < 3; ++target) {
          first_inv_jac.get(source, target) = gsl::at(temp, target);
        }
      }

      return first_inv_jac;
    }(i);

    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = 0; k < 3; ++k) {
        INFO(i);
        INFO(j);
        INFO(k);
        CHECK(inv_jac.get(j, k) == expected_inv_jac.get(j, k));
      }
    }
  }

  // Check inequivalence operator
  CHECK_FALSE(double_rotated3d != double_rotated3d);
  CHECK(double_rotated3d == serialize_and_deserialize(double_rotated3d));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMap", "[Domain][Unit]") {
  test_single_coordinate_map();
  test_coordinate_map_with_affine_map();
  test_coordinate_map_with_rotation_map();
}
