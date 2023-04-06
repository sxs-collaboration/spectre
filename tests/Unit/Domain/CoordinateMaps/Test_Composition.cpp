// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <memory>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace domain::CoordinateMaps {

namespace {
void test_composition() {
  INFO("Composition");

  using Affine2D = ProductOf2Maps<Affine, Affine>;
  register_classes_with_charm<
      CoordinateMap<Frame::ElementLogical, Frame::BlockLogical, Affine2D>,
      CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine2D>>();

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> logical_dist{-1., 1.};

  // Domain is [0, 1] x [1, 3], and first dim is split in half, so element is
  // [0, 0.5] x [1, 3]
  const ElementId<2> element_id{0, {{{1, 0}, {0, 0}}}};
  // Testing template deduction here
  const Composition map{
      element_to_block_logical_map(element_id),
      std::make_unique<
          CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine2D>>(
          Affine2D{{-1., 1., 0., 1.}, {-1., 1., 1., 3.}})};

  {
    INFO("Semantics");
    test_serialization(map);
    test_copy_semantics(map);
    auto move_map = map;
    test_move_semantics(std::move(move_map), map);
  }

  {
    INFO("Properties");
    const auto& maps = map.maps();
    CHECK(*get<0>(maps) == *element_to_block_logical_map(element_id));
    CHECK((map.get_component<Frame::ElementLogical, Frame::BlockLogical>() ==
           *get<0>(maps)));
  }

  CHECK_FALSE(map.is_identity());
  CHECK_FALSE(map.inv_jacobian_is_time_dependent());
  CHECK_FALSE(map.jacobian_is_time_dependent());

  const auto xi =
      make_with_random_values<tnsr::I<DataVector, 2, Frame::ElementLogical>>(
          make_not_null(&generator), make_not_null(&logical_dist),
          DataVector(5));
  const auto x = map(xi);
  const auto jacobian = map.jacobian(xi);
  const auto inv_jacobian = map.inv_jacobian(xi);
  CHECK_ITERABLE_APPROX(get<0>(x), (get<0>(xi) + 1.) * 0.25);
  CHECK_ITERABLE_APPROX(get<1>(x), (get<1>(xi) + 2.));
  CHECK_ITERABLE_APPROX((get<0, 0>(jacobian)), DataVector(5, 0.25));
  CHECK_ITERABLE_APPROX((get<0, 0>(inv_jacobian)), DataVector(5, 4.));
  CHECK_ITERABLE_APPROX((get<1, 1>(jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 1>(inv_jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 0>(jacobian)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<1, 0>(inv_jacobian)), DataVector(5, 0.));
  const auto x_target = tnsr::I<double, 2, Frame::Inertial>{{{0.5, 1.}}};
  const auto inv = map.inverse(x_target);
  REQUIRE(inv.has_value());
  CHECK(get<0>(*inv) == approx(1.));
  CHECK(get<1>(*inv) == approx(-1.));
}

void test_identity() {
  INFO("Identity");

  using Affine3D = ProductOf3Maps<Affine, Affine, Affine>;
  register_classes_with_charm<
      CoordinateMap<Frame::ElementLogical, Frame::BlockLogical, Affine3D>,
      CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D>>();

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> logical_dist{-1., 1.};

  const ElementId<3> element_id{0, {{{0, 0}, {0, 0}, {0, 0}}}};
  const Composition<
      tmpl::list<Frame::ElementLogical, Frame::BlockLogical, Frame::Inertial>,
      3>
      map{element_to_block_logical_map(element_id),
          std::make_unique<
              CoordinateMap<Frame::BlockLogical, Frame::Inertial, Affine3D>>(
              Affine3D{
                  {-1., 1., -1., 1.}, {-1., 1., -1., 1.}, {-1., 1., -1., 1.}})};

  CHECK(map.is_identity());

  const auto xi =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::ElementLogical>>(
          make_not_null(&generator), make_not_null(&logical_dist),
          DataVector(5));
  const auto x = map(xi);
  const auto jacobian = map.jacobian(xi);
  const auto inv_jacobian = map.inv_jacobian(xi);
  CHECK_ITERABLE_APPROX(get<0>(x), get<0>(xi));
  CHECK_ITERABLE_APPROX(get<1>(x), get<1>(xi));
  CHECK_ITERABLE_APPROX(get<2>(x), get<2>(xi));
  CHECK_ITERABLE_APPROX((get<0, 0>(jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<0, 0>(inv_jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 1>(jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 1>(inv_jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<2, 2>(jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<2, 2>(inv_jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 0>(jacobian)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<1, 0>(inv_jacobian)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<2, 0>(jacobian)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<2, 0>(inv_jacobian)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<2, 1>(jacobian)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<2, 1>(inv_jacobian)), DataVector(5, 0.));
  const auto x_target = tnsr::I<double, 3, Frame::Inertial>{{{0.5, 1., 0.}}};
  const auto inv = map.inverse(x_target);
  REQUIRE(inv.has_value());
  CHECK(get<0>(*inv) == approx(0.5));
  CHECK(get<1>(*inv) == approx(1.));
  CHECK(get<2>(*inv) == approx(0.));
}

void test_3d() {
  INFO("3D");

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> logical_dist{-1., 1.};

  const ElementId<3> element_id{0, {{{1, 0}, {1, 0}, {2, 0}}}};
  const Composition map{
      element_to_block_logical_map(element_id),
      std::make_unique<
          CoordinateMap<Frame::BlockLogical, Frame::Inertial, Wedge<3>>>(
          Wedge<3>{1., 3., 1., 1., {}, true})};

  // Check some points that are easy to calculate
  // - On inner boundary
  CHECK(get(magnitude(map(tnsr::I<double, 3, Frame::ElementLogical>{
            {logical_dist(generator), logical_dist(generator), -1.}}))) ==
        approx(1.));
  // - On outer boundary of first radial element
  CHECK(get(magnitude(map(tnsr::I<double, 3, Frame::ElementLogical>{
            {logical_dist(generator), logical_dist(generator), 1.}}))) ==
        approx(1.5));

  // Check Jacobian is consistent with inverse
  const auto xi =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::ElementLogical>>(
          make_not_null(&generator), make_not_null(&logical_dist),
          DataVector(5));
  const auto jacobian = map.jacobian(xi);
  const auto inv_jacobian = map.inv_jacobian(xi);
  const auto identity = tenex::evaluate<ti::I, ti::j>(
      jacobian(ti::I, ti::k) * inv_jacobian(ti::K, ti::j));
  CHECK_ITERABLE_APPROX((get<0, 0>(identity)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 1>(identity)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<2, 2>(identity)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 0>(identity)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<2, 0>(identity)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<2, 1>(identity)), DataVector(5, 0.));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Composition", "[Domain][Unit]") {
  test_composition();
  test_identity();
  test_3d();
}

}  // namespace domain::CoordinateMaps
