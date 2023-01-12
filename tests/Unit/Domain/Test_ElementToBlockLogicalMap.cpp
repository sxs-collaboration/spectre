// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <random>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Utilities/Gsl.hpp"

namespace domain {

SPECTRE_TEST_CASE("Unit.Domain.ElementToBlockLogicalMap", "[Domain][Unit]") {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> logical_dist{-1., 1.};

  {
    INFO("1D");
    const ElementId<1> element_id{0, {{{1, 0}}}};
    const auto map = element_to_block_logical_map(element_id);

    const auto xi =
        make_with_random_values<tnsr::I<DataVector, 1, Frame::ElementLogical>>(
            make_not_null(&generator), make_not_null(&logical_dist),
            DataVector(5));
    const auto x = (*map)(xi);
    const auto jacobian = map->jacobian(xi);
    const auto inv_jacobian = map->inv_jacobian(xi);
    CHECK_ITERABLE_APPROX(get<0>(x), get<0>(xi) * 0.5 - 0.5);
    CHECK_ITERABLE_APPROX((get<0, 0>(jacobian)), DataVector(5, 0.5));
    CHECK_ITERABLE_APPROX((get<0, 0>(inv_jacobian)), DataVector(5, 2.));
    const auto x_target = tnsr::I<double, 1, Frame::BlockLogical>{{{0.}}};
    const auto inv = map->inverse(x_target);
    REQUIRE(inv.has_value());
    CHECK(get<0>(*inv) == approx(1.));
  }

  {
    INFO("2D");
    const ElementId<2> element_id{0, {{{1, 0}, {0, 0}}}};
    const auto map = element_to_block_logical_map(element_id);

    const auto xi =
        make_with_random_values<tnsr::I<DataVector, 2, Frame::ElementLogical>>(
            make_not_null(&generator), make_not_null(&logical_dist),
            DataVector(5));
    const auto x = (*map)(xi);
    const auto jacobian = map->jacobian(xi);
    const auto inv_jacobian = map->inv_jacobian(xi);
    CHECK_ITERABLE_APPROX(get<0>(x), get<0>(xi) * 0.5 - 0.5);
    CHECK_ITERABLE_APPROX(get<1>(x), get<1>(xi));
    CHECK_ITERABLE_APPROX((get<0, 0>(jacobian)), DataVector(5, 0.5));
    CHECK_ITERABLE_APPROX((get<0, 0>(inv_jacobian)), DataVector(5, 2.));
    CHECK_ITERABLE_APPROX((get<1, 1>(jacobian)), DataVector(5, 1.));
    CHECK_ITERABLE_APPROX((get<1, 1>(inv_jacobian)), DataVector(5, 1.));
    CHECK_ITERABLE_APPROX((get<1, 0>(jacobian)), DataVector(5, 0.));
    CHECK_ITERABLE_APPROX((get<1, 0>(inv_jacobian)), DataVector(5, 0.));
    const auto x_target = tnsr::I<double, 2, Frame::BlockLogical>{{{0., -1.}}};
    const auto inv = map->inverse(x_target);
    REQUIRE(inv.has_value());
    CHECK(get<0>(*inv) == approx(1.));
    CHECK(get<1>(*inv) == approx(-1.));
  }

  {
    INFO("3D");
    const ElementId<3> element_id{0, {{{1, 0}, {0, 0}, {2, 1}}}};
    const auto map = element_to_block_logical_map(element_id);

    const auto xi =
        make_with_random_values<tnsr::I<DataVector, 3, Frame::ElementLogical>>(
            make_not_null(&generator), make_not_null(&logical_dist),
            DataVector(5));
    const auto x = (*map)(xi);
    const auto jacobian = map->jacobian(xi);
    const auto inv_jacobian = map->inv_jacobian(xi);
    CHECK_ITERABLE_APPROX(get<0>(x), get<0>(xi) * 0.5 - 0.5);
    CHECK_ITERABLE_APPROX(get<1>(x), get<1>(xi));
    CHECK_ITERABLE_APPROX(get<2>(x), get<2>(xi) * 0.25 - 0.25);
    CHECK_ITERABLE_APPROX((get<0, 0>(jacobian)), DataVector(5, 0.5));
    CHECK_ITERABLE_APPROX((get<0, 0>(inv_jacobian)), DataVector(5, 2.));
    CHECK_ITERABLE_APPROX((get<1, 1>(jacobian)), DataVector(5, 1.));
    CHECK_ITERABLE_APPROX((get<1, 1>(inv_jacobian)), DataVector(5, 1.));
    CHECK_ITERABLE_APPROX((get<2, 2>(jacobian)), DataVector(5, 0.25));
    CHECK_ITERABLE_APPROX((get<2, 2>(inv_jacobian)), DataVector(5, 4.));
    CHECK_ITERABLE_APPROX((get<1, 0>(jacobian)), DataVector(5, 0.));
    CHECK_ITERABLE_APPROX((get<1, 0>(inv_jacobian)), DataVector(5, 0.));
    CHECK_ITERABLE_APPROX((get<2, 0>(jacobian)), DataVector(5, 0.));
    CHECK_ITERABLE_APPROX((get<2, 0>(inv_jacobian)), DataVector(5, 0.));
    CHECK_ITERABLE_APPROX((get<2, 1>(jacobian)), DataVector(5, 0.));
    CHECK_ITERABLE_APPROX((get<2, 1>(inv_jacobian)), DataVector(5, 0.));
    const auto x_target =
        tnsr::I<double, 3, Frame::BlockLogical>{{{0., -1., -0.5}}};
    const auto inv = map->inverse(x_target);
    REQUIRE(inv.has_value());
    CHECK(get<0>(*inv) == approx(1.));
    CHECK(get<1>(*inv) == approx(-1.));
    CHECK(get<2>(*inv) == approx(-1.));
  }
}
}  // namespace domain
