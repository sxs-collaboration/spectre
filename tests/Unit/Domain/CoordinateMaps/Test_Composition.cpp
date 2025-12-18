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
#include "Domain/CoordinateMaps/EquatorialCompression.hpp"
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
#ifdef SPECTRE_AUTODIFF
// This test helper function computes the inverse Hessian by
// auto-differentiating through the inv_jacobian function, which gives the
// derivatives of inv_jacobian with respect to the source coordinates, i.e.
// \frac{\partial}{\partial\xi^k}\left(\frac{\partial\xi^i}{\partial
// x^j}\right). The inverse Hessian is then given by the chain rule
// \frac{\partial^2\xi^i}{\partial x^j \partial x^k} =
// \frac{\partial\xi^l}{\partial x^k} *
// \frac{\partial}{\partial\xi^l}\left(\frac{\partial\xi^i}{\partial
// x^j}\right).
template <typename DataType, typename Map, size_t Dim>
InverseHessian<DataType, Dim, Frame::ElementLogical, Frame::Inertial>
inv_hessian_helper(
    const Map& coordinate_map,
    const tnsr::I<DataType, Dim, Frame::ElementLogical>& source_point) {
  using SecondOrderDualNum = autodiff::HigherOrderDual<2, double>;

  const size_t num_pts = get<0>(source_point).size();
  ::InverseHessian<DataType, Dim, Frame::ElementLogical, Frame::Inertial>
      inverse_hessian{num_pts};

  for (size_t pts_index = 0; pts_index < num_pts; ++pts_index) {
    tnsr::I<SecondOrderDualNum, Dim, Frame::ElementLogical> dual_source_coords;

    if constexpr (std::is_same_v<DataType, double>) {
      [&]<std::size_t... Ins>(std::index_sequence<Ins...>) {
        ((get<Ins>(dual_source_coords) = get<Ins>(source_point)), ...);
      }
      (std::make_index_sequence<Dim>{});
    } else {
      [&]<std::size_t... Ins>(std::index_sequence<Ins...>) {
        ((get<Ins>(dual_source_coords) =
              gsl::at(get<Ins>(source_point), pts_index)),
         ...);
      }
      (std::make_index_sequence<Dim>{});
    }

    for (size_t k = 0; k < Dim; ++k) {
      for (size_t i = 0; i < Dim; ++i) {
        for (size_t j = i; j < Dim; ++j) {
          double inv_hessian_kij{0.0};

          for (size_t l = 0; l < Dim; ++l) {
            autodiff::seed<1>(dual_source_coords.get(l), 1.0);
            for (size_t n = 1; n < Dim; ++n) {
              autodiff::seed<1>(dual_source_coords.get((l + n) % Dim), 0.0);
            }

            const auto dual_inverse_jac =
                coordinate_map.inv_jacobian(dual_source_coords);
            const auto inv_jac_lj = autodiff::val(dual_inverse_jac.get(l, j));
            const auto deriv_kli =
                autodiff::derivative<1>(dual_inverse_jac.get(k, i));
            inv_hessian_kij += inv_jac_lj * deriv_kli;
          }

          if constexpr (std::is_same_v<DataType, double>) {
            inverse_hessian.get(k, i, j) = inv_hessian_kij;
          } else {
            inverse_hessian.get(k, i, j)[pts_index] = inv_hessian_kij;
          }
        }
      }
    }
  }
  return inverse_hessian;
}
#endif  // SPECTRE_AUTODIFF

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
  CHECK(map.function_of_time_names().empty());

  const auto xi =
      make_with_random_values<tnsr::I<DataVector, 2, Frame::ElementLogical>>(
          make_not_null(&generator), make_not_null(&logical_dist),
          DataVector(5));
  const auto x = map(xi);
  const auto jacobian = map.jacobian(xi);
  const auto inv_jacobian = map.inv_jacobian(xi);
#ifdef SPECTRE_AUTODIFF
  const auto inv_hessian = map.inv_hessian(xi, inv_jacobian);
#endif  // SPECTRE_AUTODIFF
  CHECK_ITERABLE_APPROX(get<0>(x), (get<0>(xi) + 1.) * 0.25);
  CHECK_ITERABLE_APPROX(get<1>(x), (get<1>(xi) + 2.));
  CHECK_ITERABLE_APPROX((get<0, 0>(jacobian)), DataVector(5, 0.25));
  CHECK_ITERABLE_APPROX((get<0, 0>(inv_jacobian)), DataVector(5, 4.));
  CHECK_ITERABLE_APPROX((get<1, 1>(jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 1>(inv_jacobian)), DataVector(5, 1.));
  CHECK_ITERABLE_APPROX((get<1, 0>(jacobian)), DataVector(5, 0.));
  CHECK_ITERABLE_APPROX((get<1, 0>(inv_jacobian)), DataVector(5, 0.));
#ifdef SPECTRE_AUTODIFF
  for (const auto& component : inv_hessian) {
    CHECK_ITERABLE_APPROX(component, DataVector(5, 0.0));
  }
#endif  // SPECTRE_AUTODIFF
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
  CHECK(map.function_of_time_names().empty());

  const auto xi =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::ElementLogical>>(
          make_not_null(&generator), make_not_null(&logical_dist),
          DataVector(5));
  const auto x = map(xi);
  const auto jacobian = map.jacobian(xi);
  const auto inv_jacobian = map.inv_jacobian(xi);
#ifdef SPECTRE_AUTODIFF
  const auto inv_hessian = map.inv_hessian(xi, inv_jacobian);
#endif  // SPECTRE_AUTODIFF
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
#ifdef SPECTRE_AUTODIFF
  for (const auto& component : inv_hessian) {
    CHECK_ITERABLE_APPROX(component, DataVector(5, 0.0));
  }
#endif  // SPECTRE_AUTODIFF
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
          Wedge<3>{1., 3., 1., 1., OrientationMap<3>::create_aligned(), true})};
  CHECK(map.function_of_time_names().empty());

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

#ifdef SPECTRE_AUTODIFF
  const auto inv_hessian = map.inv_hessian(xi, inv_jacobian);
  const auto alt_inv_hessian = inv_hessian_helper(map, xi);
  CHECK_ITERABLE_APPROX(inv_hessian, alt_inv_hessian);
#endif  // SPECTRE_AUTODIFF
}

void test_unsupported_autodiff() {
  INFO("Unsupported autodiff");

#ifdef SPECTRE_AUTODIFF
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<double> logical_dist{-1., 1.};

  const ElementId<3> element_id{0, {{{0, 0}, {0, 0}, {0, 0}}}};
  // EquatorialCompression does not support autodiff
  const Composition map{
      element_to_block_logical_map(element_id),
      std::make_unique<CoordinateMap<Frame::BlockLogical, Frame::Inertial,
                                     EquatorialCompression>>(
          EquatorialCompression{1.5, 2})};

  // Check that supports_hessian() correctly returns false
  CHECK_FALSE(map.supports_hessian());

  const auto xi =
      make_with_random_values<tnsr::I<double, 3, Frame::ElementLogical>>(
          make_not_null(&generator), make_not_null(&logical_dist),
          0.0);

  // Calling inv_hessian should trigger an error
  const auto inv_jacobian = map.inv_jacobian(xi);
  CHECK_THROWS_WITH(
      map.inv_hessian(xi, inv_jacobian),
      Catch::Matchers::ContainsSubstring(
          "At least one of the Maps does not support autodiff") &&
          Catch::Matchers::ContainsSubstring("EquatorialCompression"));

  // Also test with DataVector
  const auto xi_dv =
      make_with_random_values<tnsr::I<DataVector, 3, Frame::ElementLogical>>(
          make_not_null(&generator), make_not_null(&logical_dist),
          DataVector(5));

  const auto inv_jacobian_dv = map.inv_jacobian(xi_dv);
  CHECK_THROWS_WITH(
      map.inv_hessian(xi_dv, inv_jacobian_dv),
      Catch::Matchers::ContainsSubstring(
          "At least one of the Maps does not support autodiff") &&
          Catch::Matchers::ContainsSubstring("EquatorialCompression"));
#endif  // SPECTRE_AUTODIFF
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordinateMaps.Composition", "[Domain][Unit]") {
  test_composition();
  test_identity();
  test_3d();
  test_unsupported_autodiff();
}

}  // namespace domain::CoordinateMaps
