// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"

namespace domain {
namespace {
// We will call with this function with arguments that are designed
// to fail in a certain way.
void cubic_scale_non_invertible(const double a0, const double b0,
                                const double outer_boundary) noexcept {
  double t = 4.0;
  constexpr size_t deriv_order = 2;

  const std::array<DataVector, deriv_order + 1> init_func_a{
      {{a0}, {0.0}, {0.0}}};
  const std::array<DataVector, deriv_order + 1> init_func_b{
      {{b0}, {0.0}, {0.0}}};

  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  std::unordered_map<std::string, FoftPtr> f_of_t_list{};
  f_of_t_list["expansion_a"] = std::make_unique<Polynomial>(t, init_func_a);
  f_of_t_list["expansion_b"] = std::make_unique<Polynomial>(t, init_func_b);

  const FoftPtr& expansion_a_base = f_of_t_list.at("expansion_a");
  const FoftPtr& expansion_b_base = f_of_t_list.at("expansion_b");

  const CoordMapsTimeDependent::CubicScale<1> scale_map(
      outer_boundary, "expansion_a", "expansion_b");
  const std::array<double, 1> point_xi{{19.2}};

  const std::array<double, 1> mapped_point{
      {point_xi[0] *
       (expansion_a_base->func(t)[0][0] +
        (expansion_b_base->func(t)[0][0] - expansion_a_base->func(t)[0][0]) *
            square(point_xi[0] / outer_boundary))}};

  // this call should fail.
  scale_map.inverse(mapped_point, t, f_of_t_list);
}

void test_boundaries() {
  INFO("Boundaries");
  static constexpr size_t deriv_order = 2;

  const auto run_tests = [](const double x0) {
    double t = -0.5;
    const double dt = 0.6;
    const double final_time = 4.0;

    const std::array<DataVector, deriv_order + 1> init_func_a{
        {{1.0}, {0.0}, {0.0}}};
    const std::array<DataVector, deriv_order + 1> init_func_b{
        {{0.99}, {0.0}, {0.0}}};

    using Polynomial =
        domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
    using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
    std::unordered_map<std::string, FoftPtr> f_of_t_list{};
    f_of_t_list["expansion_a"] = std::make_unique<Polynomial>(t, init_func_a);
    f_of_t_list["expansion_b"] = std::make_unique<Polynomial>(t, init_func_b);

    const FoftPtr& expansion_a_base = f_of_t_list.at("expansion_a");
    const FoftPtr& expansion_b_base = f_of_t_list.at("expansion_b");

    const double outer_boundary = 20.0;
    const CoordMapsTimeDependent::CubicScale<1> scale_map(
        outer_boundary, "expansion_a", "expansion_b");
    const std::array<double, 1> point_xi{{x0}};

    while (t < final_time) {
      const double a = expansion_a_base->func_and_deriv(t)[0][0];
      const double b = expansion_b_base->func_and_deriv(t)[0][0];

      const std::array<double, 1> mapped_point{
          {point_xi[0] * (a + (b - a) * square(point_xi[0] / outer_boundary))}};

      CHECK_ITERABLE_APPROX(scale_map(point_xi, t, f_of_t_list), mapped_point);
      REQUIRE(
          static_cast<bool>(scale_map.inverse(mapped_point, t, f_of_t_list)));
      CHECK_ITERABLE_APPROX(
          scale_map.inverse(mapped_point, t, f_of_t_list).get(), point_xi);
      t += dt;
    }
  };

  // test inverse at inner and outer boundary values
  run_tests(0.0);
  run_tests(20.0);
}

template <size_t Dim>
std::array<double, Dim> coords_single_root();

template <>
std::array<double, 1> coords_single_root() {
  return {{19.2}};
}

template <>
std::array<double, 2> coords_single_root() {
  return {{19.2, 0.1}};
}

template <>
std::array<double, 3> coords_single_root() {
  return {{19.2, 0.1, -1.1}};
}

template <size_t Dim>
std::array<double, Dim> coords_multi_root();

template <>
std::array<double, 1> coords_multi_root() {
  return {{5.0}};
}

template <>
std::array<double, 2> coords_multi_root() {
  return {{5.0, 0.0}};
}

template <>
std::array<double, 3> coords_multi_root() {
  return {{5.0, 0.0, 0.0}};
}

template <size_t Dim>
void test(const bool linear_expansion) {
  INFO("Map");
  CAPTURE(Dim);
  CAPTURE(linear_expansion);
  static constexpr size_t deriv_order = 2;

  const auto run_tests = [linear_expansion](
                             const std::array<DataVector, deriv_order + 1>&
                                 init_func_a,
                             const std::array<DataVector, deriv_order + 1>&
                                 init_func_b,
                             const double outer_boundary,
                             const std::array<double, Dim> point_xi,
                             const double final_time) {
    const double initial_time = -0.5;
    double t = -0.4;
    const double dt = 0.6;

    using Polynomial =
        domain::FunctionsOfTime::PiecewisePolynomial<deriv_order>;
    using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
    std::unordered_map<std::string, FoftPtr> f_of_t_list{};
    f_of_t_list["expansion_a"] =
        std::make_unique<Polynomial>(initial_time, init_func_a);
    f_of_t_list["expansion_b"] =
        std::make_unique<Polynomial>(initial_time, init_func_b);

    const FoftPtr& expansion_a_base = f_of_t_list.at("expansion_a");
    const FoftPtr& expansion_b_base = f_of_t_list.at("expansion_b");

    const CoordMapsTimeDependent::CubicScale<Dim> scale_map(
        outer_boundary, "expansion_a",
        linear_expansion ? "expansion_a"s : "expansion_b"s);

    // test serialized/deserialized map
    const auto scale_map_deserialized = serialize_and_deserialize(scale_map);

    while (t < final_time) {
      const double a = expansion_a_base->func_and_deriv(t)[0][0];
      const double b =
          linear_expansion ? a : expansion_b_base->func_and_deriv(t)[0][0];

      const double radius_squared = square(magnitude(point_xi));
      const std::array<double, Dim> mapped_point =
          point_xi * (a + (b - a) * radius_squared / square(outer_boundary));

      CHECK_ITERABLE_APPROX(scale_map(point_xi, t, f_of_t_list), mapped_point);
      CHECK_ITERABLE_APPROX(
          scale_map.inverse(mapped_point, t, f_of_t_list).get(), point_xi);
      CHECK_ITERABLE_APPROX(scale_map_deserialized(point_xi, t, f_of_t_list),
                            mapped_point);
      CHECK_ITERABLE_APPROX(
          scale_map_deserialized.inverse(mapped_point, t, f_of_t_list).get(),
          point_xi);

      if (not linear_expansion) {
        // Check that inverse map returns invalid for mapped point outside
        // the outer boundary and inside the inner boundary.
        for (size_t i = 0; i < Dim; ++i) {
          std::array<double, Dim> bad_mapped_point = make_array<Dim>(1.1);
          gsl::at(bad_mapped_point, i) *= outer_boundary;
          CHECK(not scale_map.inverse(bad_mapped_point, t, f_of_t_list));
        }
      }

      test_jacobian(scale_map, point_xi, t, f_of_t_list);
      test_inv_jacobian(scale_map, point_xi, t, f_of_t_list);
      test_frame_velocity(scale_map, point_xi, t, f_of_t_list);

      test_jacobian(scale_map_deserialized, point_xi, t, f_of_t_list);
      test_inv_jacobian(scale_map_deserialized, point_xi, t, f_of_t_list);
      test_frame_velocity(scale_map_deserialized, point_xi, t, f_of_t_list);

      // Check inequivalence operator
      CHECK_FALSE(scale_map != scale_map);
      CHECK_FALSE(scale_map_deserialized != scale_map_deserialized);

      // Check serialization
      CHECK(scale_map == scale_map_deserialized);
      CHECK_FALSE(scale_map != scale_map_deserialized);

      test_coordinate_map_argument_types(scale_map, point_xi, t, f_of_t_list);

      t += dt;
    }
  };

  std::array<DataVector, deriv_order + 1> init_func_a1{
      {{1.0}, {0.0007}, {-0.004}}};
  std::array<DataVector, deriv_order + 1> init_func_b1{
      {{1.0}, {-0.001}, {0.003}}};
  run_tests(init_func_a1, init_func_b1, 20.0, coords_single_root<Dim>(), 4.0);

  // test again with inputs that explicitly result in multiple roots
  std::array<DataVector, deriv_order + 1> init_func_a2{{{1.0}, {0.0}, {0.0}}};
  std::array<DataVector, deriv_order + 1> init_func_b2{{{0.99}, {0.0}, {0.0}}};
  run_tests(init_func_a2, init_func_b2, 6.0, coords_multi_root<Dim>(), 0.0);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.CoordMapsTimeDependent.CubicScale",
                  "[Domain][Unit]") {
  for (const auto linear_expansion : {false, true}) {
    test<1>(linear_expansion);
    test<2>(linear_expansion);
    test<3>(linear_expansion);
  }
  test_boundaries();
  CHECK(not CoordMapsTimeDependent::CubicScale<1>{}.is_identity());
}

// [[OutputRegex, The map is invertible only if 0 < expansion_b <
// expansion_a\*2/3]]
SPECTRE_TEST_CASE(
    "Unit.Domain.CoordMapsTimeDependent.CubicScale.NonInvertible1",
    "[Domain][Unit]") {
  ERROR_TEST();
  // the two expansion factors are chosen such that the map is non-invertible
  cubic_scale_non_invertible(0.96, 0.58, 20.0);
}

// [[OutputRegex, We require expansion_a > 0 for invertibility]]
SPECTRE_TEST_CASE(
    "Unit.Domain.CoordMapsTimeDependent.CubicScale.NonInvertible2",
    "[Domain][Unit]") {
  ERROR_TEST();
  // Make a<0
  cubic_scale_non_invertible(-0.0001, 1.0, 20.0);
}

// [[OutputRegex, The map is invertible only if 0 < expansion_b <
// expansion_a\*2/3]]
SPECTRE_TEST_CASE(
    "Unit.Domain.CoordMapsTimeDependent.CubicScale.NonInvertible3",
    "[Domain][Unit]") {
  ERROR_TEST();
  // Make b==0
  cubic_scale_non_invertible(0.96, 0.0, 20.0);
}

// [[OutputRegex, For invertability, we require outer_boundary to be positive]]
SPECTRE_TEST_CASE(
    "Unit.Domain.CoordMapsTimeDependent.CubicScale.NonInvertible4",
    "[Domain][Unit]") {
  ERROR_TEST();
  cubic_scale_non_invertible(0.96, 1.0, 0.0);
}
}  // namespace domain
