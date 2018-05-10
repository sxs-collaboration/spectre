// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <unordered_map>

#include "ControlSystem/FunctionOfTime.hpp"
#include "ControlSystem/PiecewisePolynomial.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/CubicScale.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

SPECTRE_TEST_CASE("Unit.Domain.CoordMapsTimeDependent.CubicScale",
                  "[Domain][Unit]") {
  static constexpr size_t deriv_order = 2;

  const auto run_tests = [](
      const std::array<DataVector, deriv_order + 1>& init_func_a,
      const std::array<DataVector, deriv_order + 1>& init_func_b,
      const double outer_b, const double x0, const double final_time) {
    double t = -0.5;
    const double dt = 0.6;

    FunctionsOfTime::PiecewisePolynomial<deriv_order> expansion_a(t,
                                                                  init_func_a);
    FunctionOfTime& expansion_a_base = expansion_a;

    FunctionsOfTime::PiecewisePolynomial<deriv_order> expansion_b(t,
                                                                  init_func_b);
    FunctionOfTime& expansions_b_base = expansion_b;

    const std::unordered_map<std::string, FunctionOfTime&> f_of_t_list = {
        {"expansion_a", expansion_a_base}, {"expansion_b", expansions_b_base}};

    const double outer_boundary = outer_b;
    const CoordMapsTimeDependent::CubicScale scale_map(outer_boundary);
    const std::array<double, 1> point_xi{{x0}};

    // test serialized/deserialized map
    const auto scale_map_deserialized = serialize_and_deserialize(scale_map);

    while (t < final_time) {
      const double a = expansion_a_base.func_and_deriv(t)[0][0];
      const double b = expansions_b_base.func_and_deriv(t)[0][0];

      const std::array<double, 1> mapped_point{
          {point_xi[0] * (a + (b - a) * square(point_xi[0] / outer_boundary))}};

      CHECK_ITERABLE_APPROX(scale_map(point_xi, t, f_of_t_list), mapped_point);
      CHECK_ITERABLE_APPROX(scale_map.inverse(mapped_point, t, f_of_t_list),
                            point_xi);
      CHECK_ITERABLE_APPROX(scale_map_deserialized(point_xi, t, f_of_t_list),
                            mapped_point);
      CHECK_ITERABLE_APPROX(
          scale_map_deserialized.inverse(mapped_point, t, f_of_t_list),
          point_xi);

      const double jacobian =
          a + 3.0 * (b - a) * square(point_xi[0] / outer_boundary);
      const double inv_jacobian =
          1.0 / (a + 3.0 * (b - a) * square(point_xi[0] / outer_boundary));

      CHECK(scale_map.jacobian(point_xi, t, f_of_t_list).get(0, 0) == jacobian);
      CHECK(scale_map.inv_jacobian(point_xi, t, f_of_t_list).get(0, 0) ==
            inv_jacobian);
      CHECK(
          scale_map_deserialized.jacobian(point_xi, t, f_of_t_list).get(0, 0) ==
          jacobian);
      CHECK(scale_map_deserialized.inv_jacobian(point_xi, t, f_of_t_list)
                .get(0, 0) == inv_jacobian);

      const double a_dot = expansion_a_base.func_and_deriv(t)[1][0];
      const double b_dot = expansions_b_base.func_and_deriv(t)[1][0];
      const std::array<double, 1> frame_vel =
          point_xi *
          (a_dot + (b_dot - a_dot) * square(point_xi[0] / outer_boundary));

      CHECK_ITERABLE_APPROX(scale_map.frame_velocity(point_xi, t, f_of_t_list),
                            frame_vel);
      CHECK_ITERABLE_APPROX(
          scale_map_deserialized.frame_velocity(point_xi, t, f_of_t_list),
          frame_vel);

      const double a_ddot = expansion_a_base.func_and_2_derivs(t)[2][0];
      const double b_ddot = expansions_b_base.func_and_2_derivs(t)[2][0];
      const double time_time =
          a_ddot * point_xi[0] +
          (b_ddot - a_ddot) * cube(point_xi[0]) / square(outer_boundary);
      const double time_space =
          a_dot + 3.0 * (b_dot - a_dot) * square(point_xi[0] / outer_boundary);
      const double space_space =
          6.0 * (b - a) * point_xi[0] / square(outer_boundary);

      CHECK(scale_map.hessian(point_xi, t, f_of_t_list).get(0, 0, 0) ==
            time_time);
      CHECK(scale_map.hessian(point_xi, t, f_of_t_list).get(0, 1, 0) ==
            time_space);
      CHECK(scale_map.hessian(point_xi, t, f_of_t_list).get(0, 0, 1) ==
            time_space);
      CHECK(scale_map.hessian(point_xi, t, f_of_t_list).get(0, 1, 1) ==
            space_space);

      CHECK(scale_map_deserialized.hessian(point_xi, t, f_of_t_list)
                .get(0, 0, 0) == time_time);
      CHECK(scale_map_deserialized.hessian(point_xi, t, f_of_t_list)
                .get(0, 1, 0) == time_space);
      CHECK(scale_map_deserialized.hessian(point_xi, t, f_of_t_list)
                .get(0, 0, 1) == time_space);
      CHECK(scale_map_deserialized.hessian(point_xi, t, f_of_t_list)
                .get(0, 1, 1) == space_space);

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
  run_tests(init_func_a1, init_func_b1, 20.0, 19.2, 4.0);
  // test again with inputs that explicitly result in multiple roots
  std::array<DataVector, deriv_order + 1> init_func_a2{{{1.0}, {0.0}, {0.0}}};
  std::array<DataVector, deriv_order + 1> init_func_b2{{{0.99}, {0.0}, {0.0}}};
  run_tests(init_func_a2, init_func_b2, 6.0, 5.0, 0.0);
}

SPECTRE_TEST_CASE(
    "Unit.Domain.CoordMapsTimeDependent.CubicScale.TestBoundaries",
    "[Domain][Unit]") {
  static constexpr size_t deriv_order = 2;

  const auto run_tests = [](const double x0) {
    double t = -0.5;
    const double dt = 0.6;
    const double final_time = 4.0;

    const std::array<DataVector, deriv_order + 1> init_func_a{
        {{1.0}, {0.0}, {0.0}}};
    const std::array<DataVector, deriv_order + 1> init_func_b{
        {{0.99}, {0.0}, {0.0}}};

    FunctionsOfTime::PiecewisePolynomial<deriv_order> expansion_a(t,
                                                                  init_func_a);
    FunctionOfTime& expansion_a_base = expansion_a;

    FunctionsOfTime::PiecewisePolynomial<deriv_order> expansion_b(t,
                                                                  init_func_b);
    FunctionOfTime& expansions_b_base = expansion_b;

    const std::unordered_map<std::string, FunctionOfTime&> f_of_t_list = {
        {"expansion_a", expansion_a_base}, {"expansion_b", expansions_b_base}};

    const double outer_boundary = 20.0;
    const CoordMapsTimeDependent::CubicScale scale_map(outer_boundary);
    const std::array<double, 1> point_xi{{x0}};

    while (t < final_time) {
      const double a = expansion_a_base.func_and_deriv(t)[0][0];
      const double b = expansions_b_base.func_and_deriv(t)[0][0];

      const std::array<double, 1> mapped_point{
          {point_xi[0] * (a + (b - a) * square(point_xi[0] / outer_boundary))}};

      CHECK_ITERABLE_APPROX(scale_map(point_xi, t, f_of_t_list), mapped_point);
      CHECK_ITERABLE_APPROX(scale_map.inverse(mapped_point, t, f_of_t_list),
                            point_xi);
      t += dt;
    }
  };

  // test inverse at inner and outer boundary values
  run_tests(0.0);
  run_tests(20.0);
}

// [[OutputRegex, The map is only invertible if expansion_b < expansion_a\*2/3]]
SPECTRE_TEST_CASE(
    "Unit.Domain.CoordMapsTimeDependent.CubicScale.NonInvertible1",
    "[Domain][Unit]") {
  ERROR_TEST();
  // the two expansion factors are chosen such that the map is non-invertible
  double t = 4.0;
  constexpr size_t deriv_order = 2;

  const std::array<DataVector, deriv_order + 1> init_func_a{
      {{0.96}, {0.0}, {0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> expansion_a(t, init_func_a);
  FunctionOfTime& expansion_a_base = expansion_a;

  const std::array<DataVector, deriv_order + 1> init_func_b{
      {{0.58}, {0.0}, {0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> expansion_b(t, init_func_b);
  FunctionOfTime& expansions_b_base = expansion_b;

  const std::unordered_map<std::string, FunctionOfTime&> f_of_t_list = {
      {"expansion_a", expansion_a_base}, {"expansion_b", expansions_b_base}};

  const double outer_boundary = 20.0;
  const CoordMapsTimeDependent::CubicScale scale_map(outer_boundary);
  const std::array<double, 1> point_xi{{19.2}};

  const std::array<double, 1> mapped_point{
      {point_xi[0] *
       (expansion_a_base.func(t)[0][0] +
        (expansions_b_base.func(t)[0][0] - expansion_a_base.func(t)[0][0]) *
            square(point_xi[0] / outer_boundary))}};

  // this call should fail since this specific choice of expansion factors makes
  // the map non-invertible
  scale_map.inverse(mapped_point, t, f_of_t_list);
}

// [[OutputRegex, We require expansion_a > 0 for invertibility]]
SPECTRE_TEST_CASE(
    "Unit.Domain.CoordMapsTimeDependent.CubicScale.NonInvertible2",
    "[Domain][Unit]") {
  ERROR_TEST();
  double t = 4.0;
  constexpr size_t deriv_order = 2;

  const std::array<DataVector, deriv_order + 1> init_func_a{
      {{-0.0001}, {0.0}, {0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> expansion_a(t, init_func_a);
  FunctionOfTime& expansion_a_base = expansion_a;

  const std::array<DataVector, deriv_order + 1> init_func_b{
      {{1.0}, {0.0}, {0.0}}};
  FunctionsOfTime::PiecewisePolynomial<deriv_order> expansion_b(t, init_func_b);
  FunctionOfTime& expansions_b_base = expansion_b;

  const std::unordered_map<std::string, FunctionOfTime&> f_of_t_list = {
      {"expansion_a", expansion_a_base}, {"expansion_b", expansions_b_base}};

  const double outer_boundary = 20.0;
  const CoordMapsTimeDependent::CubicScale scale_map(outer_boundary);
  const std::array<double, 1> point_xi{{19.2}};

  const std::array<double, 1> mapped_point{
      {point_xi[0] *
       (expansion_a_base.func(t)[0][0] +
        (expansions_b_base.func(t)[0][0] - expansion_a_base.func(t)[0][0]) *
            square(point_xi[0] / outer_boundary))}};

  // fails due to a_of_t <= 0.0
  scale_map.inverse(mapped_point, t, f_of_t_list);
}
