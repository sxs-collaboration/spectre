// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "tests/Unit/PointwiseFunctions/GeneralRelativity/TestHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {

struct KerrSchild {
  using type = gr::Solutions::KerrSchild;
  static constexpr OptionString help{"A Kerr-Schild solution"};
};

template <typename DataType>
tnsr::I<DataType, 3, Frame::Inertial> spatial_coords(
    const DataType& used_for_size) noexcept {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(used_for_size,
                                                                  0.0);
  get<0>(x) = 1.32;
  get<1>(x) = 0.82;
  get<2>(x) = 1.24;
  return x;
}

template <typename DataType>
void test_schwarzschild(const DataType& used_for_size) noexcept {
  // Schwarzschild solution is:
  // H        = M/r
  // l_mu     = (1,x/r,y/r,z/r)
  // lapse    = (1+2M/r)^{-1/2}
  // d_lapse  = (1+2M/r)^{-3/2}(Mx^i/r^3)
  // shift^i  = (2Mx^i/r^2) / lapse^2
  // g_{ij}   = delta_{ij} + 2 M x_i x_j/r^3
  // d_i H    = -Mx_i/r^3
  // d_i l_j  = delta_{ij}/r - x^i x^j/r^3
  // d_k g_ij = -6M x_i x_j x_k/r^5 + 2 M x_i delta_{kj}/r^3
  //                                + 2 M x_j delta_{ki}/r^3

  // Parameters for KerrSchild solution
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const auto x = spatial_coords(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  gr::Solutions::KerrSchild solution(mass, spin, center);

  const auto vars =
      solution.variables(x, t, gr::Solutions::KerrSchild::tags<DataType>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& d_lapse =
      get<gr::Solutions::KerrSchild::DerivLapse<DataType>>(vars);
  const auto& shift = get<gr::Tags::Shift<3, Frame::Inertial, DataType>>(vars);
  const auto& d_shift =
      get<gr::Solutions::KerrSchild::DerivShift<DataType>>(vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataType>>>(vars);
  const auto& g =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>(vars);
  const auto& dt_g =
      get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataType>>>(
          vars);
  const auto& d_g =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataType>>(vars);

  // Check those quantities that should be zero.
  const auto zero = make_with_value<DataType>(x, 0.);
  CHECK(dt_lapse.get() == zero);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(dt_shift.get(i) == zero);
    for (size_t j = 0; j < 3; ++j) {
      CHECK(dt_g.get(i, j) == zero);
    }
  }

  const DataType r = get(magnitude(x));
  const DataType one_over_r_squared = 1.0 / square(r);
  const DataType one_over_r_cubed = 1.0 / cube(r);
  const DataType one_over_r_fifth = one_over_r_squared * one_over_r_cubed;
  auto expected_lapse = make_with_value<Scalar<DataType>>(x, 0.0);
  get(expected_lapse) = 1.0 / sqrt(1.0 + 2.0 * mass / r);
  CHECK_ITERABLE_APPROX(lapse, expected_lapse);

  auto expected_d_lapse =
      make_with_value<tnsr::i<DataType, 3, Frame::Inertial>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_d_lapse.get(i) =
        mass * x.get(i) * one_over_r_cubed * cube(get(lapse));
  }
  CHECK_ITERABLE_APPROX(d_lapse, expected_d_lapse);

  auto expected_shift =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_shift.get(i) =
        2.0 * mass * x.get(i) * one_over_r_squared * square(get(lapse));
  }
  CHECK_ITERABLE_APPROX(shift, expected_shift);

  auto expected_d_shift =
      make_with_value<tnsr::iJ<DataType, 3, Frame::Inertial>>(x, 0.0);
  for (size_t j = 0; j < 3; ++j) {
    expected_d_shift.get(j, j) =
        2.0 * mass * one_over_r_squared * square(get(lapse));
    for (size_t i = 0; i < 3; ++i) {
      expected_d_shift.get(j, i) -=
          4.0 * mass * x.get(j) * x.get(i) * square(one_over_r_squared) *
          square(get(lapse)) * (1 - mass / r * square(get(lapse)));
    }
  }
  CHECK_ITERABLE_APPROX(d_shift, expected_d_shift);

  auto expected_g =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      expected_g.get(i, j) =
          2.0 * mass * x.get(i) * x.get(j) * one_over_r_cubed;
    }
    expected_g.get(i, i) += 1.0;
  }
  CHECK_ITERABLE_APPROX(g, expected_g);

  auto expected_d_g =
      make_with_value<tnsr::ijj<DataType, 3, Frame::Inertial>>(x, 0.0);
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {  // Symmetry
        expected_d_g.get(k, i, j) =
            -6.0 * mass * x.get(i) * x.get(j) * x.get(k) * one_over_r_fifth;
        if (k == j) {
          expected_d_g.get(k, i, j) += 2.0 * mass * x.get(i) * one_over_r_cubed;
        }
        if (k == i) {
          expected_d_g.get(k, i, j) += 2.0 * mass * x.get(j) * one_over_r_cubed;
        }
      }
    }
  }
  CHECK_ITERABLE_APPROX(d_g, expected_d_g);
}

void test_einstein_solution() noexcept {
  // Parameters
  //   ...for KerrSchild solution
  const double mass = 1.7;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
  //   ...for grid
  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};

  gr::Solutions::KerrSchild solution(mass, spin, center);
  verify_time_independent_einstein_solution(
      solution, grid_size, lower_bound, upper_bound,
      std::numeric_limits<double>::epsilon() * 1.e5);
}

void test_double_vs_datavector() noexcept {
  // Parameters for KerrSchild solution
  const double mass = 1.7;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
  gr::Solutions::KerrSchild solution(mass, spin, center);

  const double t = 1.3;
  const auto x1 = spatial_coords(0.0);
  const auto x2 = spatial_coords(DataVector{0.0, 0.0, 0.0});

  const auto vars1 =
      solution.variables(x1, t, gr::Solutions::KerrSchild::tags<double>{});
  const auto& lapse1 = get<gr::Tags::Lapse<double>>(vars1);
  const auto& dt_lapse1 = get<Tags::dt<gr::Tags::Lapse<double>>>(vars1);
  const auto& d_lapse1 =
      get<gr::Solutions::KerrSchild::DerivLapse<double>>(vars1);
  const auto& shift1 = get<gr::Tags::Shift<3, Frame::Inertial, double>>(vars1);
  const auto& d_shift1 =
      get<gr::Solutions::KerrSchild::DerivShift<double>>(vars1);
  const auto& dt_shift1 =
      get<Tags::dt<gr::Tags::Shift<3, Frame::Inertial, double>>>(vars1);
  const auto& g1 =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, double>>(vars1);
  const auto& dt_g1 =
      get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, double>>>(vars1);
  const auto& d_g1 =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<double>>(vars1);

  const auto vars2 =
      solution.variables(x2, t, gr::Solutions::KerrSchild::tags<DataVector>{});
  const auto& lapse2 = get<gr::Tags::Lapse<DataVector>>(vars2);
  const auto& dt_lapse2 = get<Tags::dt<gr::Tags::Lapse<DataVector>>>(vars2);
  const auto& d_lapse2 =
      get<gr::Solutions::KerrSchild::DerivLapse<DataVector>>(vars2);
  const auto& shift2 =
      get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(vars2);
  const auto& d_shift2 =
      get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(vars2);
  const auto& dt_shift2 =
      get<Tags::dt<gr::Tags::Shift<3, Frame::Inertial, DataVector>>>(vars2);
  const auto& g2 =
      get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars2);
  const auto& dt_g2 =
      get<Tags::dt<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(
          vars2);
  const auto& d_g2 =
      get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(vars2);

  check_tensor_doubles_approx_equals_tensor_datavectors(lapse2, lapse1);
  check_tensor_doubles_approx_equals_tensor_datavectors(shift2, shift1);
  check_tensor_doubles_approx_equals_tensor_datavectors(g2, g1);
  check_tensor_doubles_approx_equals_tensor_datavectors(dt_lapse2, dt_lapse1);
  check_tensor_doubles_approx_equals_tensor_datavectors(dt_shift2, dt_shift1);
  check_tensor_doubles_approx_equals_tensor_datavectors(dt_g2, dt_g1);
  check_tensor_doubles_approx_equals_tensor_datavectors(d_lapse2, d_lapse1);
  check_tensor_doubles_approx_equals_tensor_datavectors(d_shift2, d_shift1);
  check_tensor_doubles_approx_equals_tensor_datavectors(d_g2, d_g1);
}

void test_serialize() noexcept {
  gr::Solutions::KerrSchild solution(3.0, {{0.2, 0.3, 0.2}}, {{0.0, 3.0, 4.0}});
  test_serialization(solution);
}

void test_copy_and_move() noexcept {
  gr::Solutions::KerrSchild solution(3.0, {{0.2, 0.3, 0.2}}, {{0.0, 3.0, 4.0}});
  test_copy_semantics(solution);
  auto solution_copy = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  Options<tmpl::list<KerrSchild>> opts("");
  opts.parse(
      "KerrSchild:\n"
      "  Mass: 0.5\n"
      "  Spin: [0.1,0.2,0.3]\n"
      "  Center: [1.0,3.0,2.0]");
  CHECK(opts.get<KerrSchild>() ==
        gr::Solutions::KerrSchild(0.5, {{0.1, 0.2, 0.3}}, {{1.0, 3.0, 2.0}}));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.KerrSchild",
                  "[PointwiseFunctions][Unit]") {
  test_schwarzschild<DataVector>(DataVector{0.0, 0.0, 0.0});
  test_schwarzschild<double>(0.0);
  test_einstein_solution();
  test_double_vs_datavector();
  test_copy_and_move();
  test_serialize();
  test_construct_from_options();
}

// [[OutputRegex, Spin magnitude must be < 1]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.KerrSchildSpin",
                  "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  gr::Solutions::KerrSchild solution(1.0, {{1.0, 1.0, 1.0}}, {{0.0, 0.0, 0.0}});
}

// [[OutputRegex, Mass must be non-negative]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.KerrSchildMass",
                  "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  gr::Solutions::KerrSchild solution(-1.0, {{0.0, 0.0, 0.0}},
                                     {{0.0, 0.0, 0.0}});
}

// [[OutputRegex, In string:.*At line 2 column 9:.Value -0.5 is below the lower
// bound of 0]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.KerrSchildOptM",
                  "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<KerrSchild>> opts("");
  opts.parse(
      "KerrSchild:\n"
      "  Mass: -0.5\n"
      "  Spin: [0.1,0.2,0.3]\n"
      "  Center: [1.0,3.0,2.0]");
  opts.get<KerrSchild>();
}

// [[OutputRegex, In string:.*At line 2 column 3:.Spin magnitude must be < 1]]
SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.KerrSchildOptS",
                  "[PointwiseFunctions][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<KerrSchild>> opts("");
  opts.parse(
      "KerrSchild:\n"
      "  Mass: 0.5\n"
      "  Spin: [1.1,0.9,0.3]\n"
      "  Center: [1.0,3.0,2.0]");
  opts.get<KerrSchild>();
}
