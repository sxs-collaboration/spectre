// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
using Affine = domain::CoordinateMaps::Affine;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.32;
  get<1>(x) = 0.82;
  get<2>(x) = 1.24;
  return x;
}

template <typename Frame, typename DataType>
void test_schwarzschild(const DataType& used_for_size) {
  // Schwarzschild solution is (with x_i = delta_{ij} x^i):
  // H                    = M/r
  // l_mu                 = (1,x/r,y/r,z/r)
  // lapse                = (1+2M/r)^{-1/2}
  // d_i lapse            = (1+2M/r)^{-3/2}(Mx^i/r^3)
  // shift^i              = (2Mx^i/r^2) * lapse^2
  // g_{ij}               = delta_{ij} + 2 M x_i x_j/r^3
  // d_i H                = -Mx_i/r^3
  // d_i l_j              = delta_{ij}/r - x^i x^j/r^3
  // d_k g_ij             = -6M x_i x_j x_k/r^5 + 2 M x_i delta_{kj}/r^3
  //                                + 2 M x_j delta_{ki}/r^3
  // Gamma_{ijk}          = M x_i(2r^2 delta_{jk} - 3 x_j x_k) / r^5
  // Gamma^i_{jk}         = M x^i(2r^2 delta_{jk} - 3 x_j x_k) / (2Mr^4 + r^5)
  // g^{ij} Gamma^k_{ij}  = M(8M + 3r) x^k / (r(2M+3r))^2
  // g^{ij} K_{ij}        = 2M(3M + r) * ((2M+r)r)^(-3/2)

  // Parameters for KerrSchild solution
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  gr::Solutions::KerrSchild solution(mass, spin, center);

  const auto vars = solution.variables(
      x, t, typename gr::Solutions::KerrSchild::tags<DataType, Frame>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& d_lapse =
      get<typename gr::Solutions::KerrSchild::DerivLapse<DataType, Frame>>(
          vars);
  const auto& shift = get<gr::Tags::Shift<DataType, 3, Frame>>(vars);
  const auto& d_shift =
      get<typename gr::Solutions::KerrSchild::DerivShift<DataType, Frame>>(
          vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<DataType, 3, Frame>>>(vars);
  const auto& g = get<gr::Tags::SpatialMetric<DataType, 3, Frame>>(vars);
  const auto& ig =
      get<gr::Tags::InverseSpatialMetric<DataType, 3, Frame>>(vars);

  const auto& dt_g =
      get<Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>>(vars);
  const auto& d_g = get<
      typename gr::Solutions::KerrSchild::DerivSpatialMetric<DataType, Frame>>(
      vars);
  const auto& christoffel_first_kind =
      get<typename gr::Tags::SpatialChristoffelFirstKind<DataType, 3, Frame>>(
          vars);
  const auto& christoffel_second_kind =
      get<typename gr::Tags::SpatialChristoffelSecondKind<DataType, 3, Frame>>(
          vars);
  const auto& trace_christoffel = get<
      typename gr::Tags::TraceSpatialChristoffelSecondKind<DataType, 3, Frame>>(
      vars);
  const auto& trace_extrinsic_curvature =
      get<typename gr::Tags::TraceExtrinsicCurvature<DataType>>(vars);

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

  auto expected_d_lapse = make_with_value<tnsr::i<DataType, 3, Frame>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_d_lapse.get(i) =
        mass * x.get(i) * one_over_r_cubed * cube(get(lapse));
  }
  CHECK_ITERABLE_APPROX(d_lapse, expected_d_lapse);

  auto expected_shift = make_with_value<tnsr::I<DataType, 3, Frame>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    expected_shift.get(i) =
        2.0 * mass * x.get(i) * one_over_r_squared * square(get(lapse));
  }
  CHECK_ITERABLE_APPROX(shift, expected_shift);

  auto expected_d_shift = make_with_value<tnsr::iJ<DataType, 3, Frame>>(x, 0.0);
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

  auto expected_g = make_with_value<tnsr::ii<DataType, 3, Frame>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      expected_g.get(i, j) =
          2.0 * mass * x.get(i) * x.get(j) * one_over_r_cubed;
    }
    expected_g.get(i, i) += 1.0;
  }
  CHECK_ITERABLE_APPROX(g, expected_g);
  CHECK_ITERABLE_APPROX(ig, determinant_and_inverse(expected_g).second);

  auto expected_d_g = make_with_value<tnsr::ijj<DataType, 3, Frame>>(x, 0.0);
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

  auto expected_christoffel_first_kind =
      make_with_value<tnsr::ijj<DataType, 3, Frame>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = j; k < 3; ++k) {
        const double delta_jk = j == k ? 1. : 0.;
        expected_christoffel_first_kind.get(i, j, k) =
            mass * x.get(i) *
            (-3. * x.get(j) * x.get(k) + 2. * square(r) * delta_jk) / pow(r, 5);
      }
    }
  }
  CHECK_ITERABLE_APPROX(christoffel_first_kind,
                        expected_christoffel_first_kind);

  auto expected_christoffel_second_kind =
      make_with_value<tnsr::Ijj<DataType, 3, Frame>>(x, 0.0);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      for (size_t k = j; k < 3; ++k) {
        const double delta_jk = j == k ? 1. : 0.;
        expected_christoffel_second_kind.get(i, j, k) =
            x.get(i) * mass *
            (-3. * x.get(j) * x.get(k) + 2. * square(r) * delta_jk) /
            (2. * mass * pow(r, 4) + pow(r, 5));
      }
    }
  }
  CHECK_ITERABLE_APPROX(christoffel_second_kind,
                        expected_christoffel_second_kind);

  auto expected_trace_christoffel =
      make_with_value<tnsr::I<DataType, 3, Frame>>(x, 0.0);
  DataType factor =
      mass * (8. * mass + 3. * r) / (2. * mass + r) / (2. * mass + r) / r / r;
  for (size_t i = 0; i < 3; ++i) {
    expected_trace_christoffel.get(i) = factor * x.get(i);
  }
  CHECK_ITERABLE_APPROX(trace_christoffel, expected_trace_christoffel);

  auto expected_trace_extrinsic_curvature =
      make_with_value<Scalar<DataType>>(x, 0.0);
  expected_trace_extrinsic_curvature.get() =
      2. * mass * (3. * mass + r) * pow((2. * mass + r) * r, -1.5);
  CHECK_ITERABLE_APPROX(trace_extrinsic_curvature,
                        expected_trace_extrinsic_curvature);
}

template <typename FrameType>
void test_numerical_deriv_det_spatial_metric(const DataVector& used_for_size) {
  // Parameters for KerrSchild solution
  const double mass = 1.01;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  gr::Solutions::KerrSchild solution(mass, spin, center);

  // Setup grid
  const size_t num_points_1d = 8;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  const size_t SpatialDim = 3;
  Mesh<SpatialDim> mesh{num_points_1d, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto};
  const auto coord_map =
      domain::make_coordinate_map<Frame::ElementLogical, FrameType>(Affine3D{
          Affine{-1., 1., lower_bound[0], upper_bound[0]},
          Affine{-1., 1., lower_bound[1], upper_bound[1]},
          Affine{-1., 1., lower_bound[2], upper_bound[2]},
      });
  const size_t num_points_3d = num_points_1d * num_points_1d * num_points_1d;
  // Setup coordinates
  const auto x_logical = logical_coordinates(mesh);
  const auto x = coord_map(x_logical);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Compute actual analytical derivative of the determinant
  const auto deriv_det_spatial_metric =
      get<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim, FrameType>>(
          solution.variables(
              x, t,
              tmpl::list<gr::Tags::DerivDetSpatialMetric<DataVector, SpatialDim,
                                                         FrameType>>{}));

  // Compute expected numerical derivative of the determinant
  const double null_vector_0 = -1.0;
  gr::Solutions::KerrSchild::IntermediateComputer<DataVector, FrameType>
      ks_computer(solution, x);
  gr::Solutions::KerrSchild::IntermediateVars<DataVector, FrameType> ks_cache(
      num_points_3d);

  auto H = make_with_value<Scalar<DataVector>>(
      used_for_size, std::numeric_limits<double>::signaling_NaN());
  using H_tag = gr::Solutions::KerrSchild::internal_tags::H<DataVector>;
  ks_computer(make_not_null(&H), make_not_null(&ks_cache), H_tag{});

  Variables<tmpl::list<H_tag>> H_var(num_points_3d);
  get<H_tag>(H_var) = H;
  const auto expected_deriv_H_var = partial_derivatives<tmpl::list<H_tag>>(
      H_var, mesh, coord_map.inv_jacobian(x_logical));
  const auto& expected_deriv_H =
      get<Tags::deriv<H_tag, tmpl::size_t<SpatialDim>, FrameType>>(
          expected_deriv_H_var);

  tnsr::i<DataVector, SpatialDim, FrameType>
      expected_deriv_det_spatial_metric{};
  for (size_t i = 0; i < SpatialDim; i++) {
    expected_deriv_det_spatial_metric.get(i) =
        2.0 * square(null_vector_0) * expected_deriv_H.get(i);
  }

  // A custom epsilon is used here because the Legendre polynomials don't fit
  // the derivative of 1 / r well. This was looked at for various box sizes and
  // number of 1D grid points.
  Approx approx = Approx::custom().epsilon(1e-11).scale(1.0);
  CHECK_ITERABLE_CUSTOM_APPROX(deriv_det_spatial_metric,
                               expected_deriv_det_spatial_metric, approx);
}

template <typename Frame, typename DataType>
void test_tag_retrieval(const DataType& used_for_size) {
  // Parameters for KerrSchild solution
  const double mass = 1.234;
  const std::array<double, 3> spin{{0.1, -0.2, 0.3}};
  const std::array<double, 3> center{{1.0, 2.0, 3.0}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  const gr::Solutions::KerrSchild solution(mass, spin, center);
  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      solution, x, t,
      typename gr::Solutions::KerrSchild::template tags<DataType, Frame>{});
}

template <typename Frame>
void test_einstein_solution() {
  // Parameters
  //   ...for KerrSchild solution
  const double mass = 1.7;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
  //   ...for grid
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const double time = -2.8;

  gr::Solutions::KerrSchild solution(mass, spin, center);
  TestHelpers::VerifyGrSolution::verify_consistency(
      solution, time, tnsr::I<double, 3, Frame>{lower_bound}, 0.01, 1.0e-10);
  if constexpr (std::is_same_v<Frame, ::Frame::Inertial>) {
    // Don't look at time-independent solution in other than the inertial
    // frame.
    const size_t grid_size = 8;
    const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
    TestHelpers::VerifyGrSolution::verify_time_independent_einstein_solution(
        solution, grid_size, lower_bound, upper_bound,
        std::numeric_limits<double>::epsilon() * 1.e5);
  }
}

template <typename Frame, typename DataType>
void test_zero_spin_optimization(const DataType& used_for_size) {
  // the optimizations are only taken when the spin is exactly zero, so we can
  // test it against a numerical epsilon that should not affect results
  gr::Solutions::KerrSchild solution_zero_spin(3.0, {{0., 0., 0.}},
                                               {{0.2, 0.3, 0.2}});
  gr::Solutions::KerrSchild solution_tiny_spin(3.0, {{0., 0., 1e-50}},
                                               {{0.2, 0.3, 0.2}});
  CHECK(solution_zero_spin.zero_spin());
  CHECK(not solution_tiny_spin.zero_spin());
  const auto x = spatial_coords<Frame>(used_for_size);
  using all_tags = typename gr::Solutions::KerrSchild::tags<DataType, Frame>;
  const auto all_tags_zero_spin =
      solution_zero_spin.variables(x, 0., all_tags{});
  const auto all_tags_tiny_spin =
      solution_tiny_spin.variables(x, 0., all_tags{});
  tmpl::for_each<all_tags>(
      [&all_tags_zero_spin, &all_tags_tiny_spin](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(all_tags_zero_spin),
                              get<tag>(all_tags_tiny_spin));
      });
}

void test_serialize() {
  gr::Solutions::KerrSchild solution(3.0, {{0.2, 0.3, 0.2}}, {{0.0, 3.0, 4.0}});
  test_serialization(solution);
}

void test_copy_and_move() {
  gr::Solutions::KerrSchild solution(3.0, {{0.2, 0.3, 0.2}}, {{0.0, 3.0, 4.0}});
  test_copy_semantics(solution);
  auto solution_copy = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  const auto created = TestHelpers::test_creation<gr::Solutions::KerrSchild>(
      "Mass: 0.5\n"
      "Spin: [0.1,0.2,0.3]\n"
      "Center: [1.0,3.0,2.0]");
  CHECK(created ==
        gr::Solutions::KerrSchild(0.5, {{0.1, 0.2, 0.3}}, {{1.0, 3.0, 2.0}}));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.KerrSchild",
                  "[PointwiseFunctions][Unit]") {
  test_copy_and_move();
  test_serialize();
  test_construct_from_options();

  test_schwarzschild<Frame::Inertial>(DataVector(5));
  test_schwarzschild<Frame::Inertial>(0.0);
  test_numerical_deriv_det_spatial_metric<Frame::Inertial>(DataVector(5));
  test_tag_retrieval<Frame::Inertial>(DataVector(5));
  test_tag_retrieval<Frame::Inertial>(0.0);
  test_einstein_solution<Frame::Inertial>();
  test_zero_spin_optimization<Frame::Inertial>(DataVector(5));
  test_zero_spin_optimization<Frame::Inertial>(0.0);

  test_schwarzschild<Frame::Grid>(DataVector(5));
  test_schwarzschild<Frame::Grid>(0.0);
  test_numerical_deriv_det_spatial_metric<Frame::Grid>(DataVector(5));
  test_tag_retrieval<Frame::Grid>(DataVector(5));
  test_tag_retrieval<Frame::Grid>(0.0);
  test_einstein_solution<Frame::Grid>();
  test_zero_spin_optimization<Frame::Grid>(DataVector(5));
  test_zero_spin_optimization<Frame::Grid>(0.0);

  CHECK_THROWS_WITH(
      []() {
        gr::Solutions::KerrSchild solution(1.0, {{1.0, 1.0, 1.0}},
                                           {{0.0, 0.0, 0.0}});
      }(),
      Catch::Contains("Spin magnitude must be < 1"));
  CHECK_THROWS_WITH(
      []() {
        gr::Solutions::KerrSchild solution(-1.0, {{0.0, 0.0, 0.0}},
                                           {{0.0, 0.0, 0.0}});
      }(),
      Catch::Contains("Mass must be non-negative"));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<gr::Solutions::KerrSchild>(
          "Mass: -0.5\n"
          "Spin: [0.1,0.2,0.3]\n"
          "Center: [1.0,3.0,2.0]"),
      Catch::Contains("Value -0.5 is below the lower bound of 0"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<gr::Solutions::KerrSchild>(
                        "Mass: 0.5\n"
                        "Spin: [1.1,0.9,0.3]\n"
                        "Center: [1.0,3.0,2.0]"),
                    Catch::Contains("Spin magnitude must be < 1"));
}
