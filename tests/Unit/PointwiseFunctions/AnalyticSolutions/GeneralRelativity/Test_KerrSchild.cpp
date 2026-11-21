// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <string>
#include <type_traits>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
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
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/SpecialRelativity/LorentzBoostMatrix.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

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

  const auto expected_deriv_H =
      partial_derivative(H, mesh, coord_map.inv_jacobian(x_logical));

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
  const std::array<double, 3> boost_velocity{{0.2, -0.3, 0.1}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  using all_tags = typename gr::Solutions::KerrSchild::tags<DataType, Frame>;

  // We test gr::Tags::SpacetimeChristoffelSecondKind<DataType, 3, Frame>
  // separately as it is not supported for the boosted solution.
  using tags_for_zero_boost =
      tmpl::list<gr::Tags::SpacetimeChristoffelSecondKind<DataType, 3, Frame>>;
  using regular_tags = tmpl::remove<
      all_tags, gr::Tags::SpacetimeChristoffelSecondKind<DataType, 3, Frame>>;

  // Evaluate solution
  const gr::Solutions::KerrSchild solution(mass, spin, center, boost_velocity);
  TestHelpers::AnalyticSolutions::test_tag_retrieval(solution, x, t,
                                                     regular_tags{});

  const gr::Solutions::KerrSchild solution_zero_boost(mass, spin, center);
  TestHelpers::AnalyticSolutions::test_tag_retrieval(solution_zero_boost, x, t,
                                                     tags_for_zero_boost{});
}

template <typename Frame>
void test_einstein_solution(const bool use_non_zero_velocity) {
  INFO("Verify KerrSchild solution satisfies Einstein equations");
  // Parameters
  //   ...for KerrSchild solution
  const double mass = 1.7;
  const std::array<double, 3> spin{{0.1, 0.2, 0.3}};
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
   const std::array<double, 3> boost_velocity{{0.4, -0.51, -0.5}};
  //   ...for grid
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const double time = -2.8;

  if (use_non_zero_velocity) {
    gr::Solutions::KerrSchild solution(mass, spin, center, boost_velocity);
    // Time derivatives are non zero. However we do not code up the time
    // dependence of the boosted solution for simplicity. Hence we only check
    // consistency in the spatial derivatives.
    TestHelpers::VerifyGrSolution::verify_spatial_consistency(
        solution, time, tnsr::I<double, 3, Frame>{lower_bound}, 0.01, 1.0e-10);
  } else {
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

template <typename Frame, typename DataType>
void test_boosted_for_zero_velocity(const DataType& used_for_size) {
  const double epsilon = std::numeric_limits<double>::epsilon();
  const std::array<double, 3> boost_velocity{{0.0, 0.0, epsilon}};

  gr::Solutions::KerrSchild solution_zero_velocity(
      3.0, {{0., 0., 0.}}, {{0.2, 0.3, 0.2}});
  gr::Solutions::KerrSchild solution_tiny_velocity(
      3.0, {{0., 0., 1e-50}}, {{0.2, 0.3, 0.2}}, boost_velocity);
  CHECK(solution_zero_velocity.zero_velocity());
  CHECK(not solution_tiny_velocity.zero_velocity());
  const auto x = spatial_coords<Frame>(used_for_size);

  // We do not test gr::Tags::SpacetimeChristoffelSecondKind<DataType, 3, Frame>
  // as it is not supported for the boosted solution.
  using all_tags = typename gr::Solutions::KerrSchild::tags<DataType, Frame>;
  using tags_to_test = tmpl::remove<
      all_tags, gr::Tags::SpacetimeChristoffelSecondKind<DataType, 3, Frame>>;

  const auto all_tags_zero_velocity =
      solution_zero_velocity.variables(x, 0., tags_to_test{});
  const auto all_tags_tiny_velocity =
      solution_tiny_velocity.variables(x, 0., tags_to_test{});
  tmpl::for_each<tags_to_test>(
      [&all_tags_zero_velocity, &all_tags_tiny_velocity](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(all_tags_zero_velocity),
                              get<tag>(all_tags_tiny_velocity));
      });
}

template <typename Frame, typename DataType>
void test_boosted_spacetime_metric(const DataType& used_for_size) {
  // Test that the spacetime metric of the boosted solution indeed
  // corresponds to the boosted spacetime metric tensor

  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 0.0;

  // KerrSchild solution
  const double mass = 1.0;
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  gr::Solutions::KerrSchild solution(mass, spin, center);

  // Boosted KerrSchild solution
  const std::array<double, 3> velocity{{0.5, 0.0, 0.0}};
  gr::Solutions::KerrSchild boosted_solution(mass, spin, center, velocity);

  // Need to evaluate at the boosted coordinates
  // Boost coordinates
  tnsr::I<DataType, 3, Frame> x_boosted;
  sr::lorentz_boost(make_not_null(&x_boosted), x, 0., -velocity);

  // We do not test gr::Tags::SpacetimeChristoffelSecondKind<DataType, 3, Frame>
  // as it is not supported for the boosted solution.
  using all_tags = typename gr::Solutions::KerrSchild::tags<DataType, Frame>;
  using tags_to_test = tmpl::remove<
      all_tags, gr::Tags::SpacetimeChristoffelSecondKind<DataType, 3, Frame>>;

  const auto vars = solution.variables(x_boosted, t, tags_to_test{});
  const auto& lapse = get<typename gr::Tags::Lapse<DataType>>(vars);
  const auto& shift = get<typename gr::Tags::Shift<DataType, 3, Frame>>(vars);
  const auto& spatial_metric =
      get<typename gr::Tags::SpatialMetric<DataType, 3, Frame>>(vars);

  // Compute spacetime metric
  tnsr::aa<DataType, 3, Frame> spacetime_metric;
  gr::spacetime_metric(make_not_null(&spacetime_metric), lapse, shift,
                       spatial_metric);

  auto spacetime_metric_as_asymmetric_tensor =
      make_with_value<tnsr::ab<DataType, 3, Frame>>(x, 0.0);
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < 4; ++b) {
      spacetime_metric_as_asymmetric_tensor.get(a, b) =
          spacetime_metric.get(a, b);
    }
  }

  tnsr::ab<DataType, 3, Frame> expected_boosted_spacetime_metric;
  sr::lorentz_boost(make_not_null(&expected_boosted_spacetime_metric),
                    spacetime_metric_as_asymmetric_tensor, -velocity);

  // Boosted metric quantities
  const auto boosted_vars = boosted_solution.variables(x, t, tags_to_test{});
  const auto& lapse_from_boosted_solution =
      get<typename gr::Tags::Lapse<DataType>>(boosted_vars);
  const auto& shift_from_boosted_solution =
      get<typename gr::Tags::Shift<DataType, 3, Frame>>(boosted_vars);
  const auto& spatial_metric_from_boosted_solution =
      get<typename gr::Tags::SpatialMetric<DataType, 3, Frame>>(boosted_vars);

  tnsr::aa<DataType, 3, Frame> spacetime_metric_from_boosted_solution;
  gr::spacetime_metric(make_not_null(&spacetime_metric_from_boosted_solution),
                       lapse_from_boosted_solution, shift_from_boosted_solution,
                       spatial_metric_from_boosted_solution);

  auto spacetime_metric_from_boosted_solution_as_asymmetric_tensor =
      make_with_value<tnsr::ab<DataType, 3, Frame>>(x, 0.0);
  for (size_t a = 0; a < 4; ++a) {
    for (size_t b = 0; b < 4; ++b) {
      spacetime_metric_from_boosted_solution_as_asymmetric_tensor.get(a, b) =
          spacetime_metric_from_boosted_solution.get(a, b);
    }
  }

  CHECK_ITERABLE_APPROX(
      expected_boosted_spacetime_metric,
      spacetime_metric_from_boosted_solution_as_asymmetric_tensor);
}

template <typename Frame>
void test_boosted_on_x_axis() {
  const double mass = 1.7;
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const std::array<double, 3> spin{{-0.6, 0.5, 0.3}};
  const double dimensionless_spin_z = 0.84;
  const double spin_parameter = mass * dimensionless_spin_z;
  const gr::Solutions::KerrSchild rest_solution(mass, spin, center);
  using metric_tags = tmpl::list<gr::Tags::Lapse<double>,
                                 gr::Tags::SpatialMetric<double, 3, Frame>>;
  using derivative_tags = tmpl::append<
      metric_tags,
      tmpl::list<gr::Solutions::KerrSchild::DerivLapse<double, Frame>,
                 gr::Tags::DerivDetSpatialMetric<double, 3, Frame>>>;

  for (const double velocity_x : std::array{0.0, 0.25, 0.5, 0.75, 0.99}) {
    CAPTURE(velocity_x);
    const std::array<double, 3> velocity{{velocity_x, 0.0, 0.0}};
    const double lorentz_factor = 1.0 / sqrt(1.0 - square(velocity_x));
    const gr::Solutions::KerrSchild boosted_solution(mass, spin, center,
                                                     velocity);
    const gr::Solutions::KerrSchild schwarzschild_solution(
        mass, {{0.0, 0.0, 0.0}}, center, velocity);
    const gr::Solutions::KerrSchild spinning_solution(
        mass, {{0.0, 0.0, dimensionless_spin_z}}, center, velocity);

    for (const double radial_offset : std::array{0.5, 1.0, 2.0, 3.0, 5.0}) {
      CAPTURE(radial_offset);
      // To be consistent with the analytic formula for the trace extrinsic
      // curvature (further down) we choose to test for points outside
      // 2M and both a_z / Gamma and a_z * Gamma.
      const double x_coordinate =
          std::max(2.0 * mass, spin_parameter * lorentz_factor) +
          radial_offset * mass;
      const tnsr::I<double, 3, Frame> x{{{x_coordinate, 0.0, 0.0}}};
      const tnsr::I<double, 3, Frame> rest_x{
          {{lorentz_factor * x_coordinate, 0.0, 0.0}}};

      {
        // The spacetime volume element is invariant under a boost.
        // This is because the Lorentz matrix has unit determinant.
        // We compute it using the relation
        // sqrt(-det(spacetime_metric)) = lapse * sqrt(det(spatial_metric))
        const auto rest_vars =
            rest_solution.variables(rest_x, 0.0, metric_tags{});
        const auto boosted_vars =
            boosted_solution.variables(x, 0.0, metric_tags{});
        const double rest_volume_element =
            get(get<gr::Tags::Lapse<double>>(rest_vars)) *
            sqrt(get(determinant(
                get<gr::Tags::SpatialMetric<double, 3, Frame>>(rest_vars))));
        const double boosted_volume_element =
            get(get<gr::Tags::Lapse<double>>(boosted_vars)) *
            sqrt(get(determinant(
                get<gr::Tags::SpatialMetric<double, 3, Frame>>(boosted_vars))));
        CHECK(rest_volume_element == approx(1.0));
        CHECK(boosted_volume_element == approx(rest_volume_element));
      }

      {
        // For a non-spinning Kerr-Schild solution the determinant of the
        // spacetime metric is constant, and equal to -1.
        // Therefore the all spatial derivatives of the spacetime volume element
        // vanish. Here we check the derivative in the x direction.
        // We use the property that
        // sqrt(-det(spacetime_metric)) = lapse * sqrt(det(spatial_metric))
        // to compute the spatial derivative of the left-hand side.
        // This test also checks that the lapse, determinant of the spatial
        // metric and their derivatives are consistent with each other.
        const auto vars =
            schwarzschild_solution.variables(x, 0.0, derivative_tags{});
        const double lapse = get(get<gr::Tags::Lapse<double>>(vars));
        const double sqrt_det_spatial_metric = sqrt(get(
            determinant(get<gr::Tags::SpatialMetric<double, 3, Frame>>(vars))));
        const double deriv_lapse = get<0>(
            get<gr::Solutions::KerrSchild::DerivLapse<double, Frame>>(vars));
        const double deriv_sqrt_det_spatial_metric =
            get<0>(
                get<gr::Tags::DerivDetSpatialMetric<double, 3, Frame>>(vars)) /
            (2.0 * sqrt_det_spatial_metric);
        CHECK(deriv_lapse * sqrt_det_spatial_metric +
                  lapse * deriv_sqrt_det_spatial_metric ==
              approx(0.0));
      }

      {
        // We verified in Mathematica the following analytic formula for the
        // trace of the extrinsic curvature along the positive x-axis,
        // for a black hole with spin along the positive z-direction.
        // We check that the trace of the extrinsic curvature matches with this
        // formula. Specifically, the analytic formula applies applies at
        // t = y = z = 0, x > a_z / Gamma, a_z > 0, with
        // positive x velocity and dimensional Kerr spin a_z = M * chi_z.
        const double s = sqrt(square(lorentz_factor) - 1.0);
        const double rest_x_coordinate = lorentz_factor * x_coordinate;
        const double r =
            sqrt(square(rest_x_coordinate) - square(spin_parameter));
        const double u = lorentz_factor - s * r / rest_x_coordinate;
        const double det_spatial_metric = 1.0 + 2.0 * mass * square(u) / r;
        const double expected_trace =
            mass * u / (square(r) * pow(det_spatial_metric, 1.5)) *
            (2.0 + 6.0 * mass * square(u) / r + s * u * rest_x_coordinate / r +
             2.0 * square(s) * square(spin_parameter) /
                 square(rest_x_coordinate));
        const auto vars = spinning_solution.variables(
            x, 0.0, tmpl::list<gr::Tags::TraceExtrinsicCurvature<double>>{});
        CHECK(get(get<gr::Tags::TraceExtrinsicCurvature<double>>(vars)) ==
              approx(expected_trace));
      }

      {
        // Nullness of the Kerr-Schild one-form implies
        // gamma_ij beta^i beta^j = alpha^{-2} - 2 + alpha^2.
        // This formula follows from noticing that the spacetime metric
        // satisfies g_{tt} + g^{tt} = -2. Then, use g_{tt} = - lapse^2
        // + spatial_metric_{ij} shift^{i} shift^{j} and g^{tt} = -1 / lapse^2,
        // according to the standard 3 +1 decomposition.
        // Check this identity and its ordinary partial derivatives using the
        // same spinning solution as in the trace test above.
        const auto vars = spinning_solution.variables(
            x, 0.0,
            tmpl::list<
                gr::Tags::Lapse<double>, gr::Tags::Shift<double, 3, Frame>,
                gr::Tags::SpatialMetric<double, 3, Frame>,
                gr::Solutions::KerrSchild::DerivLapse<double, Frame>,
                gr::Solutions::KerrSchild::DerivShift<double, Frame>,
                gr::Solutions::KerrSchild::DerivSpatialMetric<double, Frame>,
                Tags::dt<gr::Tags::Lapse<double>>,
                Tags::dt<gr::Tags::Shift<double, 3, Frame>>,
                Tags::dt<gr::Tags::SpatialMetric<double, 3, Frame>>>{});
        const double lapse = get(get<gr::Tags::Lapse<double>>(vars));
        const auto& shift = get<gr::Tags::Shift<double, 3, Frame>>(vars);
        const auto& metric =
            get<gr::Tags::SpatialMetric<double, 3, Frame>>(vars);
        const auto& deriv_lapse =
            get<gr::Solutions::KerrSchild::DerivLapse<double, Frame>>(vars);
        const auto& deriv_shift =
            get<gr::Solutions::KerrSchild::DerivShift<double, Frame>>(vars);
        const auto& deriv_metric =
            get<gr::Solutions::KerrSchild::DerivSpatialMetric<double, Frame>>(
                vars);
        const double dt_lapse =
            get(get<Tags::dt<gr::Tags::Lapse<double>>>(vars));
        const auto& dt_shift =
            get<Tags::dt<gr::Tags::Shift<double, 3, Frame>>>(vars);
        const auto& dt_metric =
            get<Tags::dt<gr::Tags::SpatialMetric<double, 3, Frame>>>(vars);

        double shift_norm_squared = 0.0;
        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = 0; j < 3; ++j) {
            shift_norm_squared +=
                metric.get(i, j) * shift.get(i) * shift.get(j);
          }
        }
        CHECK(shift_norm_squared ==
              approx(square(1.0 - square(lapse)) / square(lapse)));

        // Differentiate both sides, including the derivative of the metric.
        // The shift derivative is constrained along the shift direction.
        const double lapse_derivative_factor =
            2.0 * (lapse - 1.0 / cube(lapse));
        for (size_t a = 0; a < 4; ++a) {
          CAPTURE(a);
          double derivative_residual =
              -lapse_derivative_factor *
              (a == 0 ? dt_lapse : deriv_lapse.get(a - 1));
          for (size_t i = 0; i < 3; ++i) {
            for (size_t j = 0; j < 3; ++j) {
              derivative_residual +=
                  (a == 0 ? dt_metric.get(i, j)
                          : deriv_metric.get(a - 1, i, j)) *
                      shift.get(i) * shift.get(j) +
                  2.0 * metric.get(i, j) * shift.get(i) *
                      (a == 0 ? dt_shift.get(j) : deriv_shift.get(a - 1, j));
            }
          }
          CHECK(derivative_residual == approx(0.0));
        }
      }
    }
  }
}

template <typename Frame, typename DataType>
void test_time_derivative_consistency(const DataType& used_for_size) {
  const double mass = 1.7;
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const auto x = spatial_coords<Frame>(used_for_size);
  using tags =
      tmpl::list<gr::Tags::Lapse<DataType>, gr::Tags::Shift<DataType, 3, Frame>,
                 gr::Solutions::KerrSchild::DerivShift<DataType, Frame>,
                 gr::Tags::SpatialMetric<DataType, 3, Frame>,
                 Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>,
                 gr::Solutions::KerrSchild::DerivSpatialMetric<DataType, Frame>,
                 gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>>;

  for (const auto& spin :
       std::array{std::array{0.0, 0.0, 0.0}, std::array{-0.6, 0.5, 0.3}}) {
    CAPTURE(spin);
    for (const auto& velocity :
         std::array{std::array{0.0, 0.0, 0.0}, std::array{0.4, -0.3, 0.2}}) {
      CAPTURE(velocity);
      const gr::Solutions::KerrSchild solution(mass, spin, center, velocity);
      const auto vars = solution.variables(x, 0.0, tags{});
      const auto& lapse = get(get<gr::Tags::Lapse<DataType>>(vars));
      const auto& shift = get<gr::Tags::Shift<DataType, 3, Frame>>(vars);
      const auto& deriv_shift =
          get<gr::Solutions::KerrSchild::DerivShift<DataType, Frame>>(vars);
      const auto& metric =
          get<gr::Tags::SpatialMetric<DataType, 3, Frame>>(vars);
      const auto& dt_metric =
          get<Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>>(vars);
      const auto& deriv_metric =
          get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataType, Frame>>(
              vars);
      const auto& extrinsic_curvature =
          get<gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>>(vars);

      // Check dt(gamma_ij) = -2 alpha K_ij + Lie_beta(gamma)_ij.
      auto expected_dt_metric =
          make_with_value<tnsr::ii<DataType, 3, Frame>>(used_for_size, 0.0);
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = i; j < 3; ++j) {
          expected_dt_metric.get(i, j) =
              -2.0 * lapse * extrinsic_curvature.get(i, j);
          for (size_t k = 0; k < 3; ++k) {
            expected_dt_metric.get(i, j) +=
                shift.get(k) * deriv_metric.get(k, i, j) +
                metric.get(i, k) * deriv_shift.get(j, k) +
                metric.get(j, k) * deriv_shift.get(i, k);
          }
        }
      }
      CHECK_ITERABLE_APPROX(dt_metric, expected_dt_metric);
    }
  }
}

void test_serialize() {
  gr::Solutions::KerrSchild solution(3.0, {{0.2, 0.3, 0.2}}, {{0.0, 3.0, 4.0}},
                                     {{0.1, 0.2, 0.3}});
  test_serialization(solution);
}

void test_copy_and_move() {
  gr::Solutions::KerrSchild solution(3.0, {{0.2, 0.3, 0.2}}, {{0.0, 3.0, 4.0}},
                                     {{0.1, 0.2, 0.3}});
  test_copy_semantics(solution);
  auto solution_copy = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  const auto created = TestHelpers::test_creation<gr::Solutions::KerrSchild>(
      "Mass: 0.5\n"
      "Spin: [0.1,0.2,0.3]\n"
      "Center: [1.0,3.0,2.0]\n"
      "Velocity: [0.5,0.4,0.3]");
  CHECK(created == gr::Solutions::KerrSchild(0.5, {{0.1, 0.2, 0.3}},
                                             {{1.0, 3.0, 2.0}},
                                             {{0.5, 0.4, 0.3}}));
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
  test_einstein_solution<Frame::Inertial>(true);
  test_einstein_solution<Frame::Inertial>(false);
  test_zero_spin_optimization<Frame::Inertial>(DataVector(5));
  test_zero_spin_optimization<Frame::Inertial>(0.0);
  test_boosted_for_zero_velocity<Frame::Inertial>(DataVector(5));
  test_boosted_for_zero_velocity<Frame::Inertial>(0.0);
  test_boosted_spacetime_metric<Frame::Inertial, double>(0.0);
  test_boosted_on_x_axis<Frame::Inertial>();
  test_time_derivative_consistency<Frame::Inertial>(0.0);
  test_time_derivative_consistency<Frame::Inertial>(DataVector(5));

  test_schwarzschild<Frame::Grid>(DataVector(5));
  test_schwarzschild<Frame::Grid>(0.0);
  test_numerical_deriv_det_spatial_metric<Frame::Grid>(DataVector(5));
  test_tag_retrieval<Frame::Grid>(DataVector(5));
  test_tag_retrieval<Frame::Grid>(0.0);
  test_einstein_solution<Frame::Grid>(true);
  test_einstein_solution<Frame::Grid>(false);
  test_zero_spin_optimization<Frame::Grid>(DataVector(5));
  test_zero_spin_optimization<Frame::Grid>(0.0);
  test_boosted_for_zero_velocity<Frame::Grid>(DataVector(5));
  test_boosted_for_zero_velocity<Frame::Grid>(0.0);
  test_boosted_spacetime_metric<Frame::Grid, double>(0.0);
  test_boosted_on_x_axis<Frame::Grid>();
  test_time_derivative_consistency<Frame::Grid>(0.0);
  test_time_derivative_consistency<Frame::Grid>(DataVector(5));

  CHECK_THROWS_WITH(
      []() {
        gr::Solutions::KerrSchild solution(1.0, {{1.0, 1.0, 1.0}},
                                           {{0.0, 0.0, 0.0}});
      }(),
      Catch::Matchers::ContainsSubstring("Spin magnitude must be < 1"));
  CHECK_THROWS_WITH(
      []() {
        gr::Solutions::KerrSchild solution(-1.0, {{0.0, 0.0, 0.0}},
                                           {{0.0, 0.0, 0.0}});
      }(),
      Catch::Matchers::ContainsSubstring("Mass must be non-negative"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<gr::Solutions::KerrSchild>(
                        "Mass: -0.5\n"
                        "Spin: [0.1,0.2,0.3]\n"
                        "Center: [1.0,3.0,2.0]\n"
                        "Velocity: [0,0,0]"),
                    Catch::Matchers::ContainsSubstring(
                        "Value -0.5 is below the lower bound of 0"));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<gr::Solutions::KerrSchild>(
          "Mass: 0.5\n"
          "Spin: [1.1,0.9,0.3]\n"
          "Center: [1.0,3.0,2.0]\n"
          "Velocity: [0,0,0]"),
      Catch::Matchers::ContainsSubstring("Spin magnitude must be < 1"));
}
