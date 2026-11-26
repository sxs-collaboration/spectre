// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Trace.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "PointwiseFunctions/AnalyticData/Xcts/KerrSchildTeukolsky.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TeukolskyWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialGuess.hpp"
#include "PointwiseFunctions/Xcts/LongitudinalOperator.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Xcts::AnalyticData {
namespace {

using zero_amplitude_tags = tmpl::list<
    Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
    Tags::ConformalFactorMinusOne<DataVector>,
    Tags::LapseTimesConformalFactorMinusOne<DataVector>,
    Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
    Tags::ShiftExcess<DataVector, 3, Frame::Inertial>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>,
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>, 0>,
    gr::Tags::Conformal<gr::Tags::MomentumDensity<DataVector, 3>, 0>>;

tnsr::I<DataVector, 3, Frame::Inertial> test_coords() {
  tnsr::I<DataVector, 3, Frame::Inertial> x{4_st};
  get<0>(x) = DataVector{{3.1, 5.4, -4.2, 7.8}};
  get<1>(x) = DataVector{{1.2, -2.3, 3.7, -4.1}};
  get<2>(x) = DataVector{{4.5, 6.6, 5.3, -2.9}};
  return x;
}

void test_factory_and_semantics() {
  const std::string options =
      "KerrSchildTeukolsky:\n"
      "  KerrSchild:\n"
      "    Mass: 1.0\n"
      "    Spin: [0., 0., 0.999]\n"
      "    Center: [0., 0., 0.]\n"
      "    Velocity: [0., 0., 0.]\n"
      "  TeukolskyWave:\n"
      "    Amplitude: 0.02\n"
      "    Mode: 2\n"
      "    Parity: odd\n"
      "    Direction: outgoing\n"
      "    Center: [0.1, -0.2, 0.3]\n"
      "    Radius: 20.\n"
      "    Width: 4.\n"
      "  LinearVelocity: [-0.11, 0.1, 0.3]\n";
  const auto created =
      TestHelpers::test_factory_creation<elliptic::analytic_data::InitialGuess,
                                         KerrSchildTeukolsky>(options);
  REQUIRE(dynamic_cast<const KerrSchildTeukolsky*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const KerrSchildTeukolsky&>(*created);

  CHECK(solution.kerr_schild().mass() == 1.0);
  CHECK(solution.kerr_schild().dimensionless_spin() ==
        std::array<double, 3>{{0., 0., 0.999}});
  CHECK(solution.teukolsky_wave().amplitude() == 0.02);
  CHECK(solution.teukolsky_wave().mode() == 2);
  CHECK(solution.teukolsky_wave().parity() == "odd");
  CHECK(solution.teukolsky_wave().direction() == "outgoing");
  CHECK(solution.teukolsky_wave().center() ==
        std::array<double, 3>{{0.1, -0.2, 0.3}});
  CHECK_FALSE(solution.teukolsky_wave().include_minkowski_background());
  CHECK(solution.linear_velocity() == std::array<double, 3>{{-0.11, 0.1, 0.3}});

  test_serialization(solution);
  test_copy_semantics(solution);
  auto move_solution = solution;
  test_move_semantics(std::move(move_solution), solution);
}

void test_zero_amplitude_matches_wrapped_kerr_schild() {
  const gr::Solutions::KerrSchild kerr_schild{
      1.0, {{0., 0., 0.2}}, {{0., 0., 0.}}, {{0., 0., 0.}}};
  const KerrSchildTeukolsky solution{
      kerr_schild,
      gr::Solutions::TeukolskyWave{
          0., 0, "even", "ingoing", {{0., 0., 0.}}, 20., 4., false},
      std::array<double, 3>{{0., 0., 0.}}};
  const Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild> wrapped_kerr{
      kerr_schild};
  const auto x = test_coords();

  const auto vars = solution.variables(x, zero_amplitude_tags{});
  const auto expected = wrapped_kerr.variables(x, zero_amplitude_tags{});

  tmpl::for_each<zero_amplitude_tags>([&vars, &expected](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK_ITERABLE_APPROX(get<tag>(vars), get<tag>(expected));
  });
}

void test_nonzero_teukolsky_perturbation() {
  const gr::Solutions::KerrSchild kerr_schild{
      1.0, {{0., 0., 0.999}}, {{0., 0., 0.}}, {{0., 0., 0.}}};
  const gr::Solutions::TeukolskyWave teukolsky_wave{
      0.02, -1, "even", "ingoing", {{0., 0., 0.}}, 20., 4., false};
  const KerrSchildTeukolsky solution{kerr_schild, teukolsky_wave,
                                     std::array<double, 3>{{0., 0., 0.}}};
  const auto x = test_coords();

  using tags =
      tmpl::list<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                 Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
                 Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
                     DataVector, 3, Frame::Inertial>,
                 gr::Tags::TraceExtrinsicCurvature<DataVector>,
                 ::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>;
  const auto vars = solution.variables(x, tags{});

  const auto kerr_vars = kerr_schild.variables(
      x, 0.,
      tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>{});
  const auto teukolsky_vars = teukolsky_wave.variables(
      x, 0.,
      tmpl::list<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
                 ::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3,
                                                    Frame::Inertial>>>{});

  const auto& conformal_metric =
      get<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(vars);
  const auto& teukolsky_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>(
          teukolsky_vars);
  const auto& kerr_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(kerr_vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      CHECK_ITERABLE_APPROX(conformal_metric.get(i, j),
                            kerr_metric.get(i, j) + teukolsky_metric.get(i, j));
    }
  }

  Scalar<DataVector> expected_k_trace{get_size(get<0>(x))};
  trace(make_not_null(&expected_k_trace),
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(kerr_vars),
        get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(kerr_vars));
  CHECK_ITERABLE_APPROX(
      get(get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(vars)),
      get(expected_k_trace));
  CHECK_ITERABLE_APPROX(
      get(get<::Tags::dt<gr::Tags::TraceExtrinsicCurvature<DataVector>>>(vars)),
      DataVector(get_size(get<0>(x)), 0.));

  const auto& inv_conformal_metric =
      get<Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(vars);
  const auto& longitudinal_shift_background_minus_dt_conformal_metric =
      get<Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<
          DataVector, 3, Frame::Inertial>>(vars);
  Scalar<DataVector> trace_u{get_size(get<0>(x)), 0.};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      get(trace_u) -=
          conformal_metric.get(i, j) *
          longitudinal_shift_background_minus_dt_conformal_metric.get(i, j);
    }
  }
  CHECK_ITERABLE_APPROX(get(trace_u), DataVector(get_size(get<0>(x)), 0.));

  const auto& dt_teukolsky_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>>(
          teukolsky_vars);
  Scalar<DataVector> trace_dt_metric{get_size(get<0>(x)), 0.};
  for (size_t k = 0; k < 3; ++k) {
    for (size_t l = 0; l < 3; ++l) {
      get(trace_dt_metric) +=
          inv_conformal_metric.get(k, l) * dt_teukolsky_metric.get(k, l);
    }
  }
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      DataVector expected = DataVector(get_size(get<0>(x)), 0.);
      for (size_t k = 0; k < 3; ++k) {
        for (size_t l = 0; l < 3; ++l) {
          expected -= inv_conformal_metric.get(i, k) *
                      inv_conformal_metric.get(j, l) *
                      (dt_teukolsky_metric.get(k, l) -
                       conformal_metric.get(k, l) * get(trace_dt_metric) / 3.);
        }
      }
      CHECK_ITERABLE_APPROX(
          longitudinal_shift_background_minus_dt_conformal_metric.get(i, j),
          expected);
    }
  }
}

// The constant linear velocity v0 is the background shift and doesn't change
// the shift excess. These don't need numeric derivatives, so no mesh.
void test_linear_velocity_background_shift() {
  const gr::Solutions::KerrSchild kerr_schild{
      1.0, {{0., 0., 0.9}}, {{0., 0., 0.}}, {{0., 0., 0.}}};
  const gr::Solutions::TeukolskyWave teukolsky_wave{
      0.02, -1, "even", "ingoing", {{0., 0., 0.}}, 20., 4., false};
  const std::array<double, 3> velocity{{-0.11, 0.1, 0.3}};
  const KerrSchildTeukolsky solution{kerr_schild, teukolsky_wave, velocity};
  const KerrSchildTeukolsky solution_at_rest{
      kerr_schild, teukolsky_wave, std::array<double, 3>{{0., 0., 0.}}};
  const auto x = test_coords();
  const size_t num_points = get_size(get<0>(x));

  using shift_background_tag =
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>;
  using deriv_shift_background_tag =
      ::Tags::deriv<shift_background_tag, tmpl::size_t<3>, Frame::Inertial>;
  using shift_excess_tag = Tags::ShiftExcess<DataVector, 3, Frame::Inertial>;
  using tags = tmpl::list<shift_background_tag, deriv_shift_background_tag,
                          shift_excess_tag>;
  const auto vars = solution.variables(x, tags{});
  const auto vars_at_rest = solution_at_rest.variables(x, tags{});

  for (size_t i = 0; i < 3; ++i) {
    CHECK_ITERABLE_APPROX(get<shift_background_tag>(vars).get(i),
                          DataVector(num_points, gsl::at(velocity, i)));
  }
  CHECK_ITERABLE_APPROX(get<deriv_shift_background_tag>(vars),
                        (tnsr::iJ<DataVector, 3>{num_points, 0.}));
  CHECK_ITERABLE_APPROX(get<shift_excess_tag>(vars),
                        get<shift_excess_tag>(vars_at_rest));
}

// Checks the contribution of the background velocity v0 to
// `LongitudinalShiftBackgroundMinusDtConformalMetric`, i.e. its longitudinal
// operator (L v0)^ij. For a constant v0 it consists only of Christoffel terms,
// which need the numeric derivative of the conformal metric. Subtracting the
// solution at rest isolates (L v0)^ij by removing the time derivative of the
// conformal metric, which is tested in `test_nonzero_teukolsky_perturbation`.
void test_linear_velocity_longitudinal_shift() {
  const gr::Solutions::KerrSchild kerr_schild{
      1.0, {{0., 0., 0.9}}, {{0., 0., 0.}}, {{0., 0., 0.}}};
  // Center the pulse on the cube below so the time derivative of the conformal
  // metric is non-negligible there
  const gr::Solutions::TeukolskyWave teukolsky_wave{
      0.02, -1, "even", "ingoing", {{0., 0., 0.}}, 3., 1., false};
  const std::array<double, 3> velocity{{-0.11, 0.1, 0.3}};
  const KerrSchildTeukolsky solution{kerr_schild, teukolsky_wave, velocity};
  const KerrSchildTeukolsky solution_at_rest{
      kerr_schild, teukolsky_wave, std::array<double, 3>{{0., 0., 0.}}};

  // A cube outside the horizon
  const Mesh<3> mesh{8, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);
  using AffineMap = domain::CoordinateMaps::Affine;
  using AffineMap3D =
      domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;
  const domain::CoordinateMap<Frame::ElementLogical, Frame::Inertial,
                              AffineMap3D>
      coord_map{{{-1., 1., 2., 3.}, {-1., 1., 1., 2.}, {-1., 1., -0.5, 0.5}}};
  const auto x = coord_map(logical_coords);
  const auto inv_jacobian = coord_map.inv_jacobian(logical_coords);
  const size_t num_points = mesh.number_of_grid_points();

  using shift_background_tag =
      Tags::ShiftBackground<DataVector, 3, Frame::Inertial>;
  using inv_conformal_metric_tag =
      Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>;
  using christoffel_tag =
      Tags::ConformalChristoffelSecondKind<DataVector, 3, Frame::Inertial>;
  using longitudinal_tag =
      Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                              Frame::Inertial>;
  const auto vars = solution.variables(
      x, mesh, inv_jacobian,
      tmpl::list<shift_background_tag, inv_conformal_metric_tag,
                 christoffel_tag, longitudinal_tag>{});
  const auto vars_at_rest = solution_at_rest.variables(
      x, mesh, inv_jacobian, tmpl::list<longitudinal_tag>{});

  tnsr::II<DataVector, 3> expected{num_points};
  Xcts::longitudinal_operator(
      make_not_null(&expected), get<shift_background_tag>(vars),
      tnsr::iJ<DataVector, 3>{num_points, 0.},
      get<inv_conformal_metric_tag>(vars), get<christoffel_tag>(vars));
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      CHECK_ITERABLE_APPROX(get<longitudinal_tag>(vars).get(i, j) -
                                get<longitudinal_tag>(vars_at_rest).get(i, j),
                            expected.get(i, j));
    }
  }
}

void test_numeric_derivative_requires_mesh() {
  const KerrSchildTeukolsky solution{
      gr::Solutions::KerrSchild{
          1.0, {{0., 0., 0.999}}, {{0., 0., 0.}}, {{0., 0., 0.}}},
      gr::Solutions::TeukolskyWave{
          0.02, -1, "even", "ingoing", {{0., 0., 0.}}, 20., 4., false},
      std::array<double, 3>{{0., 0., 0.}}};
  using deriv_conformal_metric_tag =
      ::Tags::deriv<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                    tmpl::size_t<3>, Frame::Inertial>;
  CHECK_THROWS_WITH(
      solution.variables(test_coords(),
                         tmpl::list<deriv_conformal_metric_tag>{}),
      Catch::Matchers::ContainsSubstring(
          "Need a mesh and a Jacobian for numeric differentiation."));
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticData.Xcts.KerrSchildTeukolsky",
    "[PointwiseFunctions][Unit]") {
  test_factory_and_semantics();
  test_zero_amplitude_matches_wrapped_kerr_schild();
  test_nonzero_teukolsky_perturbation();
  test_linear_velocity_background_shift();
  test_linear_velocity_longitudinal_shift();
  test_numeric_derivative_requires_mesh();
}

}  // namespace Xcts::AnalyticData
