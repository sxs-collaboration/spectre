// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <random>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Strahlkorper.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/Tags.hpp"
#include "ParallelAlgorithms/ApparentHorizonFinder/FastFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

FastFlow::Status do_iteration(
    const gsl::not_null<ylm::Strahlkorper<Frame::Inertial>*> strahlkorper,
    const gsl::not_null<FastFlow*> flow,
    const gr::Solutions::KerrSchild& solution) {
  FastFlow::Status status = FastFlow::Status::SuccessfulIteration;

  while (status == FastFlow::Status::SuccessfulIteration) {
    const auto l_mesh = flow->current_l_mesh(*strahlkorper);
    const auto prolonged_strahlkorper =
        ylm::Strahlkorper<Frame::Inertial>(l_mesh, l_mesh, *strahlkorper);

    const auto box = db::create<
        db::AddSimpleTags<ylm::Tags::items_tags<Frame::Inertial>>,
        db::AddComputeTags<ylm::Tags::compute_items_tags<Frame::Inertial>>>(
        prolonged_strahlkorper);

    const double t = 0.0;
    const auto& cart_coords =
        db::get<ylm::Tags::CartesianCoords<Frame::Inertial>>(box);
    const auto vars = solution.variables(
        cart_coords, t, gr::Solutions::KerrSchild::tags<DataVector>{});

    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
    const auto& deriv_spatial_metric =
        get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(vars);
    const auto inverse_spatial_metric =
        determinant_and_inverse(spatial_metric).second;

    const auto status_and_info = flow->iterate_horizon_finder<Frame::Inertial>(
        strahlkorper, inverse_spatial_metric,
        gr::extrinsic_curvature(
            get<gr::Tags::Lapse<DataVector>>(vars),
            get<gr::Tags::Shift<DataVector, 3>>(vars),
            get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(vars),
            spatial_metric,
            get<Tags::dt<gr::Tags::SpatialMetric<DataVector, 3>>>(vars),
            deriv_spatial_metric),
        raise_or_lower_first_index(
            gr::christoffel_first_kind(deriv_spatial_metric),
            inverse_spatial_metric));
    status = status_and_info.first;
  }
  return status;
}

void test_construct_from_options_fast() {
  const auto created = TestHelpers::test_creation<FastFlow>(
      "Flow: Fast\n"
      "Alpha: 1.1\n"
      "Beta: 0.6\n"
      "AbsTol: 1.e-10\n"
      "TruncationTol: 1.e-3\n"
      "DivergenceTol: 1.1\n"
      "DivergenceIter: 6\n"
      "MaxIts: 200");
  CHECK(created ==
        FastFlow(FastFlow::FlowType::Fast, 1.1, 0.6, 1e-10, 1e-3, 1.1, 6, 200));
}

void test_construct_from_options_jacobi() {
  const auto created = TestHelpers::test_creation<FastFlow>(
      "Flow: Jacobi\n"
      "Alpha: 1.1\n"
      "Beta: 0.6\n"
      "AbsTol: 1.e-10\n"
      "TruncationTol: 1.e-3\n"
      "DivergenceTol: 1.1\n"
      "DivergenceIter: 6\n"
      "MaxIts: 200");
  CHECK(created == FastFlow(FastFlow::FlowType::Jacobi, 1.1, 0.6, 1e-10, 1e-3,
                            1.1, 6, 200));
}

void test_construct_from_options_curvature() {
  const auto created = TestHelpers::test_creation<FastFlow>(
      "Flow: Curvature\n"
      "Alpha: 1.1\n"
      "Beta: 0.6\n"
      "AbsTol: 1.e-10\n"
      "TruncationTol: 1.e-3\n"
      "DivergenceTol: 1.1\n"
      "DivergenceIter: 6\n"
      "MaxIts: 200");
  CHECK(created == FastFlow(FastFlow::FlowType::Curvature, 1.1, 0.6, 1e-10,
                            1e-3, 1.1, 6, 200));
}

void test_serialize() {
  FastFlow fastflow(FastFlow::FlowType::Jacobi, 1.1, 0.6, 1e-10, 1e-3, 1.1, 6,
                    200);
  test_serialization(fastflow);
}

void test_copy_and_move() {
  FastFlow fastflow(FastFlow::FlowType::Curvature, 1.1, 0.6, 1e-10, 1e-3, 1.1,
                    6, 200);
  test_copy_semantics(fastflow);
  auto fastflow_copy = fastflow;
  // clang-tidy: std::move of triviable-copyable type
  test_move_semantics(std::move(fastflow), fastflow_copy);  // NOLINT
}

void test_ostream() {
  CHECK(get_output(FastFlow::FlowType::Curvature) == "Curvature");
  CHECK(get_output(FastFlow::FlowType::Jacobi) == "Jacobi");
  CHECK(get_output(FastFlow::FlowType::Fast) == "Fast");
  CHECK(get_output(FastFlow::Status::SuccessfulIteration) == "Still iterating");
  CHECK(get_output(FastFlow::Status::AbsTol) ==
        "Converged: Absolute tolerance");
  CHECK(get_output(FastFlow::Status::TruncationTol) ==
        "Converged: Truncation tolerance");
  CHECK(get_output(FastFlow::Status::MaxIts) == "Failed: Too many iterations");
  CHECK(get_output(FastFlow::Status::NegativeRadius) ==
        "Failed: Negative radius");
  CHECK(get_output(FastFlow::Status::DivergenceError) == "Failed: Diverging");
}

void test_negative_radius_error() {
  // Set initial Strahlkorper radius to negative on purpose to get
  // error exit status.
  ylm::Strahlkorper<Frame::Inertial> strahlkorper(5, 5, -1.0, {{0, 0, 0}});
  FastFlow flow(FastFlow::FlowType::Fast, 1.0, 0.5, 1e-12, 1e-10, 1.2, 5, 100);

  const gr::Solutions::KerrSchild solution(1.0, {{0., 0., 0.}}, {{0., 0., 0.}});

  const auto status = do_iteration(&strahlkorper, &flow, solution);
  CHECK(status == FastFlow::Status::NegativeRadius);
}

void test_too_many_iterations_error() {
  ylm::Strahlkorper<Frame::Inertial> strahlkorper(5, 5, 3.0, {{0, 0, 0}});
  // Set number of iterations to 1 on purpose to get error exit status.
  FastFlow flow(FastFlow::FlowType::Fast, 1.0, 0.5, 1e-12, 1e-10, 1.2, 5, 1);

  const gr::Solutions::KerrSchild solution(1.0, {{0., 0., 0.}}, {{0., 0., 0.}});

  const auto status = do_iteration(&strahlkorper, &flow, solution);
  CHECK(status == FastFlow::Status::MaxIts);
}

void test_schwarzschild(FastFlow::Flow::type type_of_flow,
                        const size_t max_iterations) {
  ylm::Strahlkorper<Frame::Inertial> strahlkorper(5, 5, 3.0, {{0, 0, 0}});
  FastFlow flow(type_of_flow, 1.0, 0.5, 1e-12, 1e-10, 1.2, 5, max_iterations);

  const gr::Solutions::KerrSchild solution(1.0, {{0., 0., 0.}}, {{0., 0., 0.}});

  const auto iterate_and_check = [&strahlkorper, &flow, &solution]() {
    const auto status = do_iteration(&strahlkorper, &flow, solution);
    CHECK(converged(status));

    const auto box = db::create<
        db::AddSimpleTags<ylm::Tags::items_tags<Frame::Inertial>>,
        db::AddComputeTags<ylm::Tags::compute_items_tags<Frame::Inertial>>>(
        strahlkorper);
    const auto& rad = get(db::get<ylm::Tags::Radius<Frame::Inertial>>(box));
    const auto r_minmax = std::minmax_element(rad.begin(), rad.end());
    Approx custom_approx = Approx::custom().epsilon(1.e-11);
    CHECK(*r_minmax.first == custom_approx(2.0));
    CHECK(*r_minmax.second == custom_approx(2.0));
  };

  iterate_and_check();

  // We have found the horizon once.  Now perturb the strahlkorper
  // and find the horizon again. This checks that fastflow is reset
  // correctly.
  strahlkorper = [](const ylm::Strahlkorper<Frame::Inertial>& strahlkorper_l) {
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist(0.0, 0.1);
    auto coefs = strahlkorper_l.coefficients();
    for (auto coef_iter = ylm::SpherepackIterator(strahlkorper_l.l_max(),
                                                  strahlkorper_l.l_max());
         coef_iter; ++coef_iter) {
      // Change all components randomly, but make smaller changes
      // to higher-order coefficients.
      coefs[coef_iter()] *= 1.0 + dist(generator) / square(coef_iter.l() + 1.0);
    }
    return ylm::Strahlkorper<Frame::Inertial>(coefs, strahlkorper_l);
  }(strahlkorper);

  flow.reset_for_next_find();
  iterate_and_check();
}

void test_kerr(FastFlow::Flow::type type_of_flow, const double mass,
               const size_t max_iterations) {
  ylm::Strahlkorper<Frame::Inertial> strahlkorper(8, 8, 2.0 * mass,
                                                  {{0, 0, 0}});
  FastFlow flow(type_of_flow, 1.0, 0.5, 1e-12, 1e-2, 1.2, 5, max_iterations);

  const std::array<double, 3> spin = {{0.1, 0.2, 0.3}};
  const gr::Solutions::KerrSchild solution(mass, spin, {{0., 0., 0.}});

  const auto status = do_iteration(&strahlkorper, &flow, solution);
  CHECK(converged(status));

  const double spin_magnitude =
      sqrt(square(spin[0]) + square(spin[1]) + square(spin[2]));
  // Once we switch to c++17, use std::hypot(spin[0],spin[1],spin[2]);

  // minimal radius of kerr horizon is M + sqrt(M^2-a^2) (obtained in
  // direction parallel to Spin)
  const double r_min_pt = strahlkorper.radius(acos(spin[2] / spin_magnitude),
                                              atan2(spin[1], spin[0]));
  const double r_min_val = mass * (1.0 + sqrt(1.0 - square(spin_magnitude)));

  const std::array<double, 3> vector_normal_to_spin = {{0.0, -0.3, 0.2}};
  const double vector_magnitude =
      sqrt(square(vector_normal_to_spin[0]) + square(vector_normal_to_spin[1]) +
           square(vector_normal_to_spin[2]));
  // Once we switch to c++17, use std::hypot

  // maximal radius of kerr horizon is sqrt(2M^2 + 2M sqrt(M^2-a^2))
  // (obtained in direction orthogonal to spin)
  const double r_max_pt = strahlkorper.radius(
      acos(vector_normal_to_spin[2] / vector_magnitude),
      atan2(vector_normal_to_spin[1], vector_normal_to_spin[0]));
  const double r_max_val =
      mass * sqrt(2.0 + 2.0 * sqrt(1.0 - square(spin_magnitude)));

  Approx custom_approx = Approx::custom().epsilon(1.e-10).scale(1.);
  CHECK(r_min_pt == custom_approx(r_min_val));
  CHECK(r_max_pt == custom_approx(r_max_val));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FastFlowSchwarzschild",
                  "[Utilities][Unit]") {
  test_schwarzschild(FastFlow::FlowType::Fast, 100);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.JacobiSchwarzschild",
                  "[Utilities][Unit]") {
  test_schwarzschild(FastFlow::FlowType::Jacobi, 200);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.CurvatureSchwarzschild",
                  "[Utilities][Unit]") {
  test_schwarzschild(FastFlow::FlowType::Curvature, 200);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FastFlowKerr",
                  "[Utilities][Unit]") {
  test_kerr(FastFlow::FlowType::Fast, 2.0, 100);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.JacobiKerr",
                  "[Utilities][Unit]") {
  // Keep mass at 1.0 so test doesn't timeout.
  test_kerr(FastFlow::FlowType::Jacobi, 1.0, 200);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.CurvatureKerr",
                  "[Utilities][Unit]") {
  // Keep mass at 1.0 so test doesn't timeout.
  test_kerr(FastFlow::FlowType::Curvature, 1.0, 200);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizonFinder.FastFlowMisc",
                  "[Utilities][Unit]") {
  test_negative_radius_error();
  test_too_many_iterations_error();
  test_construct_from_options_fast();
  test_construct_from_options_jacobi();
  test_construct_from_options_curvature();
  test_copy_and_move();
  test_serialize();
  test_ostream();

  CHECK_THROWS_WITH(
      TestHelpers::test_creation<FastFlow>("Flow: Fast\n"
                                           "Alpha: 1.1\n"
                                           "Beta: 0.6\n"
                                           "AbsTol: 1.e-10\n"
                                           "TruncationTol: 1.e-3\n"
                                           "DivergenceTol: 0.5\n"
                                           "DivergenceIter: 6\n"
                                           "MaxIts: 200"),
      Catch::Matchers::ContainsSubstring(
          "Value 0.5 is below the lower bound of 1."));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<FastFlow>("Flow: Crud\n"
                                           "Alpha: 1.1\n"
                                           "Beta: 0.6\n"
                                           "AbsTol: 1.e-10\n"
                                           "TruncationTol: 1.e-3\n"
                                           "DivergenceTol: 1.1\n"
                                           "DivergenceIter: 6\n"
                                           "MaxIts: 200"),
      Catch::Matchers::ContainsSubstring(
          "Failed to convert \"Crud\" to FastFlow::FlowType"));
}
