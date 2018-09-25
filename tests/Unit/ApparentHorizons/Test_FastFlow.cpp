// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <string>
#include <utility>

#include "ApparentHorizons/FastFlow.hpp"
#include "ApparentHorizons/Strahlkorper.hpp"
#include "ApparentHorizons/Tags.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/ComputeSpacetimeQuantities.hpp"
#include "PointwiseFunctions/GeneralRelativity/IndexManipulation.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {

FastFlow::Status do_iteration(
    const gsl::not_null<Strahlkorper<Frame::Inertial>*> strahlkorper,
    const gsl::not_null<FastFlow*> flow,
    const gr::Solutions::KerrSchild& solution) {
  FastFlow::Status status = FastFlow::Status::SuccessfulIteration;

  while (status == FastFlow::Status::SuccessfulIteration) {
    const auto l_mesh = flow->current_l_mesh(*strahlkorper);
    const auto prolonged_strahlkorper =
        Strahlkorper<Frame::Inertial>(l_mesh, l_mesh, *strahlkorper);

    const auto box = db::create<
        db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
        db::AddComputeTags<
            StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(
        prolonged_strahlkorper);

    const double t = 0.0;
    const auto& cart_coords =
        db::get<StrahlkorperTags::CartesianCoords<Frame::Inertial>>(box);
    const auto vars = solution.variables(
        cart_coords, t, gr::Solutions::KerrSchild::tags<DataVector>{});

    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>(vars);
    const auto& deriv_spatial_metric =
        get<gr::Solutions::KerrSchild::DerivSpatialMetric<DataVector>>(vars);
    const auto inverse_spatial_metric =
        determinant_and_inverse(spatial_metric).second;

    const auto status_and_info = flow->iterate_horizon_finder<Frame::Inertial>(
        strahlkorper, inverse_spatial_metric,
        gr::extrinsic_curvature(
            get<gr::Tags::Lapse<DataVector>>(vars),
            get<gr::Tags::Shift<3, Frame::Inertial, DataVector>>(vars),
            get<gr::Solutions::KerrSchild::DerivShift<DataVector>>(vars),
            spatial_metric,
            get<Tags::dt<
                gr::Tags::SpatialMetric<3, Frame::Inertial, DataVector>>>(vars),
            deriv_spatial_metric),
        raise_or_lower_first_index(
            gr::christoffel_first_kind(deriv_spatial_metric),
            inverse_spatial_metric));
    status = status_and_info.first;
  }
  return status;
}

struct FastFlowFromOpts {
  using type = FastFlow;
  static constexpr OptionString help{"FastFlow horizon-finder"};
};

void test_construct_from_options_fast() {
  Options<tmpl::list<FastFlowFromOpts>> opts("");
  opts.parse(
      "FastFlowFromOpts:\n"
      "  Flow: Fast\n"
      "  Alpha: 1.1\n"
      "  Beta: 0.6\n"
      "  AbsTol: 1.e-10\n"
      "  TruncationTol: 1.e-3\n"
      "  DivergenceTol: 1.1\n"
      "  DivergenceIter: 6\n"
      "  MaxIts: 200");
  CHECK(opts.get<FastFlowFromOpts>() ==
        FastFlow(FastFlow::FlowType::Fast, 1.1, 0.6, 1e-10, 1e-3, 1.1, 6, 200));
}

void test_construct_from_options_jacobi() {
  Options<tmpl::list<FastFlowFromOpts>> opts("");
  opts.parse(
      "FastFlowFromOpts:\n"
      "  Flow: Jacobi\n"
      "  Alpha: 1.1\n"
      "  Beta: 0.6\n"
      "  AbsTol: 1.e-10\n"
      "  TruncationTol: 1.e-3\n"
      "  DivergenceTol: 1.1\n"
      "  DivergenceIter: 6\n"
      "  MaxIts: 200");
  CHECK(opts.get<FastFlowFromOpts>() == FastFlow(FastFlow::FlowType::Jacobi,
                                                 1.1, 0.6, 1e-10, 1e-3, 1.1, 6,
                                                 200));
}

void test_construct_from_options_curvature() {
  Options<tmpl::list<FastFlowFromOpts>> opts("");
  opts.parse(
      "FastFlowFromOpts:\n"
      "  Flow: Curvature\n"
      "  Alpha: 1.1\n"
      "  Beta: 0.6\n"
      "  AbsTol: 1.e-10\n"
      "  TruncationTol: 1.e-3\n"
      "  DivergenceTol: 1.1\n"
      "  DivergenceIter: 6\n"
      "  MaxIts: 200");
  CHECK(opts.get<FastFlowFromOpts>() == FastFlow(FastFlow::FlowType::Curvature,
                                                 1.1, 0.6, 1e-10, 1e-3, 1.1, 6,
                                                 200));
}

void test_serialize() noexcept {
  FastFlow fastflow(FastFlow::FlowType::Jacobi, 1.1, 0.6, 1e-10, 1e-3, 1.1, 6,
                    200);
  test_serialization(fastflow);
}

void test_copy_and_move() noexcept {
  FastFlow fastflow(FastFlow::FlowType::Curvature, 1.1, 0.6, 1e-10, 1e-3, 1.1,
                    6, 200);
  test_copy_semantics(fastflow);
  auto fastflow_copy = fastflow;
  // clang-tidy: std::move of triviable-copyable type
  test_move_semantics(std::move(fastflow), fastflow_copy);  // NOLINT
}

void test_ostream() noexcept {
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
  Strahlkorper<Frame::Inertial> strahlkorper(5, 5, -1.0, {{0, 0, 0}});
  FastFlow flow(FastFlow::FlowType::Fast, 1.0, 0.5, 1e-12, 1e-10, 1.2, 5, 100);

  const gr::Solutions::KerrSchild solution(1.0, {{0., 0., 0.}}, {{0., 0., 0.}});

  const auto status = do_iteration(&strahlkorper, &flow, solution);
  CHECK(status == FastFlow::Status::NegativeRadius);
}

void test_too_many_iterations_error() {
  Strahlkorper<Frame::Inertial> strahlkorper(5, 5, 3.0, {{0, 0, 0}});
  // Set number of iterations to 1 on purpose to get error exit status.
  FastFlow flow(FastFlow::FlowType::Fast, 1.0, 0.5, 1e-12, 1e-10, 1.2, 5, 1);

  const gr::Solutions::KerrSchild solution(1.0, {{0., 0., 0.}}, {{0., 0., 0.}});

  const auto status = do_iteration(&strahlkorper, &flow, solution);
  CHECK(status == FastFlow::Status::MaxIts);
}

void test_schwarzschild(FastFlow::Flow::type type_of_flow,
                        const size_t max_iterations) {
  Strahlkorper<Frame::Inertial> strahlkorper(5, 5, 3.0, {{0, 0, 0}});
  FastFlow flow(type_of_flow, 1.0, 0.5, 1e-12, 1e-10, 1.2, 5, max_iterations);

  const gr::Solutions::KerrSchild solution(1.0, {{0., 0., 0.}}, {{0., 0., 0.}});

  const auto status = do_iteration(&strahlkorper, &flow, solution);
  CHECK(converged(status));

  const auto box = db::create<
      db::AddSimpleTags<StrahlkorperTags::items_tags<Frame::Inertial>>,
      db::AddComputeTags<
          StrahlkorperTags::compute_items_tags<Frame::Inertial>>>(strahlkorper);
  const auto& rad = db::get<StrahlkorperTags::Radius<Frame::Inertial>>(box);
  const auto r_minmax = std::minmax_element(rad.begin(), rad.end());
  Approx custom_approx = Approx::custom().epsilon(1.e-11);
  CHECK(*r_minmax.first == custom_approx(2.0));
  CHECK(*r_minmax.second == custom_approx(2.0));
}

void test_kerr(FastFlow::Flow::type type_of_flow, const double mass,
               const size_t max_iterations) {
  Strahlkorper<Frame::Inertial> strahlkorper(8, 8, 2.0 * mass, {{0, 0, 0}});
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

SPECTRE_TEST_CASE("Unit.ApparentHorizons.FastFlowSchwarzschild",
                  "[Utilities][Unit]") {
  test_schwarzschild(FastFlow::FlowType::Fast, 100);
  test_schwarzschild(FastFlow::FlowType::Jacobi, 200);
  test_schwarzschild(FastFlow::FlowType::Curvature, 200);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.FastFlowKerr", "[Utilities][Unit]") {
  test_kerr(FastFlow::FlowType::Fast, 2.0, 100);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.JacobiKerr", "[Utilities][Unit]") {
  // Keep mass at 1.0 so test doesn't timeout.
  test_kerr(FastFlow::FlowType::Jacobi, 1.0, 200);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.CurvatureKerr", "[Utilities][Unit]") {
  // Keep mass at 1.0 so test doesn't timeout.
  test_kerr(FastFlow::FlowType::Curvature, 1.0, 200);
}

SPECTRE_TEST_CASE("Unit.ApparentHorizons.FastFlowMisc", "[Utilities][Unit]") {
  test_negative_radius_error();
  test_too_many_iterations_error();
  test_construct_from_options_fast();
  test_construct_from_options_jacobi();
  test_construct_from_options_curvature();
  test_copy_and_move();
  test_serialize();
  test_ostream();
}

// [[OutputRegex, Value 0.5 is below the lower bound of 1.]]
SPECTRE_TEST_CASE("Unit.ApparentHorizons.FastFlowOptFail",
                  "[Utilities][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<FastFlowFromOpts>> opts("");
  opts.parse(
      "FastFlowFromOpts:\n"
      "  Flow: Fast\n"
      "  Alpha: 1.1\n"
      "  Beta: 0.6\n"
      "  AbsTol: 1.e-10\n"
      "  TruncationTol: 1.e-3\n"
      "  DivergenceTol: 0.5\n"
      "  DivergenceIter: 6\n"
      "  MaxIts: 200");
  opts.get<FastFlowFromOpts>();
}

// [[OutputRegex, Failed to convert "Crud" to FastFlow::FlowType]]
SPECTRE_TEST_CASE("Unit.ApparentHorizons.FastFlowOptFail2",
                  "[Utilities][Unit]") {
  ERROR_TEST();
  Options<tmpl::list<FastFlowFromOpts>> opts("");
  opts.parse(
      "FastFlowFromOpts:\n"
      "  Flow: Crud\n"
      "  Alpha: 1.1\n"
      "  Beta: 0.6\n"
      "  AbsTol: 1.e-10\n"
      "  TruncationTol: 1.e-3\n"
      "  DivergenceTol: 1.1\n"
      "  DivergenceIter: 6\n"
      "  MaxIts: 200");
  opts.get<FastFlowFromOpts>();
}
