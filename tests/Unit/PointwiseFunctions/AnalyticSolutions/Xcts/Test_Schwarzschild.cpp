// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>
#include <tuple>

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Elliptic/Systems/Xcts/Geometry.hpp"
#include "Elliptic/Systems/Xcts/Tags.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/Xcts/VerifySolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags/Conformal.hpp"
#include "PointwiseFunctions/InitialDataUtilities/AnalyticSolution.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace Xcts::Solutions {
namespace {

using field_tags =
    tmpl::list<Tags::ConformalFactor<DataVector>,
               Tags::LapseTimesConformalFactor<DataVector>,
               Tags::ShiftExcess<DataVector, 3, Frame::Inertial>>;
using auxiliary_field_tags =
    tmpl::list<::Tags::deriv<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                             Frame::Inertial>,
               ::Tags::deriv<Tags::LapseTimesConformalFactor<DataVector>,
                             tmpl::size_t<3>, Frame::Inertial>,
               Tags::ShiftStrain<DataVector, 3, Frame::Inertial>>;
using flux_tags =
    tmpl::list<::Tags::Flux<Tags::ConformalFactor<DataVector>, tmpl::size_t<3>,
                            Frame::Inertial>,
               ::Tags::Flux<Tags::LapseTimesConformalFactor<DataVector>,
                            tmpl::size_t<3>, Frame::Inertial>,
               Tags::LongitudinalShiftExcess<DataVector, 3, Frame::Inertial>>;
using background_tags = tmpl::list<
    Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
    Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
    ::Tags::deriv<Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
                  tmpl::size_t<3>, Frame::Inertial>,
    gr::Tags::TraceExtrinsicCurvature<DataVector>,
    ::Tags::deriv<gr::Tags::TraceExtrinsicCurvature<DataVector>,
                  tmpl::size_t<3>, Frame::Inertial>,
    Tags::ShiftBackground<DataVector, 3, Frame::Inertial>,
    Tags::LongitudinalShiftBackgroundMinusDtConformalMetric<DataVector, 3,
                                                            Frame::Inertial>>;
using matter_source_tags = tmpl::list<
    gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>,
    gr::Tags::Conformal<gr::Tags::StressTrace<DataVector>, 0>,
    gr::Tags::Conformal<
        gr::Tags::MomentumDensity<3, Frame::Inertial, DataVector>, 0>>;
using gr_tags = tmpl::list<gr::Tags::Lapse<DataVector>,
                           gr::Tags::Shift<3, Frame::Inertial, DataVector>>;
using derived_tags = tmpl::list<
    Tags::ShiftDotDerivExtrinsicCurvatureTrace<DataVector>,
    Tags::LongitudinalShiftMinusDtConformalMetricSquare<DataVector>,
    Tags::LongitudinalShiftMinusDtConformalMetricOverLapseSquare<DataVector>>;

struct SchwarzschildProxy : Xcts::Solutions::Schwarzschild {
  using Xcts::Solutions::Schwarzschild::Schwarzschild;
  tuples::tagged_tuple_from_typelist<
      tmpl::append<field_tags, auxiliary_field_tags>>
  field_variables(const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return Xcts::Solutions::Schwarzschild::variables(
        x, tmpl::append<field_tags, auxiliary_field_tags>{});
  }
  tuples::tagged_tuple_from_typelist<flux_tags> flux_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return Xcts::Solutions::Schwarzschild::variables(x, flux_tags{});
  }
  tuples::tagged_tuple_from_typelist<background_tags> background_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return Xcts::Solutions::Schwarzschild::variables(x, background_tags{});
  }
  tuples::tagged_tuple_from_typelist<matter_source_tags>
  matter_source_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return Xcts::Solutions::Schwarzschild::variables(x, matter_source_tags{});
  }
  tuples::tagged_tuple_from_typelist<gr_tags> gr_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return Xcts::Solutions::Schwarzschild::variables(x, gr_tags{});
  }
  tuples::tagged_tuple_from_typelist<derived_tags> derived_variables(
      const tnsr::I<DataVector, 3, Frame::Inertial>& x) const {
    return Xcts::Solutions::Schwarzschild::variables(x, derived_tags{});
  }
};

void test_solution(const double mass,
                   const Xcts::Solutions::SchwarzschildCoordinates coords,
                   const double expected_radius_at_horizon,
                   const double expected_conformal_factor_at_horizon,
                   const std::string& py_module,
                   const std::string& options_string) {
  CAPTURE(mass);
  CAPTURE(coords);
  const auto created = TestHelpers::test_factory_creation<
      elliptic::analytic_data::AnalyticSolution, Schwarzschild>(options_string);
  REQUIRE(dynamic_cast<const Schwarzschild*>(created.get()) != nullptr);
  const auto& solution = dynamic_cast<const Schwarzschild&>(*created);
  {
    INFO("Properties");
    CHECK(solution.mass() == mass);
    CHECK(solution.radius_at_horizon() == approx(expected_radius_at_horizon));
  }
  {
    INFO("Semantics");
    test_serialization(solution);
    test_copy_semantics(solution);
    auto move_solution = solution;
    test_move_semantics(std::move(move_solution), solution);
  }
  {
    INFO("Conformal factor at horizon");
    // Also test retrieving doubles, instead of DataVectors
    const auto conformal_factor_at_horizon =
        get<Xcts::Tags::ConformalFactor<double>>(solution.variables(
            tnsr::I<double, 3>{{{solution.radius_at_horizon(), 0., 0.}}},
            tmpl::list<Xcts::Tags::ConformalFactor<double>>{}));
    CHECK(get(conformal_factor_at_horizon) ==
          approx(expected_conformal_factor_at_horizon));
  }
  {
    INFO("Random-value tests");
    const SchwarzschildProxy solution_proxy{mass, coords};
    const double inner_radius = expected_radius_at_horizon;
    const double outer_radius = 3. * expected_radius_at_horizon;
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::field_variables, solution_proxy, py_module,
        {"conformal_factor", "lapse_times_conformal_factor", "shift",
         "conformal_factor_gradient", "lapse_times_conformal_factor_gradient",
         "shift_strain"},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::flux_variables, solution_proxy, py_module,
        {"conformal_factor_gradient", "lapse_times_conformal_factor_gradient",
         "longitudinal_shift"},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::background_variables, solution_proxy, py_module,
        {"conformal_metric", "inv_conformal_metric", "deriv_conformal_metric",
         "extrinsic_curvature_trace", "extrinsic_curvature_trace_gradient",
         "shift_background",
         "longitudinal_shift_background_minus_dt_conformal_metric"},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::matter_source_variables, solution_proxy, py_module,
        {"energy_density", "stress_trace", "momentum_density"},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::gr_variables, solution_proxy, py_module,
        {"lapse", "shift"}, {{{inner_radius, outer_radius}}},
        std::make_tuple(mass), DataVector(5));
    pypp::check_with_random_values<1>(
        &SchwarzschildProxy::derived_variables, solution_proxy, py_module,
        {"shift_dot_extrinsic_curvature_trace_gradient",
         "longitudinal_shift_minus_dt_conformal_metric_square",
         "longitudinal_shift_minus_dt_conformal_metric_over_lapse_square"},
        {{{inner_radius, outer_radius}}}, std::make_tuple(mass), DataVector(5));
  }
  {
    INFO("Verify the solution solves the XCTS equations");
    const double inner_radius = expected_radius_at_horizon;
    const double outer_radius = 3. * expected_radius_at_horizon;
    TestHelpers::Xcts::Solutions::verify_solution<Xcts::Geometry::FlatCartesian,
                                                  0>(
        solution, {{0., 0., 0.}}, inner_radius, outer_radius, 1.e-5);
  }
}

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Xcts.Schwarzschild",
    "[PointwiseFunctions][Unit]") {
  pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/Xcts"};
  test_solution(1., SchwarzschildCoordinates::Isotropic, 0.5, 2.,
                "SchwarzschildIsotropic",
                "Schwarzschild:\n"
                "  Mass: 1.\n"
                "  Coordinates: Isotropic");
  test_solution(0.8, SchwarzschildCoordinates::Isotropic, 0.4, 2.,
                "SchwarzschildIsotropic",
                "Schwarzschild:\n"
                "  Mass: 0.8\n"
                "  Coordinates: Isotropic");
  test_solution(1., SchwarzschildCoordinates::PainleveGullstrand, 2., 1.,
                "SchwarzschildPainleveGullstrand",
                "Schwarzschild:\n"
                "  Mass: 1.\n"
                "  Coordinates: PainleveGullstrand");
  test_solution(0.8, SchwarzschildCoordinates::PainleveGullstrand, 1.6, 1.,
                "SchwarzschildPainleveGullstrand",
                "Schwarzschild:\n"
                "  Mass: 0.8\n"
                "  Coordinates: PainleveGullstrand");
  // For radius of horizon see Eq. (7.37) in https://arxiv.org/abs/gr-qc/0510016
  test_solution(1., SchwarzschildCoordinates::KerrSchildIsotropic,
                1.2727410334221052, 1.2535595643473059,
                "SchwarzschildKerrSchildIsotropic",
                "Schwarzschild:\n"
                "  Mass: 1.\n"
                "  Coordinates: KerrSchildIsotropic");
  test_solution(0.8, SchwarzschildCoordinates::KerrSchildIsotropic,
                1.2727410334221052 * 0.8, 1.2535595643473059,
                "SchwarzschildKerrSchildIsotropic",
                "Schwarzschild:\n"
                "  Mass: 0.8\n"
                "  Coordinates: KerrSchildIsotropic");
}

}  // namespace Xcts::Solutions
