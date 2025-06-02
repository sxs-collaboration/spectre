// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TrumpetSchwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {
struct TrumpetSchwarzschildProxy : gr::Solutions::TrumpetSchwarzschild {
  using gr::Solutions::TrumpetSchwarzschild::TrumpetSchwarzschild;

  template <typename DataType>
  using variables_tags =
      typename gr::Solutions::TrumpetSchwarzschild::template tags<DataType>;

  template <typename DataType>
  tuples::tagged_tuple_from_typelist<variables_tags<DataType>> test_variables(
      const tnsr::I<DataType, 3>& x, const double t) const {
    return this->variables(x, t, variables_tags<DataType>{});
  }
};

using TrumpetSchwarzschild = gr::Solutions::TrumpetSchwarzschild;

template <typename DataType>
using ResultType = tuples::TaggedTuple<
    gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
    gr::Solutions::TrumpetSchwarzschild::DerivLapse<DataType>,
    gr::Tags::Shift<DataType, 3>, ::Tags::dt<gr::Tags::Shift<DataType, 3>>,
    gr::Solutions::TrumpetSchwarzschild::DerivShift<DataType>,
    gr::Tags::SpatialMetric<DataType, 3>,
    ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3>>,
    gr::Solutions::TrumpetSchwarzschild::DerivSpatialMetric<DataType>,
    gr::Tags::SqrtDetSpatialMetric<DataType>,
    gr::Tags::ExtrinsicCurvature<DataType, 3>,
    gr::Tags::InverseSpatialMetric<DataType, 3>>;

template <typename DataType, typename Generator>
void test_trumpet_schwarzschild_random(
    const gr::Solutions::TrumpetSchwarzschild& solution,
    const DataType& used_for_size, const gsl::not_null<Generator*> generator,
    const double random_value_lower_bound,
    const double random_value_upper_bound, const double mass, const double n) {
  // check at random pts
  const std::uniform_real_distribution<> distribution(random_value_lower_bound,
                                                      random_value_upper_bound);

  const auto x = make_with_random_values<tnsr::I<DataType, 3, Frame::Inertial>>(
      generator, distribution, used_for_size);

  const auto py_vars = pypp::call<ResultType<DataType>>(
      "TrumpetSchwarzschild", "trumpet_schwarzschild_variables", x,
      std::numeric_limits<double>::signaling_NaN(), mass, n);

  const auto vars = solution.variables(
      x, std::numeric_limits<double>::signaling_NaN(),
      typename TrumpetSchwarzschild::template tags<DataType>{});

  const Approx custom_approx = Approx::custom().epsilon(1.e-9).scale(1.);
  tmpl::for_each<TrumpetSchwarzschild::tags<DataType>>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<Tag>();
        CAPTURE(tag_name);
        CAPTURE(x);
        CHECK_ITERABLE_CUSTOM_APPROX(tuples::get<Tag>(py_vars), get<Tag>(vars),
                                     custom_approx);
      });
}

template <typename DataType>
void test_trumpet_schwarzschild_fixed(
    const gr::Solutions::TrumpetSchwarzschild& solution, const double mass,
    const double n) {
  // check at fixed pts
  tnsr::I<DataType, 3, Frame::Inertial> x;
  if constexpr (std::is_same<DataType, double>::value) {
    x = tnsr::I<double, 3, Frame::Inertial>{
        {{0.04879024622126530, 0.01279253964649560, 0.05082515227577830}}};
  } else {
    x = tnsr::I<DataVector, 3, Frame::Inertial>{
        {{{51.5006439411830, 0.5680968170041150, 4.180590600246490},
          {33.88201665342590, 0.4970452133901940, 3.029648991759770},
          {13.96500204970360, 0.2830205552189050, 4.848632186557860}}}};
  }
  const auto py_vars = pypp::call<ResultType<DataType>>(
      "TrumpetSchwarzschild", "trumpet_schwarzschild_variables", x,
      std::numeric_limits<double>::signaling_NaN(), mass, n);

  const auto vars = solution.variables(
      x, std::numeric_limits<double>::signaling_NaN(),
      typename TrumpetSchwarzschild::template tags<DataType>{});

  const Approx custom_approx = Approx::custom().epsilon(1.e-9).scale(1.);
  tmpl::for_each<TrumpetSchwarzschild::tags<DataType>>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<Tag>();
        CAPTURE(tag_name);
        CAPTURE(x);
        CHECK_ITERABLE_CUSTOM_APPROX(tuples::get<Tag>(py_vars), get<Tag>(vars),
                                     custom_approx);
      });
}

void test_consistency(const TrumpetSchwarzschild& solution) {
  TestHelpers::VerifyGrSolution::verify_consistency(
      solution, 1.234, tnsr::I<double, 3>{{{1.2, 2.3, 3.4}}}, 0.01, 1.e-8);

  const size_t grid_size = 8;
  const std::array<double, 3> lower_bound{{0.8, 1.22, 1.30}};
  const std::array<double, 3> upper_bound{{0.82, 1.24, 1.32}};
  TestHelpers::VerifyGrSolution::verify_time_independent_einstein_solution(
      solution, grid_size, lower_bound, upper_bound,
      std::numeric_limits<double>::epsilon() * 1e5);
}

void test_serialize(const TrumpetSchwarzschild& solution, const double mass,
                    const double n) {
  test_serialization(solution);
  test_trumpet_schwarzschild_fixed<double>(serialize_and_deserialize(solution),
                                           mass, n);
}

TrumpetSchwarzschild test_construct_from_options(
    const TrumpetSchwarzschild& solution, const double mass, const double n) {
  auto created =
      TestHelpers::test_creation<gr::Solutions::TrumpetSchwarzschild>(
          "Mass: "s + std::to_string(mass) +
          "\n"
          "N: " +
          std::to_string(n));
  CHECK(created == solution);
  return created;
}

void test_copy_and_move(TrumpetSchwarzschild& solution,
                        TrumpetSchwarzschild& solution_copy) {
  test_copy_semantics(solution);
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}
}  // namespace

// [[TimeOut, 40]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.TrumpetSchwarzschild",
    "[PointwiseFunctions][Unit]") {
  const double mass = 2.;
  const double n = 2.;

  {
    gr::Solutions::TrumpetSchwarzschild solution(mass, n);
    MAKE_GENERATOR(generator);

    const pypp::SetupLocalPythonEnvironment local_python_env{
        "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/"};

    // check if environment support Scipy attributes for random grid pts check
    const bool can_import =
        pypp::call<bool>("TrumpetSchwarzschild", "check_import");
    if (can_import) {
      // random grid pts check
      test_trumpet_schwarzschild_random(
          solution, std::numeric_limits<double>::signaling_NaN(),
          make_not_null(&generator), -1.e-4 * mass, -1.e-1 * mass, mass, n);
      test_trumpet_schwarzschild_random(
          solution, DataVector{1, std::numeric_limits<double>::signaling_NaN()},
          make_not_null(&generator), 1.e-1 * mass, 1. * mass, mass, n);
      test_trumpet_schwarzschild_random(
          solution, DataVector{1, std::numeric_limits<double>::signaling_NaN()},
          make_not_null(&generator), 1. * mass, 10. * mass, mass, n);
      test_trumpet_schwarzschild_random(
          solution, DataVector{1, std::numeric_limits<double>::signaling_NaN()},
          make_not_null(&generator), 10. * mass, 100. * mass, mass, n);
      test_trumpet_schwarzschild_random(
          solution, DataVector{1, std::numeric_limits<double>::signaling_NaN()},
          make_not_null(&generator), 100. * mass, 4999 / sqrt(3) * mass, mass,
          n);
    } else {
      // fixed grid pts check
      test_trumpet_schwarzschild_fixed<DataVector>(solution, mass, n);
    }

    test_consistency(solution);
    test_serialize(solution, mass, n);
    auto solution_copy = test_construct_from_options(solution, mass, n);
    test_copy_and_move(solution, solution_copy);
  }

  CHECK_THROWS_WITH(
      []() { const gr::Solutions::TrumpetSchwarzschild solution(0., 1.); }(),
      Catch::Matchers::ContainsSubstring(
          "Black hole mass must be positive, but given "));
  CHECK_THROWS_WITH(
      []() {
        const gr::Solutions::TrumpetSchwarzschild solution(0.1, -0.25);
      }(),
      Catch::Matchers::ContainsSubstring(
          "Parameter n must be non-negative, but given "));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<gr::Solutions::TrumpetSchwarzschild>(
          "Mass: 0.\n"
          "N: 1."),
      Catch::Matchers::ContainsSubstring(
          "Value 0 is below the lower bound of 0.1"));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<gr::Solutions::TrumpetSchwarzschild>(
          "Mass: 1.\n"
          "N: -0.25\n"),
      Catch::Matchers::ContainsSubstring(
          "Value -0.25 is below the lower bound of 0"));
}
