// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>
#include <vector>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TeukolskyWave.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"

namespace {

using TeukolskyWave = gr::Solutions::TeukolskyWave;

template <typename DataType>
using WithoutBackgroundTags = tmpl::list<
    gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
    gr::Tags::Shift<DataType, 3, Frame::Inertial>,
    ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame::Inertial>>,
    gr::Tags::SpatialMetric<DataType, 3, Frame::Inertial>,
    ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame::Inertial>>>;

template <typename DataType>
using WithBackgroundTags = tmpl::flatten<tmpl::list<
    WithoutBackgroundTags<DataType>,
    tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>,
               gr::Tags::ExtrinsicCurvature<DataType, 3, Frame::Inertial>,
               gr::Tags::InverseSpatialMetric<DataType, 3, Frame::Inertial>>>>;

template <typename DataType>
using WithoutBackgroundResultType =
    tuples::tagged_tuple_from_typelist<WithoutBackgroundTags<DataType>>;

template <typename DataType>
using WithBackgroundResultType =
    tuples::tagged_tuple_from_typelist<WithBackgroundTags<DataType>>;

struct SolutionParameters {
  double amplitude{};
  int mode{};
  std::string parity{};
  std::string direction{};
  std::array<double, 3> center{};
  double radius{};
  double width{};
  bool include_minkowski_background{};
  double time{};
};

struct TeukolskyWaveProxy : TeukolskyWave {
  using TeukolskyWave::TeukolskyWave;

  template <typename DataType>
  WithBackgroundResultType<DataType> test_variables_with_background(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, const double t) const {
    return this->variables(x, t, WithBackgroundTags<DataType>{});
  }

  template <typename DataType>
  WithoutBackgroundResultType<DataType> test_variables_without_background(
      const tnsr::I<DataType, 3, Frame::Inertial>& x, const double t) const {
    return this->variables(x, t, WithoutBackgroundTags<DataType>{});
  }
};

std::vector<std::string> with_background_python_functions() {
  return {"teukolsky_wave_lapse",
          "teukolsky_wave_dt_lapse",
          "teukolsky_wave_shift",
          "teukolsky_wave_dt_shift",
          "teukolsky_wave_spatial_metric",
          "teukolsky_wave_dt_spatial_metric",
          "teukolsky_wave_sqrt_det_spatial_metric",
          "teukolsky_wave_extrinsic_curvature",
          "teukolsky_wave_inverse_spatial_metric"};
}

std::vector<std::string> without_background_python_functions() {
  return {"teukolsky_wave_no_background_lapse",
          "teukolsky_wave_no_background_dt_lapse",
          "teukolsky_wave_no_background_shift",
          "teukolsky_wave_no_background_dt_shift",
          "teukolsky_wave_no_background_spatial_metric",
          "teukolsky_wave_no_background_dt_spatial_metric"};
}

SolutionParameters random_solution_parameters(
    gsl::not_null<std::mt19937*> generator,
    const bool include_minkowski_background) {
  std::uniform_real_distribution<double> amplitude_magnitude(1.0e-5, 1.0e-3);
  std::uniform_int_distribution<int> sign_distribution(0, 1);
  std::uniform_int_distribution<int> mode_distribution(-2, 2);
  std::uniform_int_distribution<int> bool_distribution(0, 1);
  std::uniform_real_distribution<double> center_distribution(-2.0, 2.0);
  std::uniform_real_distribution<double> radius_distribution(4.0, 9.0);
  std::uniform_real_distribution<double> width_distribution(0.7, 2.2);
  std::uniform_real_distribution<double> time_distribution(-1.5, 1.5);

  const double sign = sign_distribution(*generator) == 0 ? -1.0 : 1.0;
  return SolutionParameters{
      sign * amplitude_magnitude(*generator),
      mode_distribution(*generator),
      bool_distribution(*generator) == 0 ? "even" : "odd",
      bool_distribution(*generator) == 0 ? "outgoing" : "ingoing",
      {center_distribution(*generator), center_distribution(*generator),
       center_distribution(*generator)},
      radius_distribution(*generator),
      width_distribution(*generator),
      include_minkowski_background,
      time_distribution(*generator)};
}

TeukolskyWaveProxy make_solution(const SolutionParameters& parameters) {
  return {parameters.amplitude, parameters.mode,
          parameters.parity,    parameters.direction,
          parameters.center,    parameters.radius,
          parameters.width,     parameters.include_minkowski_background};
}

template <typename DataType>
tnsr::I<DataType, 3, Frame::Inertial> random_coords(
    gsl::not_null<std::mt19937*> generator, const DataType& used_for_size,
    const double lower = -9.0, const double upper = 9.0) {
  const std::uniform_real_distribution<double> distribution(lower, upper);
  return make_with_random_values<tnsr::I<DataType, 3, Frame::Inertial>>(
      generator, distribution, used_for_size);
}

template <typename DataType>
tnsr::I<DataType, 3, Frame::Inertial> safe_random_coords(
    gsl::not_null<std::mt19937*> generator, const DataType& used_for_size,
    const std::array<double, 3>& center) {
  auto x = random_coords(generator, used_for_size, -3.0, 3.0);
  get<0>(x) += center[0] + 4.0;
  get<1>(x) += center[1];
  get<2>(x) += center[2];
  return x;
}

template <typename DataType>
void test_background_true_basics(const TeukolskyWave& solution,
                                 const double time,
                                 const DataType& used_for_size,
                                 gsl::not_null<std::mt19937*> generator) {
  const auto x =
      safe_random_coords(generator, used_for_size, solution.center());
  const auto vars = solution.variables(x, time, WithBackgroundTags<DataType>{});

  CHECK(solution.include_minkowski_background());
  CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataType>>(vars)),
                        make_with_value<DataType>(used_for_size, 1.0));
  CHECK_ITERABLE_APPROX(get(get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars)),
                        make_with_value<DataType>(used_for_size, 0.0));

  const auto zero_shift =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  const auto& shift = get<gr::Tags::Shift<DataType, 3>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataType, 3>>>(vars);
  CHECK_ITERABLE_APPROX(shift, zero_shift);
  CHECK_ITERABLE_APPROX(dt_shift, zero_shift);

  const auto& gamma = get<gr::Tags::SpatialMetric<DataType, 3>>(vars);
  const auto& dt_gamma =
      get<Tags::dt<gr::Tags::SpatialMetric<DataType, 3>>>(vars);
  const auto det_and_inverse = determinant_and_inverse(gamma);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, 3>>(vars);
  CHECK_ITERABLE_APPROX(inverse_spatial_metric, det_and_inverse.second);
  CHECK_ITERABLE_APPROX(
      get(get<gr::Tags::SqrtDetSpatialMetric<DataType>>(vars)),
      sqrt(get(det_and_inverse.first)));

  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, 3>>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      CHECK_ITERABLE_APPROX(extrinsic_curvature.get(i, j),
                            -0.5 * dt_gamma.get(i, j));
    }
  }
}

template <typename DataType>
void test_background_false_basics(const TeukolskyWave& solution,
                                  const double time,
                                  const DataType& used_for_size,
                                  gsl::not_null<std::mt19937*> generator) {
  const auto x =
      safe_random_coords(generator, used_for_size, solution.center());
  const auto vars =
      solution.variables(x, time, WithoutBackgroundTags<DataType>{});

  CHECK_FALSE(solution.include_minkowski_background());
  CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataType>>(vars)),
                        make_with_value<DataType>(used_for_size, 0.0));
  CHECK_ITERABLE_APPROX(get(get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars)),
                        make_with_value<DataType>(used_for_size, 0.0));

  const auto zero_shift =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(x, 0.0);
  const auto& shift = get<gr::Tags::Shift<DataType, 3>>(vars);
  const auto& dt_shift = get<Tags::dt<gr::Tags::Shift<DataType, 3>>>(vars);
  CHECK_ITERABLE_APPROX(shift, zero_shift);
  CHECK_ITERABLE_APPROX(dt_shift, zero_shift);
}

void test_origin_errors(gsl::not_null<std::mt19937*> generator) {
  const auto parameters = random_solution_parameters(generator, true);
  const double time = parameters.time;
  const TeukolskyWave with_background = make_solution(parameters);
  const TeukolskyWave without_background(
      parameters.amplitude, parameters.mode, parameters.parity,
      parameters.direction, parameters.center, parameters.radius,
      parameters.width, false);

  auto check_origin_error = [&time](const TeukolskyWave& solution,
                                    const auto& x) {
    CHECK_THROWS_WITH(
        (get<gr::Tags::SpatialMetric<typename std::decay_t<decltype(get<0>(x))>,
                                     3>>(
            solution.template variable<gr::Tags::SpatialMetric<
                typename std::decay_t<decltype(get<0>(x))>, 3>>(x, time))),
        Catch::Matchers::ContainsSubstring("radius <=") and
            Catch::Matchers::ContainsSubstring("Minimum radius was"));
  };

  tnsr::I<double, 3, Frame::Inertial> x_at_center{};
  get<0>(x_at_center) = parameters.center[0];
  get<1>(x_at_center) = parameters.center[1];
  get<2>(x_at_center) = parameters.center[2];
  check_origin_error(with_background, x_at_center);
  check_origin_error(without_background, x_at_center);

  tnsr::I<double, 3, Frame::Inertial> x_inside_cutoff{};
  get<0>(x_inside_cutoff) = parameters.center[0] + 0.05;
  get<1>(x_inside_cutoff) = parameters.center[1];
  get<2>(x_inside_cutoff) = parameters.center[2];
  check_origin_error(with_background, x_inside_cutoff);

  tnsr::I<double, 3, Frame::Inertial> x_near_cutoff{};
  get<0>(x_near_cutoff) = parameters.center[0] + 0.09;
  get<1>(x_near_cutoff) = parameters.center[1];
  get<2>(x_near_cutoff) = parameters.center[2];
  check_origin_error(with_background, x_near_cutoff);
  check_origin_error(without_background, x_near_cutoff);

  tnsr::I<DataVector, 3, Frame::Inertial> x_datavector{DataVector(3)};
  get<0>(x_datavector) =
      DataVector{parameters.center[0], parameters.center[0] + 0.2,
                 parameters.center[0] + 1.0};
  get<1>(x_datavector) = DataVector{parameters.center[1], parameters.center[1],
                                    parameters.center[1]};
  get<2>(x_datavector) = DataVector{parameters.center[2], parameters.center[2],
                                    parameters.center[2]};
  check_origin_error(with_background, x_datavector);
}

void test_far_field_background_recovery(
    gsl::not_null<std::mt19937*> generator) {
  // This checks Gaussian localization: far from the pulse center, the
  // perturbation should be negligible and the metric should reduce to the
  // requested background state.
  std::uniform_real_distribution<double> direction_distribution(-1.0, 1.0);
  for (const bool include_background : {true, false}) {
    const auto parameters =
        random_solution_parameters(generator, include_background);
    const TeukolskyWave solution = make_solution(parameters);

    const double propagation_shift =
        parameters.direction == "ingoing" ? parameters.time : -parameters.time;
    const double far_radius =
        parameters.radius - propagation_shift + 12.0 * parameters.width;

    tnsr::I<double, 3, Frame::Inertial> x{};
    get<0>(x) = parameters.center[0] + far_radius;
    get<1>(x) = parameters.center[1] + 0.2 * direction_distribution(*generator);
    get<2>(x) = parameters.center[2] - 0.2 * direction_distribution(*generator);

    const auto vars =
        solution.variables(x, parameters.time, WithoutBackgroundTags<double>{});
    const auto& gamma = get<gr::Tags::SpatialMetric<double, 3>>(vars);
    const auto& dt_gamma =
        get<Tags::dt<gr::Tags::SpatialMetric<double, 3>>>(vars);
    const auto local_approx = approx.custom().margin(1.0e-12);

    CHECK(get<0, 0>(gamma) == local_approx(include_background ? 1.0 : 0.0));
    CHECK(get<1, 1>(gamma) == local_approx(include_background ? 1.0 : 0.0));
    CHECK(get<2, 2>(gamma) == local_approx(include_background ? 1.0 : 0.0));
    CHECK(get<0, 1>(gamma) == local_approx(0.0));
    CHECK(get<0, 2>(gamma) == local_approx(0.0));
    CHECK(get<1, 2>(gamma) == local_approx(0.0));
    CHECK(get<0, 0>(dt_gamma) == local_approx(0.0));
    CHECK(get<1, 1>(dt_gamma) == local_approx(0.0));
    CHECK(get<2, 2>(dt_gamma) == local_approx(0.0));
    CHECK(get<0, 1>(dt_gamma) == local_approx(0.0));
    CHECK(get<0, 2>(dt_gamma) == local_approx(0.0));
    CHECK(get<1, 2>(dt_gamma) == local_approx(0.0));
  }
}

void test_axis_regularization(gsl::not_null<std::mt19937*> generator) {
  const auto parameters = random_solution_parameters(generator, true);
  std::uniform_real_distribution<double> offset_distribution(0.5, 2.0);
  const std::array<double, 2> axis_signs{{-1.0, 1.0}};
  // Use a different offset than the implementation's internal axis-limit
  // stencil so this remains an independent limit check.
  const double rho = 3.0e-8;

  for (const std::string parity : {"even", "odd"}) {
    for (int mode = -2; mode <= 2; ++mode) {
      const TeukolskyWave solution(parameters.amplitude, mode, parity,
                                   parameters.direction, parameters.center,
                                   parameters.radius, parameters.width, true);
      for (const double axis_sign : axis_signs) {
        tnsr::I<double, 3, Frame::Inertial> axis_x{};
        get<0>(axis_x) = parameters.center[0];
        get<1>(axis_x) = parameters.center[1];
        get<2>(axis_x) =
            parameters.center[2] +
            axis_sign * (parameters.radius + offset_distribution(*generator));

        const auto axis_vars = solution.variables(axis_x, parameters.time,
                                                  WithBackgroundTags<double>{});
        const auto& axis_gamma =
            get<gr::Tags::SpatialMetric<double, 3>>(axis_vars);
        const auto& axis_dt_gamma =
            get<Tags::dt<gr::Tags::SpatialMetric<double, 3>>>(axis_vars);
        const auto det_and_inverse = determinant_and_inverse(axis_gamma);
        const auto& inverse_spatial_metric =
            get<gr::Tags::InverseSpatialMetric<double, 3>>(axis_vars);
        CHECK_ITERABLE_APPROX(inverse_spatial_metric, det_and_inverse.second);
        CHECK(get(get<gr::Tags::SqrtDetSpatialMetric<double>>(axis_vars)) >
              0.0);

        const auto near_axis_limit_approx =
            approx.custom().epsilon(1.0e-10).scale(1.0);
        CAPTURE(mode);
        CAPTURE(parity);
        CAPTURE(axis_sign);

        tnsr::I<double, 3, Frame::Inertial> near_axis_x{};
        get<0>(near_axis_x) = parameters.center[0] + rho;
        get<1>(near_axis_x) = parameters.center[1];
        get<2>(near_axis_x) = get<2>(axis_x);
        tnsr::I<double, 3, Frame::Inertial> near_axis_y{};
        get<0>(near_axis_y) = parameters.center[0];
        get<1>(near_axis_y) = parameters.center[1] + rho;
        get<2>(near_axis_y) = get<2>(axis_x);

        const auto near_axis_x_vars = solution.variables(
            near_axis_x, parameters.time, WithBackgroundTags<double>{});
        const auto near_axis_y_vars = solution.variables(
            near_axis_y, parameters.time, WithBackgroundTags<double>{});
        const auto& near_axis_x_gamma =
            get<gr::Tags::SpatialMetric<double, 3>>(near_axis_x_vars);
        const auto& near_axis_x_dt_gamma =
            get<Tags::dt<gr::Tags::SpatialMetric<double, 3>>>(near_axis_x_vars);
        const auto& near_axis_y_gamma =
            get<gr::Tags::SpatialMetric<double, 3>>(near_axis_y_vars);
        const auto& near_axis_y_dt_gamma =
            get<Tags::dt<gr::Tags::SpatialMetric<double, 3>>>(near_axis_y_vars);

        for (size_t i = 0; i < 3; ++i) {
          for (size_t j = i; j < 3; ++j) {
            CHECK(std::isfinite(axis_gamma.get(i, j)));
            CHECK(std::isfinite(axis_dt_gamma.get(i, j)));
            const double near_axis_gamma_limit =
                0.5 *
                (near_axis_x_gamma.get(i, j) + near_axis_y_gamma.get(i, j));
            const double near_axis_dt_gamma_limit =
                0.5 * (near_axis_x_dt_gamma.get(i, j) +
                       near_axis_y_dt_gamma.get(i, j));
            CHECK(near_axis_x_gamma.get(i, j) ==
                  near_axis_limit_approx(near_axis_y_gamma.get(i, j)));
            CHECK(near_axis_x_dt_gamma.get(i, j) ==
                  near_axis_limit_approx(near_axis_y_dt_gamma.get(i, j)));
            CHECK(axis_gamma.get(i, j) ==
                  near_axis_limit_approx(near_axis_gamma_limit));
            CHECK(axis_dt_gamma.get(i, j) ==
                  near_axis_limit_approx(near_axis_dt_gamma_limit));
          }
        }
      }
    }
  }
}

void test_spec_pointwise_agreement() {
  // This regression test checks agreement with SpEC's
  // PointwiseFunctions/AnalyticSolutions/LinearizedGravity/TeukolskyWave.cpp
  // at one representative point for the double-valued no-background case.
  const TeukolskyWave solution{0.01, 2,   "even", "outgoing", {{0.0, 0.0, 0.0}},
                               18.5, 1.2, false};
  tnsr::I<double, 3, Frame::Inertial> x{};
  get<0>(x) = 17.1;
  get<1>(x) = -0.2;
  get<2>(x) = 0.3;

  const auto vars = solution.variables(x, 0.0, WithoutBackgroundTags<double>{});
  const auto& gamma = get<gr::Tags::SpatialMetric<double, 3>>(vars);
  const auto& dt_gamma =
      get<Tags::dt<gr::Tags::SpatialMetric<double, 3>>>(vars);

  const auto spec_approx = approx.custom().epsilon(1.0e-13).scale(1.0);
  CHECK(get<0, 0>(gamma) == spec_approx(3.0726477174852849e-06));
  CHECK(get<0, 1>(gamma) == spec_approx(4.6856056129806923e-06));
  CHECK(get<0, 2>(gamma) == spec_approx(7.7441285036194232e-06));
  CHECK(get<1, 1>(gamma) == spec_approx(4.2473114121358127e-04));
  CHECK(get<1, 2>(gamma) == spec_approx(2.5498181687206425e-07));
  CHECK(get<2, 2>(gamma) == spec_approx(-4.2780378893106667e-04));

  CHECK(get<0, 0>(dt_gamma) == spec_approx(2.0523780729181779e-06));
  CHECK(get<0, 1>(dt_gamma) == spec_approx(-4.9599082769854031e-06));
  CHECK(get<0, 2>(dt_gamma) == spec_approx(-1.2565408165979533e-05));
  CHECK(get<1, 1>(dt_gamma) == spec_approx(-6.2085896232436382e-04));
  CHECK(get<1, 2>(dt_gamma) == spec_approx(-3.2099570756649984e-07));
  CHECK(get<2, 2>(dt_gamma) == spec_approx(6.1880658425144612e-04));
}

template <typename DataType>
void test_python_comparison_with_background(const TeukolskyWaveProxy& solution,
                                            const DataType& used_for_size) {
  for (const std::string parity : {"even", "odd"}) {
    for (int mode = -2; mode <= 2; ++mode) {
      const TeukolskyWaveProxy branch_solution(
          solution.amplitude(), mode, parity, solution.direction(),
          {{20.0, 0.0, 0.0}}, solution.radius(), solution.width(), true);
      CAPTURE(parity);
      CAPTURE(mode);
      pypp::check_with_random_values<2>(
          &TeukolskyWaveProxy::template test_variables_with_background<
              DataType>,
          branch_solution, "TeukolskyWave", with_background_python_functions(),
          {{{-9.0, 9.0}, {-2.0, 2.0}}},
          std::make_tuple(branch_solution.amplitude(), branch_solution.mode(),
                          branch_solution.parity(), branch_solution.direction(),
                          branch_solution.center(), branch_solution.radius(),
                          branch_solution.width()),
          used_for_size);
    }
  }
}

template <typename DataType>
void test_python_comparison_without_background(
    const TeukolskyWaveProxy& solution, const DataType& used_for_size) {
  for (const std::string parity : {"even", "odd"}) {
    for (int mode = -2; mode <= 2; ++mode) {
      const TeukolskyWaveProxy branch_solution(
          solution.amplitude(), mode, parity, solution.direction(),
          {{20.0, 0.0, 0.0}}, solution.radius(), solution.width(), false);
      CAPTURE(parity);
      CAPTURE(mode);
      pypp::check_with_random_values<2>(
          &TeukolskyWaveProxy::template test_variables_without_background<
              DataType>,
          branch_solution, "TeukolskyWave",
          without_background_python_functions(), {{{-9.0, 9.0}, {-2.0, 2.0}}},
          std::make_tuple(branch_solution.amplitude(), branch_solution.mode(),
                          branch_solution.parity(), branch_solution.direction(),
                          branch_solution.center(), branch_solution.radius(),
                          branch_solution.width()),
          used_for_size);
    }
  }
}

void test_construct_from_options() {
  const auto created_solution = TestHelpers::test_creation<TeukolskyWave>(
      "Amplitude: 1e-4\n"
      "Mode: 2\n"
      "Parity: even\n"
      "Direction: outgoing\n"
      "Center: [0.0, 0.0, 0.0]\n"
      "Radius: 8.0\n"
      "Width: 1.5");
  CHECK(created_solution == TeukolskyWave(1.0e-4, 2, "even", "outgoing",
                                          {{0.0, 0.0, 0.0}}, 8.0, 1.5, true));
}

void test_background_false_errors() {
  const TeukolskyWave solution{
      1.0e-4, 2, "even", "outgoing", {{0.0, 0.0, 0.0}}, 8.0, 1.5, false};
  tnsr::I<double, 3, Frame::Inertial> x{};
  get<0>(x) = 6.2;
  get<1>(x) = -1.4;
  get<2>(x) = 3.6;
  const double t = 1.3;
  CHECK_THROWS_WITH(
      (get<gr::Tags::InverseSpatialMetric<double, 3>>(
          solution.template variable<gr::Tags::InverseSpatialMetric<double, 3>>(
              x, t))),
      Catch::Matchers::ContainsSubstring(
          "IncludeMinkowskiBackground must be true to compute inverse spatial "
          "metric"));
  CHECK_THROWS_WITH(
      (get<gr::Tags::SqrtDetSpatialMetric<double>>(
          solution.template variable<gr::Tags::SqrtDetSpatialMetric<double>>(
              x, t))),
      Catch::Matchers::ContainsSubstring("IncludeMinkowskiBackground must be "
                                         "true to compute sqrt(det(gamma))"));
  CHECK_THROWS_WITH(
      (get<gr::Tags::ExtrinsicCurvature<double, 3>>(
          solution.template variable<gr::Tags::ExtrinsicCurvature<double, 3>>(
              x, t))),
      Catch::Matchers::ContainsSubstring(
          "IncludeMinkowskiBackground must be true to compute extrinsic "
          "curvature"));
}

void test_invalid_construction_throws() {
  CHECK_THROWS_WITH(
      []() {
        const TeukolskyWave bad_solution(1.0e-4, 3, "even", "outgoing",
                                         {{0.0, 0.0, 0.0}}, 8.0, 1.5, true);
      }(),
      Catch::Matchers::ContainsSubstring("Mode must lie between -2 and 2"));
  CHECK_THROWS_WITH(
      []() {
        const TeukolskyWave bad_solution(1.0e-4, 2, "even", "outgoing",
                                         {{0.0, 0.0, 0.0}}, 8.0, 0.0, true);
      }(),
      Catch::Matchers::ContainsSubstring("Width must be greater than 0"));
  CHECK_THROWS_WITH(
      []() {
        const TeukolskyWave bad_solution(1.0e-4, 2, "even", "outgoing",
                                         {{0.0, 0.0, 0.0}}, 8.0, -1.0, true);
      }(),
      Catch::Matchers::ContainsSubstring("Width must be greater than 0"));
  CHECK_THROWS_WITH(
      []() {
        const TeukolskyWave bad_solution(1.0e-4, 2, "bad", "outgoing",
                                         {{0.0, 0.0, 0.0}}, 8.0, 1.5, true);
      }(),
      Catch::Matchers::ContainsSubstring(
          "Parity must be either 'even' or 'odd'"));
  CHECK_THROWS_WITH(
      []() {
        const TeukolskyWave bad_solution(1.0e-4, 2, "even", "bad",
                                         {{0.0, 0.0, 0.0}}, 8.0, 1.5, true);
      }(),
      Catch::Matchers::ContainsSubstring(
          "Direction must be either 'outgoing' or 'ingoing'"));
}

void test_serialize_and_copy(gsl::not_null<std::mt19937*> generator) {
  const auto parameters = random_solution_parameters(generator, true);
  const TeukolskyWave solution = make_solution(parameters);
  test_serialization(solution);
  test_copy_semantics(solution);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.TeukolskyWave",
                  "[PointwiseFunctions][Unit]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/"};

  MAKE_GENERATOR(generator);
  const auto with_background_parameters =
      random_solution_parameters(make_not_null(&generator), true);
  const TeukolskyWaveProxy with_background_solution =
      make_solution(with_background_parameters);
  test_background_true_basics(
      with_background_solution, with_background_parameters.time,
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
  test_background_true_basics(
      with_background_solution, with_background_parameters.time,
      std::numeric_limits<double>::signaling_NaN(), make_not_null(&generator));
  test_python_comparison_with_background(
      with_background_solution, std::numeric_limits<double>::signaling_NaN());

  const auto without_background_parameters =
      random_solution_parameters(make_not_null(&generator), false);
  const TeukolskyWaveProxy without_background_solution =
      make_solution(without_background_parameters);
  test_background_false_basics(
      without_background_solution, without_background_parameters.time,
      DataVector(5, std::numeric_limits<double>::signaling_NaN()),
      make_not_null(&generator));
  test_background_false_basics(
      without_background_solution, without_background_parameters.time,
      std::numeric_limits<double>::signaling_NaN(), make_not_null(&generator));
  test_python_comparison_without_background(
      without_background_solution,
      std::numeric_limits<double>::signaling_NaN());

  test_origin_errors(make_not_null(&generator));
  test_far_field_background_recovery(make_not_null(&generator));
  test_axis_regularization(make_not_null(&generator));
  test_spec_pointwise_agreement();
  test_construct_from_options();
  test_background_false_errors();
  test_invalid_construction_throws();
  test_serialize_and_copy(make_not_null(&generator));
}
