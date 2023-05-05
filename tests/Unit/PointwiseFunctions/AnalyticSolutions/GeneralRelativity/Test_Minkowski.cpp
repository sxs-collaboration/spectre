// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Minkowski.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

namespace {
template <size_t Dim, typename T>
void test_minkowski(const T& value) {
  gr::Solutions::Minkowski<Dim> minkowski{};

  const tnsr::I<T, Dim> x{value};
  const double t = 1.2;

  const auto one = make_with_value<T>(value, 1.);
  const auto zero = make_with_value<T>(value, 0.);

  const auto lapse = get<gr::Tags::Lapse<T>>(
      minkowski.variables(x, t, tmpl::list<gr::Tags::Lapse<T>>{}));
  const auto dt_lapse = get<Tags::dt<gr::Tags::Lapse<T>>>(
      minkowski.variables(x, t, tmpl::list<Tags::dt<gr::Tags::Lapse<T>>>{}));
  const auto d_lapse =
      get<Tags::deriv<gr::Tags::Lapse<T>, tmpl::size_t<Dim>, Frame::Inertial>>(
          minkowski.variables(
              x, t,
              tmpl::list<Tags::deriv<gr::Tags::Lapse<T>, tmpl::size_t<Dim>,
                                     Frame::Inertial>>{}));
  const auto shift = get<gr::Tags::Shift<T, Dim>>(
      minkowski.variables(x, t, tmpl::list<gr::Tags::Shift<T, Dim>>{}));
  const auto dt_shift =
      get<Tags::dt<gr::Tags::Shift<T, Dim>>>(minkowski.variables(
          x, t, tmpl::list<Tags::dt<gr::Tags::Shift<T, Dim>>>{}));
  const auto d_shift = get<
      Tags::deriv<gr::Tags::Shift<T, Dim>, tmpl::size_t<Dim>, Frame::Inertial>>(
      minkowski.variables(
          x, t,
          tmpl::list<Tags::deriv<gr::Tags::Shift<T, Dim>, tmpl::size_t<Dim>,
                                 Frame::Inertial>>{}));
  const auto g = get<gr::Tags::SpatialMetric<T, Dim>>(
      minkowski.variables(x, t, tmpl::list<gr::Tags::SpatialMetric<T, Dim>>{}));
  const auto dt_g =
      get<Tags::dt<gr::Tags::SpatialMetric<T, Dim>>>(minkowski.variables(
          x, t, tmpl::list<Tags::dt<gr::Tags::SpatialMetric<T, Dim>>>{}));
  const auto d_g = get<Tags::deriv<gr::Tags::SpatialMetric<T, Dim>,
                                   tmpl::size_t<Dim>, Frame::Inertial>>(
      minkowski.variables(
          x, t,
          tmpl::list<Tags::deriv<gr::Tags::SpatialMetric<T, Dim>,
                                 tmpl::size_t<Dim>, Frame::Inertial>>{}));
  const auto inv_g =
      get<gr::Tags::InverseSpatialMetric<T, Dim>>(minkowski.variables(
          x, t, tmpl::list<gr::Tags::InverseSpatialMetric<T, Dim>>{}));
  const auto extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<T, Dim>>(minkowski.variables(
          x, t, tmpl::list<gr::Tags::ExtrinsicCurvature<T, Dim>>{}));
  const auto det_g = get<gr::Tags::SqrtDetSpatialMetric<T>>(minkowski.variables(
      x, t, tmpl::list<gr::Tags::SqrtDetSpatialMetric<T>>{}));
  const auto dt_det_g =
      get<Tags::dt<gr::Tags::SqrtDetSpatialMetric<T>>>(minkowski.variables(
          x, t, tmpl::list<Tags::dt<gr::Tags::SqrtDetSpatialMetric<T>>>{}));
  const auto d_det_g =
      get<gr::Tags::DerivDetSpatialMetric<T, Dim>>(minkowski.variables(
          x, t, tmpl::list<gr::Tags::DerivDetSpatialMetric<T, Dim>>{}));
  const auto christoffel_first_kind =
      get<gr::Tags::SpatialChristoffelFirstKind<T, Dim>>(minkowski.variables(
          x, t, tmpl::list<gr::Tags::SpatialChristoffelFirstKind<T, Dim>>{}));
  const auto christoffel_second_kind =
      get<gr::Tags::SpatialChristoffelSecondKind<T, Dim>>(minkowski.variables(
          x, t, tmpl::list<gr::Tags::SpatialChristoffelSecondKind<T, Dim>>{}));
  const auto trace_christoffel = get<
      gr::Tags::TraceSpatialChristoffelSecondKind<T, Dim>>(minkowski.variables(
      x, t, tmpl::list<gr::Tags::TraceSpatialChristoffelSecondKind<T, Dim>>{}));
  const auto trace_extrinsic_curvature =
      get<gr::Tags::TraceExtrinsicCurvature<T>>(minkowski.variables(
          x, t, tmpl::list<gr::Tags::TraceExtrinsicCurvature<T>>{}));

  CHECK(lapse.get() == one);
  CHECK(dt_lapse.get() == zero);
  CHECK(det_g.get() == one);
  CHECK(dt_det_g.get() == zero);
  CHECK(trace_extrinsic_curvature.get() == zero);
  for (size_t i = 0; i < Dim; ++i) {
    CHECK(shift.get(i) == zero);
    CHECK(dt_shift.get(i) == zero);
    CHECK(d_lapse.get(i) == zero);
    CHECK(g.get(i, i) == one);
    CHECK(inv_g.get(i, i) == one);
    CHECK(d_det_g.get(i) == zero);
    CHECK(trace_christoffel.get(i) == zero);
    for (size_t j = 0; j < i; ++j) {
      CHECK(g.get(i, j) == zero);
      CHECK(inv_g.get(i, j) == zero);
    }
    for (size_t j = 0; j < Dim; ++j) {
      CHECK(d_shift.get(i, j) == zero);
      CHECK(dt_g.get(i, j) == zero);
      CHECK(extrinsic_curvature.get(i, j) == zero);
      for (size_t k = 0; k < Dim; ++k) {
        CHECK(christoffel_first_kind.get(i, j, k) == zero);
        CHECK(christoffel_second_kind.get(i, j, k) == zero);
        CHECK(d_g.get(i, j, k) == zero);
      }
    }
  }
  test_serialization(minkowski);
  // test operator !=
  CHECK_FALSE(minkowski != minkowski);

  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      minkowski, x, t,
      typename gr::Solutions::Minkowski<Dim>::template tags<T>{});
}

void test_einstein_solution() {
  gr::Solutions::Minkowski<3> solution{};
  TestHelpers::VerifyGrSolution::verify_consistency(
      solution, 1.234, tnsr::I<double, 3>{{{1.2, 2.3, 3.4}}}, 0.01, 1.0e-10);
  TestHelpers::VerifyGrSolution::verify_time_independent_einstein_solution(
      solution, 8, {{1.2, 2.3, 3.4}}, {{1.22, 2.32, 3.42}}, 1.0e-10);
}

template <size_t Dim>
void test_option_creation() {
  TestHelpers::test_creation<gr::Solutions::Minkowski<Dim>>("");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Gr.Minkowski",
                  "[PointwiseFunctions][Unit]") {
  const double x = 1.2;
  const DataVector x_dv{1., 2., 3.};

  test_minkowski<1>(x);
  test_minkowski<1>(x_dv);
  test_minkowski<2>(x);
  test_minkowski<2>(x_dv);
  test_minkowski<3>(x);
  test_minkowski<3>(x_dv);

  test_einstein_solution();

  test_option_creation<1>();
  test_option_creation<2>();
  test_option_creation<3>();
}
