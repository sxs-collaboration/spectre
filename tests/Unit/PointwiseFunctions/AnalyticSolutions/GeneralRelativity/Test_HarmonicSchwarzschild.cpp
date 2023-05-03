// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/GeneralRelativity/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/DerivativesOfSpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeMetric.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// SpEC implementation of HarmonicSchwarzschild
namespace spec {
// Schwarzschild black hole of a t=const slice of time-harmonic
// coordinates.
template <typename Frame, typename DataType>
class HarmonicSchwarzschild {
 public:
  // Constructor
  HarmonicSchwarzschild(const double mass,
                        const std::array<double, 3>& center) {
    mMass = mass;
    mCenter = center;
  }

 public:
  bool IsTimeDependent() const { return false; }

  void SetCoordinates(const tnsr::I<DataType, 3, Frame>& x) {
    mCoords = x;
    const Scalar<DataType> R(
        sqrt(square(get<0>(mCoords) - gsl::at(mCenter, 0)) +
             square(get<1>(mCoords) - gsl::at(mCenter, 1)) +
             square(get<2>(mCoords) - gsl::at(mCenter, 2))));
    get(mMoR) = mMass / get(R);
    get(m2MoRpM) = 2.0 * mMass / (mMass + get(R));
    get(mgrr) = 1.0 + get(m2MoRpM) + square(get(m2MoRpM)) + cube(get(m2MoRpM));
    get(mdgrrdr) = (-0.5 / mMass) * square(get(m2MoRpM)) -
                   (1.0 / mMass) * cube(get(m2MoRpM)) -
                   (1.5 / mMass) * square(square(get(m2MoRpM)));
    for (int i = 0; i < 3; ++i)
      mXoR.get(i) = (mCoords.get(i) - gsl::at(mCenter, i)) / get(R);
  }

  // g(i,j) = g_{ij}
  void LowerMetric(const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> g) const {
    Scalar<DataType> gT(square(1.0 + get(mMoR)));
    Scalar<DataType> f1(get(mgrr) - get(gT));

    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {  // symmetry
        g->get(i, j) = get(f1) * mXoR.get(i) * mXoR.get(j);
        if (i == j)
          g->get(i, j) += get(gT);
      }
    }
  }
  void DtLowerMetric(
      const gsl::not_null<tnsr::ii<DataType, 3, Frame>*> dtg) const {
    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {  // symmetry
        dtg->get(i, j) = get(make_with_value<Scalar<DataType>>(get(mMoR), 0.0));
      }
    }
  }
  // dg(i,j)(k) = \partial_k g_{ij}
  void DerivLowerMetric(
      const gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> dg) const {
    Scalar<DataType> gT(square(1.0 + get(mMoR)));
    Scalar<DataType> dgTdr((-2.0 / mMass) *
                           (square(get(mMoR)) + cube(get(mMoR))));

    Scalar<DataType> f1((1.0 / mMass) * get(mMoR) * (get(mgrr) - get(gT)));
    Scalar<DataType> f2(get(mdgrrdr) - get(dgTdr) - 2.0 * get(f1));

    for (int i = 0; i < 3; ++i) {
      for (int j = i; j < 3; ++j) {  // symmetry
        for (int k = 0; k < 3; ++k) {
          dg->get(k, i, j) = get(f2) * mXoR.get(i) * mXoR.get(j) * mXoR.get(k);
          if (i == k)
            dg->get(k, i, j) += get(f1) * mXoR.get(j);
          if (j == k)
            dg->get(k, i, j) += get(f1) * mXoR.get(i);
          if (i == j)
            dg->get(k, i, j) += get(dgTdr) * mXoR.get(k);
        }
      }
    }
  }

  // N() is the 'N' that appears in \partial_t g_{ij} = - 2 N K_{ij}+...
  void PhysicalLapse(const gsl::not_null<Scalar<DataType>*> N) const {
    get(*N) = 1.0 / sqrt(get(mgrr));
  }
  void DtPhysicalLapse(const gsl::not_null<Scalar<DataType>*> dtN) const {
    get(*dtN) = get(make_with_value<Scalar<DataType>>(get(mMoR), 0.0));
  }
  void DerivPhysicalLapse(
      const gsl::not_null<tnsr::i<DataType, 3, Frame>*> dN) const {
    Scalar<DataType> f1(-0.5 * get(mdgrrdr) / pow(get(mgrr), 1.5));

    for (int i = 0; i < 3; ++i)
      dN->get(i) = get(f1) * mXoR.get(i);
  }

  // beta(i) = \beta^i
  void UpperShift(
      const gsl::not_null<tnsr::I<DataType, 3, Frame>*> beta) const {
    Scalar<DataType> f1(square(get(m2MoRpM)) / get(mgrr));

    for (int i = 0; i < 3; ++i)
      beta->get(i) = get(f1) * mXoR.get(i);
  }
  void DtUpperShift(
      const gsl::not_null<tnsr::I<DataType, 3, Frame>*> dtbeta) const {
    for (int i = 0; i < 3; ++i)
      dtbeta->get(i) = get(make_with_value<Scalar<DataType>>(get(mMoR), 0.0));
  }
  // dbeta(i)(k) = \partial_k beta^i
  void DerivUpperShift(
      const gsl::not_null<tnsr::iJ<DataType, 3, Frame>*> dbeta) const {
    Scalar<DataType> f1((1.0 / mMass) * get(mMoR) * square(get(m2MoRpM)) /
                        get(mgrr));

    Scalar<DataType> f2(-get(f1) -
                        (1.0 / mMass) * cube(get(m2MoRpM)) / get(mgrr) -
                        get(mdgrrdr) * square(get(m2MoRpM) / get(mgrr)));

    for (int i = 0; i < 3; ++i) {
      for (int j = 0; j < 3; ++j) {
        dbeta->get(j, i) = get(f2) * mXoR.get(i) * mXoR.get(j);
        if (i == j)
          dbeta->get(j, i) += get(f1);
      }
    }
  }
  int SpatialDim() const { return 3; }

 private:
  tnsr::I<DataType, 3, Frame> mCoords;
  Scalar<DataType> mMoR;             // M/R
  Scalar<DataType> m2MoRpM;          // 2M/(R+M)
  tnsr::I<DataType, 3, Frame> mXoR;  // x_i/R
  Scalar<DataType> mgrr;
  Scalar<DataType> mdgrrdr;
  double mMass;
  std::array<double, 3> mCenter;
};
}  // namespace spec

// Get test coordinates
template <typename Frame, typename DataType>
tnsr::I<DataType, 3, Frame> spatial_coords(const DataType& used_for_size) {
  auto x = make_with_value<tnsr::I<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0>(x) = 1.32;
  get<1>(x) = 0.82;
  get<2>(x) = 1.24;
  return x;
}

template <typename Frame, typename DataType>
void test_tag_retrieval(const DataType& used_for_size) {
  // Parameters for HarmonicSchwarzschild solution
  const double mass = 1.234;
  const std::array<double, 3> center{{1.0, 2.0, 3.0}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  const gr::Solutions::HarmonicSchwarzschild solution(mass, center);
  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      solution, x, t,
      typename gr::Solutions::HarmonicSchwarzschild::template tags<DataType,
                                                                   Frame>{});
}

void test_serialize() {
  gr::Solutions::HarmonicSchwarzschild solution(3.0, {{0.0, 3.0, 4.0}});
  test_serialization(solution);
}

void test_copy_and_move() {
  gr::Solutions::HarmonicSchwarzschild solution(3.0, {{0.0, 3.0, 4.0}});
  test_copy_semantics(solution);
  auto solution_copy = solution;
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}

void test_construct_from_options() {
  const auto created =
      TestHelpers::test_creation<gr::Solutions::HarmonicSchwarzschild>(
          "Mass: 0.5\n"
          "Center: [1.0,3.0,2.0]");
  CHECK(created ==
        gr::Solutions::HarmonicSchwarzschild(0.5, {{1.0, 3.0, 2.0}}));
}

// Test that computed spacetime quantities are computed as expected. See
// documentation for `gr::Solutions::HarmonicSchwarzschild` to see equations for
// expected quantities.
template <typename Frame, typename DataType>
void test_computed_quantities(const DataType& used_for_size) {
  // Parameters for HarmonicSchwarzschild solution
  const double mass = 1.03;
  const std::array<double, 3> center{{0.2, -0.1, 0.4}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  gr::Solutions::HarmonicSchwarzschild solution(mass, center);

  // Get solution's spacetime quantities
  const auto vars = solution.variables(
      x, t,
      typename gr::Solutions::HarmonicSchwarzschild::tags<DataType, Frame>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& d_lapse =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivLapse<DataType,
                                                                    Frame>>(
          vars);
  const auto& shift = get<gr::Tags::Shift<DataType, 3, Frame>>(vars);
  const auto& d_shift =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivShift<DataType,
                                                                    Frame>>(
          vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<DataType, 3, Frame>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, 3, Frame>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>>(vars);
  const auto& d_spatial_metric =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivSpatialMetric<
          DataType, Frame>>(vars);
  const auto& sqrt_det_spatial_metric =
      get<typename gr::Tags::SqrtDetSpatialMetric<DataType>>(vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, 3, Frame>>(vars);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>>(vars);

  // Check that metric * inverse metric = identity
  auto identity =
      make_with_value<tnsr::iJ<DataType, 3, Frame>>(used_for_size, 0.0);
  get<0, 0>(identity) = 1.0;
  get<1, 1>(identity) = 1.0;
  get<2, 2>(identity) = 1.0;

  tnsr::iJ<DataType, 3, Frame> metric_times_inverse_metric{};
  for (size_t i = 0; i < 3; i++) {
    for (size_t j = 0; j < 3; j++) {
      metric_times_inverse_metric.get(i, j) =
          spatial_metric.get(i, 0) * inverse_spatial_metric.get(0, j);
      for (size_t k = 1; k < 3; k++) {
        metric_times_inverse_metric.get(i, j) +=
            spatial_metric.get(i, k) * inverse_spatial_metric.get(k, j);
      }
    }
  }
  CHECK_ITERABLE_APPROX(metric_times_inverse_metric, identity);

  // Check those quantities that should be zero
  const auto zero = make_with_value<DataType>(x, 0.);
  CHECK(dt_lapse.get() == zero);
  for (size_t i = 0; i < 3; ++i) {
    CHECK(dt_shift.get(i) == zero);
    for (size_t j = 0; j < 3; ++j) {
      CHECK(dt_spatial_metric.get(i, j) == zero);
    }
  }

  // Check remaining quantities

  tnsr::I<DataType, 3, Frame> expected_x_minus_center{};
  for (size_t i = 0; i < 3; ++i) {
    expected_x_minus_center.get(i) = x.get(i) - gsl::at(center, i);
  }

  const DataType expected_r = get(magnitude(expected_x_minus_center));
  const DataType expected_one_over_r_squared = 1.0 / square(expected_r);
  const DataType expected_one_over_r_cubed = 1.0 / cube(expected_r);
  const DataType expected_two_m_over_m_plus_r =
      2.0 * mass / (mass + expected_r);
  const DataType expected_spatial_metric_rr =
      1.0 + expected_two_m_over_m_plus_r +
      square(expected_two_m_over_m_plus_r) + cube(expected_two_m_over_m_plus_r);
  const DataType expected_d_spatial_metric_rr =
      -1.0 / (2.0 * mass) * square(expected_two_m_over_m_plus_r) -
      (1.0 / mass) * cube(expected_two_m_over_m_plus_r) -
      (3.0 / (2.0 * mass)) * pow<4>(expected_two_m_over_m_plus_r);
  const DataType expected_f_0 = square(1 + mass / expected_r);
  const DataType expected_d_f_0 =
      2.0 * (1 + mass / expected_r) * (-mass * expected_one_over_r_squared);
  const DataType expected_f_1 =
      (expected_spatial_metric_rr - expected_f_0) / expected_r;
  const DataType expected_f_2 =
      expected_d_spatial_metric_rr - expected_d_f_0 - 2.0 * expected_f_1;
  const DataType expected_f_3 = square(expected_two_m_over_m_plus_r) /
                                (expected_r * expected_spatial_metric_rr);
  const DataType expected_f_4 =
      -expected_f_3 -
      (1.0 / mass) * cube(expected_two_m_over_m_plus_r) /
          expected_spatial_metric_rr -
      expected_d_spatial_metric_rr *
          square((expected_two_m_over_m_plus_r) / expected_spatial_metric_rr);

  auto expected_lapse = make_with_value<Scalar<DataType>>(x, 0.0);
  get(expected_lapse) = 1.0 / sqrt(expected_spatial_metric_rr);
  CHECK_ITERABLE_APPROX(lapse, expected_lapse);

  tnsr::i<DataType, 3, Frame> expected_d_lapse{};
  for (size_t i = 0; i < 3; ++i) {
    expected_d_lapse.get(i) = -0.5 * cube(get(expected_lapse)) *
                              expected_d_spatial_metric_rr *
                              expected_x_minus_center.get(i) / expected_r;
  }
  CHECK_ITERABLE_APPROX(d_lapse, expected_d_lapse);

  tnsr::I<DataType, 3, Frame> expected_shift{};
  for (size_t i = 0; i < 3; ++i) {
    expected_shift.get(i) = square(expected_two_m_over_m_plus_r) *
                            expected_x_minus_center.get(i) /
                            (expected_r * expected_spatial_metric_rr);
  }
  CHECK_ITERABLE_APPROX(shift, expected_shift);

  tnsr::iJ<DataType, 3, Frame> expected_d_shift{};
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      expected_d_shift.get(k, i) =
          expected_f_4 * expected_x_minus_center.get(i) *
          expected_x_minus_center.get(k) * expected_one_over_r_squared;
      if (i == k) {
        expected_d_shift.get(k, i) += expected_f_3;
      }
    }
  }
  CHECK_ITERABLE_APPROX(d_shift, expected_d_shift);

  tnsr::ii<DataType, 3, Frame> expected_spatial_metric{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      expected_spatial_metric.get(i, j) =
          (expected_spatial_metric_rr - expected_f_0) *
          expected_x_minus_center.get(i) * expected_x_minus_center.get(j) *
          expected_one_over_r_squared;
      if (i == j) {
        expected_spatial_metric.get(i, j) += expected_f_0;
      }
    }
  }
  CHECK_ITERABLE_APPROX(spatial_metric, expected_spatial_metric);

  tnsr::ijj<DataType, 3, Frame> expected_d_spatial_metric{};
  for (size_t k = 0; k < 3; ++k) {
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        expected_d_spatial_metric.get(k, i, j) =
            expected_f_2 * expected_x_minus_center.get(i) *
            expected_x_minus_center.get(j) * expected_x_minus_center.get(k) *
            expected_one_over_r_cubed;
        if (i == k) {
          expected_d_spatial_metric.get(k, i, j) +=
              expected_f_1 * expected_x_minus_center.get(j) / expected_r;
        }
        if (j == k) {
          expected_d_spatial_metric.get(k, i, j) +=
              expected_f_1 * expected_x_minus_center.get(i) / expected_r;
        }
        if (i == j) {
          expected_d_spatial_metric.get(k, i, j) +=
              expected_d_f_0 * expected_x_minus_center.get(k) / expected_r;
        }
      }
    }
  }
  CHECK_ITERABLE_APPROX(d_spatial_metric, expected_d_spatial_metric);

  const auto expected_det_and_inverse_spatial_metric =
      determinant_and_inverse(expected_spatial_metric);
  const auto expected_sqrt_det_spatial_metric =
      sqrt(get(expected_det_and_inverse_spatial_metric.first));
  CHECK_ITERABLE_APPROX(get(sqrt_det_spatial_metric),
                        expected_sqrt_det_spatial_metric);

  const auto& expected_inverse_spatial_metric =
      expected_det_and_inverse_spatial_metric.second;
  CHECK_ITERABLE_APPROX(inverse_spatial_metric,
                        expected_inverse_spatial_metric);

  const auto expected_extrinsic_curvature = gr::extrinsic_curvature(
      expected_lapse, expected_shift, expected_d_shift, expected_spatial_metric,
      tnsr::ii<DataType, 3, Frame>(zero), expected_d_spatial_metric);
  CHECK_ITERABLE_APPROX(extrinsic_curvature, expected_extrinsic_curvature);
}

// Check that SpECTRE implementation matches SpEC implementation
template <typename Frame, typename DataType>
void test_against_spec_impl(const DataType used_for_size) {
  // Parameters for HarmonicSchwarzschild solution
  const double mass = 1.03;
  const std::array<double, 3> center{{0.2, -0.1, 0.4}};
  const auto x = spatial_coords<Frame>(used_for_size);
  const double t = 1.3;

  // Evaluate solution
  gr::Solutions::HarmonicSchwarzschild solution(mass, center);

  // Get solution's spacetime quantities
  const auto vars = solution.variables(
      x, t,
      typename gr::Solutions::HarmonicSchwarzschild::tags<DataType, Frame>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& d_lapse =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivLapse<DataType,
                                                                    Frame>>(
          vars);
  const auto& shift = get<gr::Tags::Shift<DataType, 3, Frame>>(vars);
  const auto& d_shift =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivShift<DataType,
                                                                    Frame>>(
          vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<DataType, 3, Frame>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, 3, Frame>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>>(vars);
  const auto& d_spatial_metric =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivSpatialMetric<
          DataType, Frame>>(vars);

  // Evaluate SpEC solution
  spec::HarmonicSchwarzschild<Frame, DataType> spec_solution(mass, center);
  spec_solution.SetCoordinates(x);

  tnsr::ii<DataType, 3, Frame> spec_spatial_metric{};
  spec_solution.LowerMetric(make_not_null(&spec_spatial_metric));
  tnsr::ii<DataType, 3, Frame> spec_dt_spatial_metric{};
  spec_solution.DtLowerMetric(make_not_null(&spec_dt_spatial_metric));
  tnsr::ijj<DataType, 3, Frame> spec_d_spatial_metric{};
  spec_solution.DerivLowerMetric(make_not_null(&spec_d_spatial_metric));
  Scalar<DataType> spec_lapse{};
  spec_solution.PhysicalLapse(make_not_null(&spec_lapse));
  Scalar<DataType> spec_dt_lapse{};
  spec_solution.DtPhysicalLapse(make_not_null(&spec_dt_lapse));
  tnsr::i<DataType, 3, Frame> spec_d_lapse{};
  spec_solution.DerivPhysicalLapse(make_not_null(&spec_d_lapse));
  tnsr::I<DataType, 3, Frame> spec_shift{};
  spec_solution.UpperShift(make_not_null(&spec_shift));
  tnsr::I<DataType, 3, Frame> spec_dt_shift{};
  spec_solution.DtUpperShift(make_not_null(&spec_dt_shift));
  tnsr::iJ<DataType, 3, Frame> spec_d_shift{};
  spec_solution.DerivUpperShift(make_not_null(&spec_d_shift));

  // Check that SpECTRE implementation matches SpEC implementation
  CHECK_ITERABLE_APPROX(spatial_metric, spec_spatial_metric);
  CHECK_ITERABLE_APPROX(dt_spatial_metric, spec_dt_spatial_metric);
  CHECK_ITERABLE_APPROX(d_spatial_metric, spec_d_spatial_metric);
  CHECK_ITERABLE_APPROX(lapse, spec_lapse);
  CHECK_ITERABLE_APPROX(dt_lapse, spec_dt_lapse);
  CHECK_ITERABLE_APPROX(d_lapse, spec_d_lapse);
  CHECK_ITERABLE_APPROX(shift, spec_shift);
  CHECK_ITERABLE_APPROX(dt_shift, spec_dt_shift);
  CHECK_ITERABLE_APPROX(d_shift, spec_d_shift);
}

template <typename Frame>
void test_einstein_solution() {
  // Parameters for KerrSchild solution
  const double mass = 1.7;
  const std::array<double, 3> center{{0.3, 0.2, 0.4}};
  // Setup grid
  const std::array<double, 3> lower_bound{{0.82, 1.24, 1.32}};
  const double time = -2.8;

  gr::Solutions::HarmonicSchwarzschild solution(mass, center);
  TestHelpers::VerifyGrSolution::verify_consistency(
      solution, time, tnsr::I<double, 3, Frame>{lower_bound}, 0.01, 1.0e-10);
  if constexpr (std::is_same_v<Frame, ::Frame::Inertial>) {
    // Don't look at time-independent solution in other than the inertial
    // frame.
    const size_t grid_size = 8;
    const std::array<double, 3> upper_bound{{0.8, 1.22, 1.30}};
    TestHelpers::VerifyGrSolution::verify_time_independent_einstein_solution(
        solution, grid_size, lower_bound, upper_bound,
        std::numeric_limits<double>::epsilon() * 1.e5);
  }
}

// Check that the solution satisfies the harmonic conditions:
// eq 4.42, 4.44, and 4.45 of \cite BaumgarteShapiro
template <typename Frame, typename DataType>
void test_harmonic_conditions_satisfied(const DataType& used_for_size) {
  // Parameters for HarmonicSchwarzschild solution
  const double mass = 1.21;
  const std::array<double, 3> center{{0.3, 0.1, -0.4}};
  const auto x = spatial_coords<Frame>(used_for_size);
  // Arbitrary time for time-independent solution.
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Evaluate solution
  gr::Solutions::HarmonicSchwarzschild solution(mass, center);

  // Get solution's spacetime quantities
  const auto vars = solution.variables(
      x, t,
      typename gr::Solutions::HarmonicSchwarzschild::tags<DataType, Frame>{});
  const auto& lapse = get<gr::Tags::Lapse<DataType>>(vars);
  const auto& d_lapse =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivLapse<DataType,
                                                                    Frame>>(
          vars);
  const auto& dt_lapse = get<Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  const auto& shift = get<gr::Tags::Shift<DataType, 3, Frame>>(vars);
  const auto& d_shift =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivShift<DataType,
                                                                    Frame>>(
          vars);
  const auto& dt_shift =
      get<Tags::dt<gr::Tags::Shift<DataType, 3, Frame>>>(vars);
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataType, 3, Frame>>(vars);
  const auto& d_spatial_metric =
      get<typename gr::Solutions::HarmonicSchwarzschild::DerivSpatialMetric<
          DataType, Frame>>(vars);
  const auto& dt_spatial_metric =
      get<Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>>>(vars);
  const auto& inverse_spatial_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, 3, Frame>>(vars);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, 3, Frame>>(vars);

  // Check that eq 4.42 of \cite BaumgarteShapiro is satisfied:
  //   \Gamma^i = 0
  const auto spacetime_metric =
      gr::spacetime_metric(lapse, shift, spatial_metric);
  const auto da_spacetime_metric = gr::derivatives_of_spacetime_metric(
      lapse, dt_lapse, d_lapse, shift, dt_shift, d_shift, spatial_metric,
      dt_spatial_metric, d_spatial_metric);
  const auto inverse_spacetime_metric =
      determinant_and_inverse(spacetime_metric).second;
  const auto spacetime_christoffel_second_kind = gr::christoffel_second_kind(
      da_spacetime_metric, inverse_spacetime_metric);
  const auto expected_contracted_spacetime_christoffel_second_kind =
      make_with_value<tnsr::A<DataType, 3, Frame>>(used_for_size, 0.0);

  CHECK_ITERABLE_APPROX(
      tenex::evaluate<ti::A>(
          inverse_spacetime_metric(ti::B, ti::C) *
          spacetime_christoffel_second_kind(ti::A, ti::b, ti::c)),
      expected_contracted_spacetime_christoffel_second_kind);

  // Check that eq 4.44 of \cite BaumgarteShapiro is satisfied:
  //   (\partial_t - \beta^j \partial_j)\alpha = -\alpha^2 K
  CHECK_ITERABLE_APPROX(
      tenex::evaluate(dt_lapse() - shift(ti::J) * d_lapse(ti::j)),
      tenex::evaluate(-square(lapse()) * extrinsic_curvature(ti::i, ti::j) *
                      inverse_spatial_metric(ti::I, ti::J)));

  // Check that eq 4.45 of \cite BaumgarteShapiro is satisfied:
  //   (\partial_t - \beta^j \partial_j)\beta^i =
  //     -\alpha^2(\gamma^{ij} \partial_j ln \alpha -
  //     \gamma^{jk} \Gamma^i_{jk})
  //
  // Note: the textbook incorrectly has a + where there should be a - in front
  // of the trace of the Christoffel symbols
  const auto spatial_christoffel_second_kind =
      gr::christoffel_second_kind(d_spatial_metric, inverse_spatial_metric);

  CHECK_ITERABLE_APPROX(
      tenex::evaluate<ti::I>(dt_shift(ti::I) -
                             shift(ti::J) * d_shift(ti::j, ti::I)),
      tenex::evaluate<ti::I>(
          -square(lapse()) *
          (inverse_spatial_metric(ti::I, ti::J) * d_lapse(ti::j) / lapse() -
           inverse_spatial_metric(ti::J, ti::K) *
               spatial_christoffel_second_kind(ti::I, ti::j, ti::k))));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.HarmonicSchwarzschild",
    "[PointwiseFunctions][Unit]") {
  test_copy_and_move();
  test_serialize();
  test_construct_from_options();

  test_tag_retrieval<Frame::Inertial>(DataVector(5));
  test_tag_retrieval<Frame::Inertial>(0.0);
  test_tag_retrieval<Frame::Grid>(DataVector(5));
  test_tag_retrieval<Frame::Grid>(0.0);

  test_computed_quantities<Frame::Inertial>(DataVector(5));
  test_computed_quantities<Frame::Inertial>(0.0);
  test_computed_quantities<Frame::Grid>(DataVector(5));
  test_computed_quantities<Frame::Grid>(0.0);

  test_against_spec_impl<Frame::Inertial>(DataVector(5));
  test_against_spec_impl<Frame::Inertial>(0.0);
  test_against_spec_impl<Frame::Grid>(DataVector(5));
  test_against_spec_impl<Frame::Grid>(0.0);

  test_einstein_solution<Frame::Grid>();
  test_einstein_solution<Frame::Inertial>();

  test_harmonic_conditions_satisfied<Frame::Inertial>(DataVector(5));
  test_harmonic_conditions_satisfied<Frame::Inertial>(0.0);
  test_harmonic_conditions_satisfied<Frame::Grid>(DataVector(5));
  test_harmonic_conditions_satisfied<Frame::Grid>(0.0);

  CHECK_THROWS_WITH(
      []() {
        gr::Solutions::HarmonicSchwarzschild solution(-1.0, {{0.0, 0.0, 0.0}});
      }(),
      Catch::Contains("Mass must be non-negative"));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<gr::Solutions::HarmonicSchwarzschild>(
          "Mass: -0.5\n"
          "Center: [1.0,3.0,2.0]"),
      Catch::Contains("Value -0.5 is below the lower bound of 0"));
}
