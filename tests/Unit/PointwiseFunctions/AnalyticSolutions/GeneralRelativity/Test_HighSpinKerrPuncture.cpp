// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <limits>
#include <random>
#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/Pypp.hpp"
#include "Framework/PyppFundamentals.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/VerifyGrSolution.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HighSpinKerrPuncture.hpp"
#include "PointwiseFunctions/GeneralRelativity/ExtrinsicCurvature.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Serialization/Serialize.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using HighSpinKerrPuncture = gr::Solutions::HighSpinKerrPuncture;

// TaggedTuple in the same tag order as HighSpinKerrPuncture::tags, matching the
// dict returned by high_spin_kerr_puncture_variables in
// HighSpinKerrPuncture.py.
template <typename DataType>
using ResultType = tuples::TaggedTuple<
    gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
    HighSpinKerrPuncture::DerivLapse<DataType>, gr::Tags::Shift<DataType, 3>,
    ::Tags::dt<gr::Tags::Shift<DataType, 3>>,
    HighSpinKerrPuncture::DerivShift<DataType>,
    gr::Tags::SpatialMetric<DataType, 3>,
    ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3>>,
    HighSpinKerrPuncture::DerivSpatialMetric<DataType>,
    gr::Tags::SqrtDetSpatialMetric<DataType>,
    gr::Tags::ExtrinsicCurvature<DataType, 3>,
    gr::Tags::InverseSpatialMetric<DataType, 3>>;

double r_plus_of(const double mass, const double dimensionless_spin) {
  return mass * (1. + sqrt(1. - square(dimensionless_spin)));
}

// Build a single Cartesian point with coordinate radius `radius` in a random
// direction. theta stays away from the axis because the Python reference's
// Jacobians carry 1/sin(theta), whose roundoff amplifies;
// the axis itself is covered separately in individual tests.
template <typename Generator>
std::array<double, 3> point_at_radius(const gsl::not_null<Generator*> generator,
                                      const double radius) {
  const double cos_theta =
      std::cos(std::uniform_real_distribution<>{0.15, M_PI - 0.15}(*generator));
  const double phi =
      std::uniform_real_distribution<>{0., 2. * M_PI}(*generator);
  const double sin_theta = sqrt(1. - square(cos_theta));
  return {{radius * sin_theta * std::cos(phi),
           radius * sin_theta * std::sin(phi), radius * cos_theta}};
}

// Largest absolute component of a tensor of doubles or DataVectors.
template <typename TensorType>
double max_abs_component(const TensorType& tensor) {
  double result = 0.;
  for (auto it = tensor.begin(); it != tensor.end(); ++it) {
    using std::abs;
    result = std::max(result, max(abs(*it)));
  }
  return result;
}

// pypp comparison of all twelve tags against the independent Python reference.
template <typename DataType>
void test_pypp(const HighSpinKerrPuncture& solution,
               const tnsr::I<DataType, 3, Frame::Inertial>& x,
               const double mass, const double dimensionless_spin) {
  const double t = std::numeric_limits<double>::signaling_NaN();
  const auto py_vars = pypp::call<ResultType<DataType>>(
      "HighSpinKerrPuncture", "high_spin_kerr_puncture_variables", x, t, mass,
      dimensionless_spin);
  const auto vars = solution.variables(
      x, t, typename HighSpinKerrPuncture::template tags<DataType>{});
  tmpl::for_each<typename HighSpinKerrPuncture::template tags<DataType>>(
      [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
        const std::string tag_name = db::tag_name<Tag>();
        CAPTURE(tag_name);
        CAPTURE(x);
        CAPTURE(mass);
        CAPTURE(dimensionless_spin);
        const double tag_scale =
            std::max(1., std::max(max_abs_component(tuples::get<Tag>(py_vars)),
                                  max_abs_component(get<Tag>(vars))));
        const Approx tag_approx =
            Approx::custom().epsilon(1.e-12).scale(tag_scale);
        CHECK_ITERABLE_CUSTOM_APPROX(tuples::get<Tag>(py_vars), get<Tag>(vars),
                                     tag_approx);
      });
}

// Item 1: pypp random-value comparison. For each (mass, chi), sample
// points in radial shells on BOTH sheets: deep inside the throat, a shell
// straddling the throat sphere, near-throat outer, intermediate, and far field.
// The DataVector case packs one point per shell; the double case checks a
// single generic point per (mass, chi).
template <typename Generator>
void test_pypp_all_shells(const gsl::not_null<Generator*> generator,
                          const double mass, const double dimensionless_spin) {
  const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);

  // Radial-shell bounds as multiples of the throat radius (inner sheet:
  // r < throat; outer sheet: r > throat). With the signed lapse every tag is
  // smooth across the throat sphere, so one shell deliberately straddles it.
  const std::array<std::array<double, 2>, 5> shells{{
      {{0.05, 0.8}},  // deep inside the throat
      {{0.9, 1.1}},   // straddling the throat
      {{1.1, 3.}},    // near-throat outer up to intermediate
      {{3., 12.}},    // intermediate
      {{40., 200.}},  // far field
  }};

  // DataVector case: one point per shell.
  const size_t num_shells = shells.size();
  tnsr::I<DataVector, 3, Frame::Inertial> x_dv(num_shells);
  for (size_t shell = 0; shell < num_shells; ++shell) {
    const double lower = throat * gsl::at(shells, shell)[0];
    const double upper = throat * gsl::at(shells, shell)[1];
    const double radius =
        std::uniform_real_distribution<>{lower, upper}(*generator);
    const auto point = point_at_radius(generator, radius);
    for (size_t i = 0; i < 3; ++i) {
      x_dv.get(i)[shell] = gsl::at(point, i);
    }
  }
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  test_pypp<DataVector>(solution, x_dv, mass, dimensionless_spin);

  // double case: one generic outer-sheet point (intermediate shell), in
  // [3, 12] throat radii.
  const double radius_double =
      throat * std::uniform_real_distribution<>{3., 12.}(*generator);
  const auto point_double = point_at_radius(generator, radius_double);
  tnsr::I<double, 3, Frame::Inertial> x_double{};
  for (size_t i = 0; i < 3; ++i) {
    x_double.get(i) = gsl::at(point_double, i);
  }
  test_pypp<double>(solution, x_double, mass, dimensionless_spin);
}

// Item 2: verify_spatial_consistency (FD checks of the derivative tags and
// the metric/inverse/sqrt-det relations) plus the stationary ADM identity
// 2 alpha K_ij = beta^k d_k gamma_ij + gamma_ki d_j beta^k
// + gamma_kj d_i beta^k on both sheets.
// The ADM identity is checked against gr::extrinsic_curvature
// rather than via verify_consistency, whose hardcoded near-machine
// tolerance is exceeded by roundoff amplified by large spatial quantities
// on the inner sheet.
void test_consistency(const double mass, const double dimensionless_spin) {
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
  const double time = 1.234;
  const double delta = 1.e-4;
  const double tolerance = 1.e-8;

  const Approx adm_identity_approx = Approx::custom().epsilon(1.e-12).scale(1.);

  const auto check_point = [&](const tnsr::I<double, 3>& x) {
    CAPTURE(x);
    TestHelpers::VerifyGrSolution::verify_spatial_consistency(solution, time, x,
                                                              delta, tolerance);
    const auto vars = solution.variables(
        x, time, typename HighSpinKerrPuncture::template tags<double>{});
    const auto adm_extrinsic_curvature = gr::extrinsic_curvature(
        get<gr::Tags::Lapse<double>>(vars),
        get<gr::Tags::Shift<double, 3>>(vars),
        get<HighSpinKerrPuncture::DerivShift<double>>(vars),
        get<gr::Tags::SpatialMetric<double, 3>>(vars),
        get<::Tags::dt<gr::Tags::SpatialMetric<double, 3>>>(vars),
        get<HighSpinKerrPuncture::DerivSpatialMetric<double>>(vars));
    const auto& extrinsic_curvature =
        get<gr::Tags::ExtrinsicCurvature<double, 3>>(vars);
    CHECK_ITERABLE_CUSTOM_APPROX(extrinsic_curvature, adm_extrinsic_curvature,
                                 adm_identity_approx);
  };

  // A generic outer-sheet point far from throat/axis.
  check_point(tnsr::I<double, 3>{{{1.2 * throat, 2.3 * throat, 3.4 * throat}}});
  // Near-axis point (small varpi) on the outer sheet.
  check_point(
      tnsr::I<double, 3>{{{1.e-3 * throat, 2.e-3 * throat, 5. * throat}}});
  // Near-throat on the outer sheet.
  check_point(
      tnsr::I<double, 3>{{{0.45 * throat, 0.6 * throat, 1.2990381 * throat}}});
  // Generic inner-sheet point.
  check_point(tnsr::I<double, 3>{{{0.3 * throat, 0.2 * throat, 0.5 * throat}}});
  // Near-axis inner-sheet point.
  check_point(
      tnsr::I<double, 3>{{{1.e-3 * throat, 2.e-3 * throat, 0.5 * throat}}});
  // Exact on-axis points (x = y = 0).
  check_point(tnsr::I<double, 3>{{{0., 0., 3. * throat}}});
  check_point(tnsr::I<double, 3>{{{0., 0., -0.5 * throat}}});
}

// Item 3: verify_time_independent_einstein_solution evaluates the full
// generalized-harmonic RHS on a spectral grid and checks that nothing evolves
// and the gauge constraint holds. Outer sheet only: gh::TimeDerivative
// (Evolution/Systems/GeneralizedHarmonic/TimeDerivative.tpp) derives the
// lapse from the spacetime metric via gr::lapse
// (PointwiseFunctions/GeneralRelativity/Lapse.hpp), which can only return
// the positive root sqrt(alpha^2), while Pi was built from the signed lapse
// by gh::pi in gh::Solutions::WrappedGr. On the inner sheet r < r_+/4 the
// two lapses differ in sign, so the check misinterprets the data there; a
// Pi built from |alpha| instead would be genuinely non-stationary (pure
// gauge motion inside the horizon).
void test_einstein_solution() {
  const auto check_box = [](const double mass,
                            const double dimensionless_spin) {
    const HighSpinKerrPuncture solution(mass, dimensionless_spin);
    const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
    const size_t grid_size = 8;
    const std::array<double, 3> lower_bound{
        {throat * 2.8, throat * 4.3, throat * 4.6}};
    const std::array<double, 3> upper_bound{
        {lower_bound[0] + 0.02, lower_bound[1] + 0.02, lower_bound[2] + 0.02}};
    // Measured floor: epsilon()*1e4 fails (spectral-derivative roundoff);
    // epsilon()*1e5 gives ~10x headroom.
    TestHelpers::VerifyGrSolution::verify_time_independent_einstein_solution(
        solution, grid_size, lower_bound, upper_bound,
        std::numeric_limits<double>::epsilon() * 1e5);
  };
  check_box(1., 0.99);
  check_box(1.2, 0.6);
}

// Item 4: exact Schwarzschild limit chi = 0, all twelve tags. In isotropic
// coordinates the puncture radial coordinate r coincides with the isotropic
// radius, so with psi = 1 + M/2r and n_k = x_k/r:
//   gamma_ij = psi^4 delta_ij, alpha = (1 - M/2r)/psi, K_ij = 0, beta^i = 0,
//   d_k alpha = M/(r psi)^2 n_k, d_k gamma_ij = -(2M/r^2) psi^3 n_k delta_ij,
//   d_k beta^j = 0, sqrt(det gamma) = psi^6, gamma^ij = psi^-4 delta^ij,
//   and all time-derivative tags vanish.
// The lapse is signed (negative for r < M/2), matching the isotropic-
// coordinates lapse of the Schwarzschild solution.
template <typename DataType>
void test_schwarzschild_limit() {
  const double mass = 1.7;
  const HighSpinKerrPuncture solution(mass, 0.);
  const double throat = 0.25 * r_plus_of(mass, 0.);  // = M/2

  // Points on both sheets, off-axis and off-equator; the DataVector case
  // additionally samples the exact z-axis.
  DataType r_values{};
  if constexpr (std::is_same_v<DataType, double>) {
    r_values = 2.5 * throat;  // outer sheet
  } else {
    r_values = DataType{{0.4 * throat, 0.7 * throat, 1.5 * throat, 4. * throat,
                         0.9 * throat, 3. * throat}};
  }
  auto x = make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(r_values, 0.);
  // Fixed generic direction (0.3, 0.4, 0.8660254), a unit vector.
  get<0>(x) = 0.3 * r_values;
  get<1>(x) = 0.4 * r_values;
  get<2>(x) = 0.8660254037844386 * r_values;
  if constexpr (not std::is_same_v<DataType, double>) {
    // Exact z-axis points (both signs, one per sheet). Analytically the
    // chi = 0 solution has no preferred axis, but an implementation with
    // axis coordinate singularities (e.g. explicit 1/sin(theta) factors)
    // would fail here regardless of spin.
    get<0>(x)[4] = 0.;
    get<1>(x)[4] = 0.;
    get<2>(x)[4] = 0.9 * throat;
    get<0>(x)[5] = 0.;
    get<1>(x)[5] = 0.;
    get<2>(x)[5] = -3. * throat;
  }

  const double t = std::numeric_limits<double>::signaling_NaN();
  const auto vars = solution.variables(
      x, t, typename HighSpinKerrPuncture::template tags<DataType>{});

  const DataType r =
      sqrt(square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x)));
  const DataType conformal_factor = 1. + 0.5 * mass / r;
  const DataType expected_lapse = (1. - 0.5 * mass / r) / conformal_factor;
  const DataType expected_gamma_diag = pow<4>(conformal_factor);

  CHECK_ITERABLE_APPROX(get(get<gr::Tags::Lapse<DataType>>(vars)),
                        expected_lapse);
  const auto& spatial_metric = get<gr::Tags::SpatialMetric<DataType, 3>>(vars);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      if (i == j) {
        CHECK_ITERABLE_APPROX(spatial_metric.get(i, j), expected_gamma_diag);
      } else {
        CHECK_ITERABLE_APPROX(spatial_metric.get(i, j),
                              make_with_value<DataType>(r_values, 0.));
      }
    }
  }
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataType, 3>>(vars);
  const auto zero_ii =
      make_with_value<tnsr::ii<DataType, 3, Frame::Inertial>>(r_values, 0.);
  CHECK_ITERABLE_APPROX(extrinsic_curvature, zero_ii);
  const auto& shift = get<gr::Tags::Shift<DataType, 3>>(vars);
  const auto zero_I =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(r_values, 0.);
  CHECK_ITERABLE_APPROX(shift, zero_I);

  // All time-derivative tags vanish.
  const auto zero_scalar = make_with_value<Scalar<DataType>>(r_values, 0.);
  const auto& dt_lapse = get<::Tags::dt<gr::Tags::Lapse<DataType>>>(vars);
  CHECK_ITERABLE_APPROX(dt_lapse, zero_scalar);
  const auto& dt_shift = get<::Tags::dt<gr::Tags::Shift<DataType, 3>>>(vars);
  CHECK_ITERABLE_APPROX(dt_shift, zero_I);
  const auto& dt_spatial_metric =
      get<::Tags::dt<gr::Tags::SpatialMetric<DataType, 3>>>(vars);
  CHECK_ITERABLE_APPROX(dt_spatial_metric, zero_ii);

  // deriv(Shift) = 0 at chi = 0.
  const auto& d_shift = get<HighSpinKerrPuncture::DerivShift<DataType>>(vars);
  const auto zero_iJ =
      make_with_value<tnsr::iJ<DataType, 3, Frame::Inertial>>(r_values, 0.);
  CHECK_ITERABLE_APPROX(d_shift, zero_iJ);

  // deriv(Lapse) = M/(r psi)^2 n_k.
  const auto& d_lapse = get<HighSpinKerrPuncture::DerivLapse<DataType>>(vars);
  const DataType d_alpha_d_r = mass / square(r) / square(conformal_factor);
  for (size_t k = 0; k < 3; ++k) {
    const DataType expected_d_lapse = d_alpha_d_r * x.get(k) / r;
    CHECK_ITERABLE_APPROX(d_lapse.get(k), expected_d_lapse);
  }

  // deriv(SpatialMetric): d_k gamma_ij = -(2M/r^2) psi^3 n_k delta_ij.
  const auto& d_spatial_metric =
      get<HighSpinKerrPuncture::DerivSpatialMetric<DataType>>(vars);
  const DataType d_gamma_radial =
      -2. * mass * pow<3>(conformal_factor) / square(r);
  const auto zero = make_with_value<DataType>(r_values, 0.);
  for (size_t k = 0; k < 3; ++k) {
    const DataType n_k = x.get(k) / r;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        if (i == j) {
          const DataType expected_d_gamma = d_gamma_radial * n_k;
          CHECK_ITERABLE_APPROX(d_spatial_metric.get(k, i, j),
                                expected_d_gamma);
        } else {
          CHECK_ITERABLE_APPROX(d_spatial_metric.get(k, i, j), zero);
        }
      }
    }
  }

  // sqrt(det gamma) = psi^6.
  const auto& sqrt_det = get<gr::Tags::SqrtDetSpatialMetric<DataType>>(vars);
  const DataType expected_sqrt_det = pow<6>(conformal_factor);
  CHECK_ITERABLE_APPROX(get(sqrt_det), expected_sqrt_det);

  // gamma^ij = psi^-4 delta^ij.
  const auto& inverse_metric =
      get<gr::Tags::InverseSpatialMetric<DataType, 3>>(vars);
  const DataType expected_inverse_diag = 1. / pow<4>(conformal_factor);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      if (i == j) {
        CHECK_ITERABLE_APPROX(inverse_metric.get(i, j), expected_inverse_diag);
      } else {
        CHECK_ITERABLE_APPROX(inverse_metric.get(i, j), zero);
      }
    }
  }
}

// Item 5: identity tr K = gamma^{ij} K_ij = 0 to roundoff, on both sheets,
// using the class's own InverseSpatialMetric and ExtrinsicCurvature tags.
void test_trace_k_vanishes(const double mass, const double dimensionless_spin) {
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Points spanning both sheets, off-axis and off-equator.
  tnsr::I<DataVector, 3, Frame::Inertial> x(size_t{4});
  const std::array<double, 4> radii{
      {0.5 * throat, 0.8 * throat, 1.5 * throat, 5. * throat}};
  for (size_t p = 0; p < 4; ++p) {
    const double radius = gsl::at(radii, p);
    get<0>(x)[p] = 0.3 * radius;
    get<1>(x)[p] = 0.4 * radius;
    get<2>(x)[p] = 0.8660254037844386 * radius;
  }
  const auto vars = solution.variables(
      x, t,
      tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::ExtrinsicCurvature<DataVector, 3>>{});
  const auto& inverse_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(vars);
  const auto& extrinsic_curvature =
      get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(vars);
  DataVector trace_k(size_t{4}, 0.);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      trace_k += inverse_metric.get(i, j) * extrinsic_curvature.get(i, j);
    }
  }

  const Approx trace_approx = Approx::custom().epsilon(1.e-14).scale(1.);
  const DataVector expected_trace(size_t{4}, 0.);
  CHECK_ITERABLE_CUSTOM_APPROX(trace_k, expected_trace, trace_approx);
}

// Item 6: axis regularity. On the z-axis (x = y = 0, both signs of z, both
// sheets) all tags are finite, K_ij = 0, and gamma is diagonal with
// gamma_xx = gamma_yy.
void test_axis_regularity(const double mass, const double dimensionless_spin) {
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Points on the +z and -z axis, on both sheets.
  const std::array<double, 4> z_values{
      {0.6 * throat, -0.6 * throat, 3. * throat, -3. * throat}};
  for (const double z : z_values) {
    CAPTURE(z);
    const tnsr::I<double, 3> x_axis{{{0., 0., z}}};
    const auto vars = solution.variables(
        x_axis, t, typename HighSpinKerrPuncture::template tags<double>{});
    // Finiteness of every returned component.
    tmpl::for_each<typename HighSpinKerrPuncture::template tags<double>>(
        [&]<typename Tag>(tmpl::type_<Tag> /*meta*/) {
          const std::string tag_name = db::tag_name<Tag>();
          CAPTURE(tag_name);
          const auto& tensor = get<Tag>(vars);
          for (auto it = tensor.begin(); it != tensor.end(); ++it) {
            CHECK(std::isfinite(*it));
          }
        });
    // On the axis the extrinsic curvature must vanish.
    const auto& extrinsic_curvature =
        get<gr::Tags::ExtrinsicCurvature<double, 3>>(vars);
    const auto zero_ii = make_with_value<tnsr::ii<double, 3>>(0., 0.);
    CHECK_ITERABLE_APPROX(extrinsic_curvature, zero_ii);
    // The shift is azimuthal (lambda^i = (-y, x, 0)) so it vanishes on axis.
    const auto& shift = get<gr::Tags::Shift<double, 3>>(vars);
    const auto zero_I = make_with_value<tnsr::I<double, 3>>(0., 0.);
    CHECK_ITERABLE_APPROX(shift, zero_I);
    // gamma is diagonal with gamma_xx = gamma_yy on axis.
    const auto& spatial_metric = get<gr::Tags::SpatialMetric<double, 3>>(vars);
    CHECK(get<0, 1>(spatial_metric) == approx(0.));
    CHECK(get<0, 2>(spatial_metric) == approx(0.));
    CHECK(get<1, 2>(spatial_metric) == approx(0.));
    CHECK(get<0, 0>(spatial_metric) == approx(get<1, 1>(spatial_metric)));
  }
}

// Item 7: symmetry point-checks. (a) axisymmetry under rotations R about z:
// alpha(Rx) = alpha(x), beta(Rx) = R beta(x), gamma(Rx) = R gamma(x) R^T,
// K(Rx) = R K(x) R^T. (b) equatorial reflection P = diag(1, 1, -1) analogously.
void test_symmetries(const double mass, const double dimensionless_spin) {
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
  const double t = std::numeric_limits<double>::signaling_NaN();

  const Approx symmetry_approx = Approx::custom().epsilon(1.e-13).scale(1.);

  // Base points on both sheets, generic direction.
  const std::array<double, 2> radii{{0.6 * throat, 4. * throat}};

  const auto apply_matrix = [](const std::array<std::array<double, 3>, 3>& mat,
                               const std::array<double, 3>& vec) {
    std::array<double, 3> result{{0., 0., 0.}};
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        gsl::at(result, i) += gsl::at(gsl::at(mat, i), j) * gsl::at(vec, j);
      }
    }
    return result;
  };

  const auto check_covariance = [&](const std::array<std::array<double, 3>, 3>&
                                        mat,
                                    const std::array<double, 3>& base_point) {
    const tnsr::I<double, 3> x_base{
        {{base_point[0], base_point[1], base_point[2]}}};
    const auto mapped = apply_matrix(mat, base_point);
    const tnsr::I<double, 3> x_mapped{{{mapped[0], mapped[1], mapped[2]}}};

    const auto vars_base = solution.variables(
        x_base, t, typename HighSpinKerrPuncture::template tags<double>{});
    const auto vars_mapped = solution.variables(
        x_mapped, t, typename HighSpinKerrPuncture::template tags<double>{});

    // Scalar lapse is invariant.
    CHECK(get(get<gr::Tags::Lapse<double>>(vars_mapped)) ==
          symmetry_approx(get(get<gr::Tags::Lapse<double>>(vars_base))));

    // Shift transforms as a vector: beta(Rx) = R beta(x).
    const auto& shift_base = get<gr::Tags::Shift<double, 3>>(vars_base);
    const std::array<double, 3> shift_base_array{
        {get<0>(shift_base), get<1>(shift_base), get<2>(shift_base)}};
    const auto expected_shift = apply_matrix(mat, shift_base_array);
    const auto& shift_mapped = get<gr::Tags::Shift<double, 3>>(vars_mapped);
    for (size_t i = 0; i < 3; ++i) {
      CHECK(shift_mapped.get(i) == symmetry_approx(gsl::at(expected_shift, i)));
    }

    // gamma and K transform as rank-2 tensors: T(Rx) = R T(x) R^T.
    const auto check_rank_two = [&](const auto& tensor_base,
                                    const auto& tensor_mapped) {
      for (size_t i = 0; i < 3; ++i) {
        for (size_t j = 0; j < 3; ++j) {
          double expected = 0.;
          for (size_t k = 0; k < 3; ++k) {
            for (size_t l = 0; l < 3; ++l) {
              expected += gsl::at(gsl::at(mat, i), k) *
                          gsl::at(gsl::at(mat, j), l) * tensor_base.get(k, l);
            }
          }
          CHECK(tensor_mapped.get(i, j) == symmetry_approx(expected));
        }
      }
    };
    check_rank_two(get<gr::Tags::SpatialMetric<double, 3>>(vars_base),
                   get<gr::Tags::SpatialMetric<double, 3>>(vars_mapped));
    check_rank_two(get<gr::Tags::ExtrinsicCurvature<double, 3>>(vars_base),
                   get<gr::Tags::ExtrinsicCurvature<double, 3>>(vars_mapped));
  };

  for (const double radius : radii) {
    const std::array<double, 3> base_point{
        {0.3 * radius, 0.4 * radius, 0.8660254037844386 * radius}};

    // (a) Rotations about z by a few angles, including pi/2.
    for (const double angle : {0.37, M_PI_2, 2.1}) {
      const double c = std::cos(angle);
      const double s = std::sin(angle);
      const std::array<std::array<double, 3>, 3> rotation{
          {{{c, -s, 0.}}, {{s, c, 0.}}, {{0., 0., 1.}}}};
      check_covariance(rotation, base_point);
    }

    // (b) Equatorial reflection P = diag(1, 1, -1).
    const std::array<std::array<double, 3>, 3> reflection{
        {{{1., 0., 0.}}, {{0., 1., 0.}}, {{0., 0., -1.}}}};
    check_covariance(reflection, base_point);

    // On-axis base point: rotations about z fix it, so only the reflection
    // is checked; it relates the +z and -z axis points.
    check_covariance(reflection, std::array<double, 3>{{0., 0., radius}});
  }
}

// Item 8: closed-form InverseSpatialMetric and SqrtDetSpatialMetric versus the
// numerical determinant_and_inverse of the class's SpatialMetric, plus the
// identity gamma_ik gamma^kj = delta_i^j. Both sheets.
void test_inverse_and_determinant(const double mass,
                                  const double dimensionless_spin) {
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
  const double t = std::numeric_limits<double>::signaling_NaN();

  // Four generic-direction points spanning both sheets plus two exact
  // on-axis points (one per sheet), where the lambda block of the
  // closed-form inverse degenerates (varpi^2 = 0).
  constexpr size_t num_points = 6;
  tnsr::I<DataVector, 3, Frame::Inertial> x(num_points);
  const std::array<double, 4> radii{
      {0.4 * throat, 0.8 * throat, 1.5 * throat, 6. * throat}};
  for (size_t p = 0; p < 4; ++p) {
    const double radius = gsl::at(radii, p);
    get<0>(x)[p] = 0.3 * radius;
    get<1>(x)[p] = 0.4 * radius;
    get<2>(x)[p] = 0.8660254037844386 * radius;
  }
  get<0>(x)[4] = 0.;
  get<1>(x)[4] = 0.;
  get<2>(x)[4] = 0.7 * throat;
  get<0>(x)[5] = 0.;
  get<1>(x)[5] = 0.;
  get<2>(x)[5] = -2. * throat;
  const auto vars = solution.variables(
      x, t,
      tmpl::list<gr::Tags::SpatialMetric<DataVector, 3>,
                 gr::Tags::InverseSpatialMetric<DataVector, 3>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>{});
  const auto& spatial_metric =
      get<gr::Tags::SpatialMetric<DataVector, 3>>(vars);
  const auto& inverse_metric =
      get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(vars);
  const auto& sqrt_det = get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars);

  const auto numerical_det_and_inverse =
      determinant_and_inverse(spatial_metric);
  CHECK_ITERABLE_APPROX(get(sqrt_det),
                        sqrt(get(numerical_det_and_inverse.first)));
  CHECK_ITERABLE_APPROX(inverse_metric, numerical_det_and_inverse.second);

  // gamma_ik gamma^kj = delta_i^j.
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      DataVector product(num_points, 0.);
      for (size_t k = 0; k < 3; ++k) {
        product += spatial_metric.get(i, k) * inverse_metric.get(k, j);
      }
      const DataVector expected_identity(num_points, (i == j) ? 1. : 0.);
      CHECK_ITERABLE_APPROX(product, expected_identity);
    }
  }
}

// Item 9: test_tag_retrieval (single-tag retrieval consistent with bulk).
template <typename DataType>
void test_tag_retrieval(const DataType& used_for_size) {
  const double mass = 1.234;
  const double dimensionless_spin = 0.5;
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
  auto x =
      make_with_value<tnsr::I<DataType, 3, Frame::Inertial>>(used_for_size, 0.);
  // A generic outer-sheet point.
  get<0>(x) = 1.1 * throat;
  get<1>(x) = 2.2 * throat;
  get<2>(x) = 3.3 * throat;
  const double t = 1.3;
  TestHelpers::AnalyticSolutions::test_tag_retrieval(
      solution, x, t, typename HighSpinKerrPuncture::template tags<DataType>{});
}

// Item 10: serialization / creation / copy / move, mirroring the trumpet test.
void test_serialize() {
  const double mass = 1.5;
  const double dimensionless_spin = 0.8;
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  test_serialization(solution);
  // Re-run a pypp fixed-point check on the deserialized solution.
  const auto deserialized = serialize_and_deserialize(solution);
  const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
  const tnsr::I<double, 3> x{{{1.2 * throat, 2.3 * throat, 3.4 * throat}}};
  test_pypp<double>(deserialized, x, mass, dimensionless_spin);
}

void test_construct_from_options() {
  const double mass = 1.5;
  const double dimensionless_spin = 0.8;
  const HighSpinKerrPuncture solution(mass, dimensionless_spin);
  const auto created = TestHelpers::test_creation<HighSpinKerrPuncture>(
      "Mass: " + std::to_string(mass) +
      "\n"
      "DimensionlessSpin: " +
      std::to_string(dimensionless_spin));
  CHECK(created == solution);
  CHECK(solution != HighSpinKerrPuncture(mass, 0.5 * dimensionless_spin));
  CHECK(solution != HighSpinKerrPuncture(2. * mass, dimensionless_spin));
}

void test_copy_and_move() {
  HighSpinKerrPuncture solution(1.5, 0.8);
  const HighSpinKerrPuncture solution_copy(1.5, 0.8);
  test_copy_semantics(solution);
  // clang-tidy: std::move of trivially copyable type
  test_move_semantics(std::move(solution), solution_copy);  // NOLINT
}
}  // namespace

// [[TimeOut, 40]]
SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.Gr.HighSpinKerrPuncture",
    "[PointwiseFunctions][Unit]") {
  const pypp::SetupLocalPythonEnvironment local_python_env{
      "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/"};

  MAKE_GENERATOR(generator);

  // Item 1: pypp comparison of all twelve tags across radial shells on both
  // sheets, for chi in {0, 0.7, 0.99, -0.99} and mass != 1 for one case.
  test_pypp_all_shells(make_not_null(&generator), 1., 0.);
  test_pypp_all_shells(make_not_null(&generator), 1., 0.7);
  test_pypp_all_shells(make_not_null(&generator), 1., 0.99);
  test_pypp_all_shells(make_not_null(&generator), 1., -0.99);
  test_pypp_all_shells(make_not_null(&generator), 2.3, 0.7);
  // A fixed point exactly on the equator (z = 0), where the odd-in-z part of
  // the extrinsic curvature vanishes identically.
  {
    const double mass = 1.2;
    const double dimensionless_spin = 0.7;
    const HighSpinKerrPuncture solution(mass, dimensionless_spin);
    const double throat = 0.25 * r_plus_of(mass, dimensionless_spin);
    const tnsr::I<double, 3> x_equator{{{1.3 * throat, 1.7 * throat, 0.}}};
    test_pypp<double>(solution, x_equator, mass, dimensionless_spin);
  }

  // Item 2: verify_consistency, moderate and high spin.
  test_consistency(1.1, 0.5);
  test_consistency(1., 0.99);

  // Item 3: verify_time_independent_einstein_solution (outer sheet; see the
  // note above test_einstein_solution for why the inner sheet is excluded).
  test_einstein_solution();

  // Item 4: exact Schwarzschild limit chi = 0.
  test_schwarzschild_limit<double>();
  test_schwarzschild_limit<DataVector>();

  // Item 5: tr K = 0.
  test_trace_k_vanishes(1.3, 0.7);
  test_trace_k_vanishes(1., 0.99);

  // Item 6: axis regularity.
  test_axis_regularity(1.2, 0.6);
  test_axis_regularity(1., 0.99);

  // Item 7: symmetries (axisymmetry + equatorial reflection).
  test_symmetries(1.2, 0.6);
  test_symmetries(2.3, 0.7);

  // Item 8: inverse metric and determinant.
  test_inverse_and_determinant(1.2, 0.6);
  test_inverse_and_determinant(1., 0.99);

  // Item 9: tag retrieval.
  test_tag_retrieval<double>(0.);
  test_tag_retrieval<DataVector>(DataVector(size_t{5}));

  // Item 10: serialization / creation / copy / move.
  test_serialize();
  test_construct_from_options();
  test_copy_and_move();

  // Options / error tests.
  CHECK_THROWS_WITH([]() { const HighSpinKerrPuncture solution(0., 0.5); }(),
                    Catch::Matchers::ContainsSubstring(
                        "Black hole mass must be positive, but given "));
  CHECK_THROWS_WITH([]() { const HighSpinKerrPuncture solution(-1.5, 0.5); }(),
                    Catch::Matchers::ContainsSubstring(
                        "Black hole mass must be positive, but given "));
  CHECK_THROWS_WITH(
      []() { const HighSpinKerrPuncture solution(1., 1.); }(),
      Catch::Matchers::ContainsSubstring(
          "The dimensionless spin must satisfy |chi| < 1 strictly"));
  CHECK_THROWS_WITH(
      []() { const HighSpinKerrPuncture solution(1., -1.2); }(),
      Catch::Matchers::ContainsSubstring(
          "The dimensionless spin must satisfy |chi| < 1 strictly"));
  CHECK_THROWS_WITH(TestHelpers::test_creation<HighSpinKerrPuncture>(
                        "Mass: 0.\n"
                        "DimensionlessSpin: 0.5"),
                    Catch::Matchers::ContainsSubstring(
                        "Black hole mass must be positive, but given "));
  CHECK_THROWS_WITH(
      TestHelpers::test_creation<HighSpinKerrPuncture>(
          "Mass: 1.\n"
          "DimensionlessSpin: 1.5"),
      Catch::Matchers::ContainsSubstring(
          "The dimensionless spin must satisfy |chi| < 1 strictly"));
#ifdef SPECTRE_DEBUG
  // Evaluating at the puncture (the coordinate origin) trips the debug
  // ASSERT in IntermediateVars.
  CHECK_THROWS_WITH(([]() {
                      const HighSpinKerrPuncture solution(1., 0.5);
                      const tnsr::I<double, 3> x_origin{{{0., 0., 0.}}};
                      solution.variables(x_origin, 0.,
                                         tmpl::list<gr::Tags::Lapse<double>>{});
                    }()),
                    Catch::Matchers::ContainsSubstring(
                        "must not coincide with a grid point"));
#endif  // SPECTRE_DEBUG
}
