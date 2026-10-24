// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <complex>
#include <cstddef>
#include <string>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/ComplexModalVector.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCoefficients.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshCollocation.hpp"
#include "NumericalAlgorithms/SpinWeightedSphericalHarmonics/SwshTransform.hpp"
#include "NumericalAlgorithms/Strahlkorper/Strahlkorper.hpp"
#include "NumericalAlgorithms/Strahlkorper/StrahlkorperFunctions.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/TeukolskyWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/GeneralRelativity/Surfaces/ReggeWheelerZerilli.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {

size_t mode_index(const size_t l_max, const size_t l, const int m) {
  return Spectral::Swsh::goldberg_mode_index(l_max, l, m);
}

std::complex<double> mode(const ComplexModalVector& modes, const size_t l_max,
                          const size_t l, const int m) {
  return modes[mode_index(l_max, l, m)];
}

std::complex<double> strain_mode(const gr::surfaces::ReggeWheelerZerilli& rwz,
                                 const size_t l_max, const size_t l,
                                 const int m) {
  return mode(rwz.r_times_strain.data(), l_max, l, m);
}

void set_mode(const gsl::not_null<ComplexModalVector*> modes,
              const size_t l_max, const size_t l, const int m,
              const std::complex<double>& value) {
  (*modes)[mode_index(l_max, l, m)] = value;
}

struct MoncriefInputs {
  std::complex<double> h_t{};
  std::complex<double> dr_h_t{};
  std::complex<double> dt_h_r{};
  std::complex<double> h_rr{};
  std::complex<double> q_r{};
  std::complex<double> k{};
  std::complex<double> dr_k{};
  std::complex<double> g{};
  std::complex<double> dr_g{};
};

void test_moncrief_modes() {
  constexpr size_t l_max = 3;
  constexpr double radius = 10.0;
  const size_t number_of_modes = square(l_max + 1);
  ComplexModalVector h_t{number_of_modes, 0.0};
  ComplexModalVector dr_h_t{number_of_modes, 0.0};
  ComplexModalVector dt_h_r{number_of_modes, 0.0};
  ComplexModalVector h_rr{number_of_modes, 0.0};
  ComplexModalVector q_r{number_of_modes, 0.0};
  ComplexModalVector k{number_of_modes, 0.0};
  ComplexModalVector dr_k{number_of_modes, 0.0};
  ComplexModalVector g{number_of_modes, 0.0};
  ComplexModalVector dr_g{number_of_modes, 0.0};

  const MoncriefInputs input_22{{1.0, -0.5}, {-0.25, 0.75}, {0.8, 0.1},
                                {0.5, -0.3}, {-0.4, 0.2},   {0.3, 0.7},
                                {-0.1, 0.4}, {0.2, -0.6},   {0.05, 0.15}};
  const MoncriefInputs input_3m1{{-0.4, 0.9},   {0.3, -0.2}, {-0.8, 0.5},
                                 {1.1, 0.4},    {0.6, -0.7}, {-0.2, 0.1},
                                 {0.45, -0.35}, {0.15, 0.2}, {-0.05, 0.08}};

  const auto set_inputs = [&h_t, &dr_h_t, &dt_h_r, &h_rr, &q_r, &k, &dr_k, &g,
                           &dr_g](const size_t l, const int m,
                                  const MoncriefInputs& input) {
    set_mode(make_not_null(&h_t), l_max, l, m, input.h_t);
    set_mode(make_not_null(&dr_h_t), l_max, l, m, input.dr_h_t);
    set_mode(make_not_null(&dt_h_r), l_max, l, m, input.dt_h_r);
    set_mode(make_not_null(&h_rr), l_max, l, m, input.h_rr);
    set_mode(make_not_null(&q_r), l_max, l, m, input.q_r);
    set_mode(make_not_null(&k), l_max, l, m, input.k);
    set_mode(make_not_null(&dr_k), l_max, l, m, input.dr_k);
    set_mode(make_not_null(&g), l_max, l, m, input.g);
    set_mode(make_not_null(&dr_g), l_max, l, m, input.dr_g);
  };
  set_inputs(2, 2, input_22);
  // This independently supplied negative-m mode verifies that the modal API
  // consumes the full Goldberg data instead of reconstructing negative m.
  set_inputs(3, -1, input_3m1);

  // Modes below l=2 are outside the RWZ radiative sector.
  set_mode(make_not_null(&h_t), l_max, 1, 1, {10.0, 20.0});
  set_mode(make_not_null(&h_rr), l_max, 1, -1, {-30.0, 40.0});

  const auto rwz = gr::surfaces::regge_wheeler_zerilli_moncrief(
      h_t, dr_h_t, dt_h_r, h_rr, q_r, k, dr_k, g, dr_g, l_max, radius);
  static_assert(decltype(rwz.r_times_strain)::spin == -2);

  // Reference values obtained by substituting the inputs above into the
  // Moncrief definitions documented in ReggeWheelerZerilli.hpp. Keeping the
  // values here avoids reproducing the implementation formulas in the test.
  const std::complex<double> expected_phi_plus_22{2.95, -331.0 / 60.0};
  const std::complex<double> expected_phi_minus_22{3.125, -1.875};
  const std::complex<double> expected_phi_plus_3m1{-31.0 / 300.0,
                                                   281.0 / 150.0};
  const std::complex<double> expected_phi_minus_3m1{-1.18, 0.88};
  const std::complex<double> imaginary_unit{0.0, 1.0};
  ComplexModalVector expected_phi_plus{number_of_modes, 0.0};
  ComplexModalVector expected_phi_minus{number_of_modes, 0.0};
  ComplexModalVector expected_r_times_strain{number_of_modes, 0.0};
  set_mode(make_not_null(&expected_phi_plus), l_max, 2, 2,
           expected_phi_plus_22);
  set_mode(make_not_null(&expected_phi_minus), l_max, 2, 2,
           expected_phi_minus_22);
  set_mode(make_not_null(&expected_r_times_strain), l_max, 2, 2,
           sqrt(24.0) *
               (expected_phi_plus_22 + imaginary_unit * expected_phi_minus_22));
  set_mode(make_not_null(&expected_phi_plus), l_max, 3, -1,
           expected_phi_plus_3m1);
  set_mode(make_not_null(&expected_phi_minus), l_max, 3, -1,
           expected_phi_minus_3m1);
  set_mode(make_not_null(&expected_r_times_strain), l_max, 3, -1,
           sqrt(120.0) * (expected_phi_plus_3m1 +
                          imaginary_unit * expected_phi_minus_3m1));
  CHECK_ITERABLE_APPROX(rwz.phi_plus, expected_phi_plus);
  CHECK_ITERABLE_APPROX(rwz.phi_minus, expected_phi_minus);
  CHECK_ITERABLE_APPROX(rwz.r_times_strain.data(), expected_r_times_strain);

  gr::surfaces::ReggeWheelerZerilli output_argument_result{};
  gr::surfaces::regge_wheeler_zerilli_moncrief(
      make_not_null(&output_argument_result), h_t, dr_h_t, dt_h_r, h_rr, q_r, k,
      dr_k, g, dr_g, l_max, radius);
  CHECK_ITERABLE_APPROX(output_argument_result.phi_plus, rwz.phi_plus);
  CHECK_ITERABLE_APPROX(output_argument_result.phi_minus, rwz.phi_minus);
  CHECK_ITERABLE_APPROX(output_argument_result.r_times_strain.data(),
                        rwz.r_times_strain.data());
}

struct GhSphereData {
  tnsr::I<DataVector, 3, Frame::Inertial> coords{};
  tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric{};
  tnsr::aa<DataVector, 3, Frame::Inertial> pi{};
  tnsr::iaa<DataVector, 3, Frame::Inertial> phi{};
};

GhSphereData minkowski_data(
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper) {
  GhSphereData result{};
  result.coords = ylm::cartesian_coords(strahlkorper);
  const size_t size = get<0>(result.coords).size();
  result.spacetime_metric = tnsr::aa<DataVector, 3, Frame::Inertial>{size, 0.0};
  result.pi = tnsr::aa<DataVector, 3, Frame::Inertial>{size, 0.0};
  result.phi = tnsr::iaa<DataVector, 3, Frame::Inertial>{size, 0.0};
  get<0, 0>(result.spacetime_metric) = -1.0;
  for (size_t i = 0; i < 3; ++i) {
    result.spacetime_metric.get(i + 1, i + 1) = 1.0;
  }
  return result;
}

void check_all_modes_zero(const gr::surfaces::ReggeWheelerZerilli& rwz,
                          const double tolerance) {
  const ComplexModalVector zeros{rwz.phi_plus.size(), 0.0};
  const Approx zero_approx = Approx::custom().epsilon(0.0).margin(tolerance);
  CHECK_ITERABLE_CUSTOM_APPROX(rwz.phi_plus, zeros, zero_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(rwz.phi_minus, zeros, zero_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(rwz.r_times_strain.data(), zeros, zero_approx);
}

void test_minkowski() {
  constexpr size_t l_max = 5;
  constexpr size_t angular_l_max = l_max + 2;
  constexpr double radius = 3.0;
  const std::array<double, 3> center{{0.1, -0.2, 0.3}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      angular_l_max, angular_l_max, radius, center};
  const auto data = minkowski_data(strahlkorper);
  const auto rwz = gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
      data.spacetime_metric, data.pi, data.phi, data.coords,
      strahlkorper.ylm_spherepack(), l_max, center, radius);
  check_all_modes_zero(rwz, 1.0e-14);
}

struct ExpectedSpecMode {
  size_t l;
  int m;
  std::complex<double> r_times_strain;
};

// Generated with SpEC 85326db0d633ab6436b621d09e90d072373fb1b7 by
// applying ComputeReggeWheelerZerilliFlat to the analytic data in
// spec_regression_data(). Increasing the SpEC surface resolution from 12x24
// to 16x32 changed r*h by at most 1.6e-16.
constexpr std::array<ExpectedSpecMode, 21> expected_spec_modes{{
    {2, -2, {-1.60948834257061972e-3, -7.57939163199307050e-4}},
    {2, -1, {1.67894093521450292e-3, 8.21352399961057132e-4}},
    {2, 0, {-1.07991384088803721e-3, 1.05624449643029975e-2}},
    {2, 1, {1.93259388226050280e-4, 7.24722705848362958e-5}},
    {2, 2, {2.09263681313602004e-3, -2.41272267488550161e-3}},
    {3, -3, {-3.17254857543564691e-4, -5.19276916255269946e-4}},
    {3, -2, {1.08617106159982087e-3, 2.83452535812188681e-4}},
    {3, -1, {-1.93770994694657868e-4, -1.50648003650769217e-6}},
    {3, 0, {-4.18792644190715119e-4, -9.39347986969824787e-5}},
    {3, 1, {-8.28752330079265552e-4, 2.56101606204803701e-5}},
    {3, 2, {6.28835877768316154e-4, -1.11951841875375374e-4}},
    {3, 3, {-1.09398226739523185e-5, -2.74224888359522550e-4}},
    {4, -4, {-9.28962092157553291e-5, 4.76390816491117177e-6}},
    {4, -3, {6.90561312513063432e-5, 3.36859176833184848e-6}},
    {4, -2, {5.58182292094264632e-5, -5.40176411700038356e-6}},
    {4, -1, {-2.99203882940425269e-5, 6.49336086380552212e-5}},
    {4, 0, {7.23132835029881857e-5, 3.98577153166085395e-5}},
    {4, 1, {1.97347241939539583e-5, 1.90981201876619865e-5}},
    {4, 2, {-3.78123488193597389e-5, 5.58182292094248233e-5}},
    {4, 3, {-4.21073971044347184e-5, -3.70545094519757307e-5}},
    {4, 4, {7.86044847210584251e-5, -7.14586224736726697e-5}},
}};

GhSphereData spec_regression_data(
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper) {
  constexpr double epsilon = 1.0e-3;
  auto result = minkowski_data(strahlkorper);
  for (size_t s = 0; s < get<0>(result.coords).size(); ++s) {
    const double x = get<0>(result.coords)[s] / 8.0;
    const double y = get<1>(result.coords)[s] / 8.0;
    const double z = get<2>(result.coords)[s] / 8.0;
    const std::array<double, 3> radial{{x, y, z}};

    const double t11 =
        epsilon * (0.70 * x + 0.20 * y * z + 0.10 * (x * x - y * y));
    const double t22 =
        epsilon * (-0.40 * y + 0.30 * x * z - 0.12 * (3.0 * z * z - 1.0));
    const double t33 = epsilon * (0.25 * z + 0.18 * x * y);
    const double t12 = epsilon * (0.31 * x * y + 0.17 * z + 0.07 * x * x * z);
    const double t13 = epsilon * (-0.29 * y * z + 0.16 * x + 0.05 * y * y * x);
    const double t23 = epsilon * (0.37 * x * z - 0.14 * y + 0.06 * z * z * y);
    get<1, 1>(result.spacetime_metric)[s] += t11;
    get<2, 2>(result.spacetime_metric)[s] += t22;
    get<3, 3>(result.spacetime_metric)[s] += t33;
    get<1, 2>(result.spacetime_metric)[s] = t12;
    get<1, 3>(result.spacetime_metric)[s] = t13;
    get<2, 3>(result.spacetime_metric)[s] = t23;

    get<1, 0>(result.spacetime_metric)[s] =
        epsilon * (0.21 * y + 0.13 * x * z + 0.09 * (y * y - z * z));
    get<2, 0>(result.spacetime_metric)[s] =
        epsilon * (-0.27 * x + 0.19 * y * z + 0.11 * x * y * z);
    get<3, 0>(result.spacetime_metric)[s] =
        epsilon * (0.23 * x * y - 0.15 * z + 0.08 * x * x * y);

    const std::array<std::array<double, 3>, 3> dt_spatial_metric{{
        {{epsilon * (0.12 * x * y - 0.20 * z + 0.03 * y * z * z),
          epsilon * (-0.18 * x + 0.09 * y * z + 0.04 * x * x * y),
          epsilon * (0.22 * y + 0.08 * x * z - 0.06 * x * y * y)}},
        {{0.0, epsilon * (0.15 * x - 0.11 * y * y + 0.07 * x * y * z),
          epsilon * (-0.24 * z + 0.05 * x * y + 0.02 * y * z * z)}},
        {{0.0, 0.0,
          epsilon * (-0.10 * y + 0.17 * x * z - 0.08 * (x * x - z * z))}},
    }};
    const std::array<std::array<double, 3>, 3> dr_spatial_metric{{
        {{epsilon * (-0.16 * y + 0.09 * x * z + 0.05 * x * x * y),
          epsilon * (0.13 * z + 0.06 * x * y - 0.04 * y * y * z),
          epsilon * (0.19 * x - 0.12 * y * z + 0.03 * x * z * z)}},
        {{0.0, epsilon * (0.14 * z - 0.08 * x * y + 0.02 * x * x * z),
          epsilon * (-0.17 * y + 0.10 * x * z + 0.05 * x * x * y)}},
        {{0.0, 0.0,
          epsilon * (0.11 * x + 0.07 * y * z - 0.09 * (y * y - z * z))}},
    }};
    const std::array<double, 3> dr_metric_time_space{{
        epsilon * (0.08 * x - 0.14 * y * z + 0.06 * x * x * y),
        epsilon * (-0.12 * y + 0.16 * x * z - 0.04 * y * y * z),
        epsilon * (0.18 * z + 0.05 * x * y - 0.07 * x * z * z),
    }};

    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = i; j < 3; ++j) {
        result.pi.get(i + 1, j + 1)[s] =
            -gsl::at(gsl::at(dt_spatial_metric, i), j);
        for (size_t k = 0; k < 3; ++k) {
          result.phi.get(k, i + 1, j + 1)[s] =
              gsl::at(radial, k) * gsl::at(gsl::at(dr_spatial_metric, i), j);
        }
      }
      for (size_t k = 0; k < 3; ++k) {
        result.phi.get(k, i + 1, 0)[s] =
            gsl::at(radial, k) * gsl::at(dr_metric_time_space, i);
      }
    }
  }
  return result;
}

void test_against_spec() {
  constexpr size_t l_max = 4;
  constexpr size_t angular_l_max = l_max + 2;
  constexpr double radius = 8.0;
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      angular_l_max, angular_l_max, radius, center};
  const auto data = spec_regression_data(strahlkorper);
  const auto rwz = gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
      data.spacetime_metric, data.pi, data.phi, data.coords,
      strahlkorper.ylm_spherepack(), l_max, center, radius);

  ComplexModalVector expected_r_times_strain{square(l_max + 1), 0.0};
  for (const auto& expected : expected_spec_modes) {
    set_mode(make_not_null(&expected_r_times_strain), l_max, expected.l,
             expected.m, expected.r_times_strain);
  }
  const Approx spec_approx = Approx::custom().epsilon(0.0).margin(5.0e-13);
  CHECK_ITERABLE_CUSTOM_APPROX(rwz.r_times_strain.data(),
                               expected_r_times_strain, spec_approx);
}

using SpatialMetricTag =
    gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>;
using DtSpatialMetricTag = ::Tags::dt<SpatialMetricTag>;

tnsr::I<DataVector, 3, Frame::Inertial> radially_shifted_coords(
    const tnsr::I<DataVector, 3, Frame::Inertial>& coords,
    const std::array<double, 3>& center, const double radius,
    const double radial_offset) {
  auto result = coords;
  const double scale = 1.0 + radial_offset / radius;
  for (size_t i = 0; i < 3; ++i) {
    result.get(i) =
        gsl::at(center, i) + scale * (coords.get(i) - gsl::at(center, i));
  }
  return result;
}

GhSphereData teukolsky_gh_data(
    const gr::Solutions::TeukolskyWave& solution,
    const ylm::Strahlkorper<Frame::Inertial>& strahlkorper, const double time) {
  const auto& center = solution.center();
  const double radius = strahlkorper.average_radius();
  GhSphereData result{};
  result.coords = ylm::cartesian_coords(strahlkorper);
  const size_t size = get<0>(result.coords).size();
  result.spacetime_metric = tnsr::aa<DataVector, 3, Frame::Inertial>{size, 0.0};
  result.pi = tnsr::aa<DataVector, 3, Frame::Inertial>{size, 0.0};
  result.phi = tnsr::iaa<DataVector, 3, Frame::Inertial>{size, 0.0};

  const auto vars = solution.variables(
      result.coords, time, tmpl::list<SpatialMetricTag, DtSpatialMetricTag>{});
  const auto& spatial_metric_perturbation = get<SpatialMetricTag>(vars);
  const auto& dt_spatial_metric = get<DtSpatialMetricTag>(vars);
  get<0, 0>(result.spacetime_metric) = -1.0;
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      result.spacetime_metric.get(i + 1, j + 1) =
          spatial_metric_perturbation.get(i, j);
      if (i == j) {
        result.spacetime_metric.get(i + 1, j + 1) += 1.0;
      }
      result.pi.get(i + 1, j + 1) = -dt_spatial_metric.get(i, j);
    }
  }

  const double step = 1.0e-3 * radius;
  const auto metric_at_offset = [&solution, &result, &center, radius,
                                 time](const double offset) {
    const auto offset_coords =
        radially_shifted_coords(result.coords, center, radius, offset);
    return get<SpatialMetricTag>(
        solution.variable<SpatialMetricTag>(offset_coords, time));
  };
  const auto metric_m2 = metric_at_offset(-2.0 * step);
  const auto metric_m1 = metric_at_offset(-step);
  const auto metric_p1 = metric_at_offset(step);
  const auto metric_p2 = metric_at_offset(2.0 * step);
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = i; j < 3; ++j) {
      const DataVector dr_metric =
          (metric_m2.get(i, j) - 8.0 * metric_m1.get(i, j) +
           8.0 * metric_p1.get(i, j) - metric_p2.get(i, j)) /
          (12.0 * step);
      for (size_t k = 0; k < 3; ++k) {
        result.phi.get(k, i + 1, j + 1) =
            (result.coords.get(k) - gsl::at(center, k)) / radius * dr_metric;
      }
    }
  }
  return result;
}

gr::surfaces::ReggeWheelerZerilli extract_teukolsky_wave(
    const gr::Solutions::TeukolskyWave& solution, const size_t l_max,
    const double radius, const double time) {
  const size_t angular_l_max = l_max + 2;
  const auto& center = solution.center();
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      angular_l_max, angular_l_max, radius, center};
  const auto data = teukolsky_gh_data(solution, strahlkorper, time);
  return gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
      data.spacetime_metric, data.pi, data.phi, data.coords,
      strahlkorper.ylm_spherepack(), l_max, center, radius);
}

gr::surfaces::ReggeWheelerZerilli extract_teukolsky_wave(
    const double amplitude, const int input_m, const std::string& parity,
    const std::array<double, 3>& center) {
  constexpr size_t l_max = 6;
  constexpr double radius = 10.0;
  constexpr double time = 2.4;
  const gr::Solutions::TeukolskyWave solution{
      amplitude, input_m, parity, "outgoing", center, 8.0, 1.5, false};
  return extract_teukolsky_wave(solution, l_max, radius, time);
}

SpinWeighted<ComplexModalVector, -2> direct_teukolsky_strain_modes(
    const gr::Solutions::TeukolskyWave& solution, const size_t l_max,
    const double radius, const double time) {
  tnsr::i<DataVector, 3> unit_cartesian_coords{};
  tnsr::i<DataVector, 2, Frame::Spherical<Frame::Inertial>> angular_coords{};
  Spectral::Swsh::create_angular_and_cartesian_coordinates(
      make_not_null(&unit_cartesian_coords), make_not_null(&angular_coords),
      l_max);

  const size_t number_of_points =
      Spectral::Swsh::number_of_swsh_collocation_points(l_max);
  tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords{number_of_points};
  for (size_t i = 0; i < 3; ++i) {
    inertial_coords.get(i) =
        gsl::at(solution.center(), i) + radius * unit_cartesian_coords.get(i);
  }
  const auto spatial_metric_vars =
      solution.variable<SpatialMetricTag>(inertial_coords, time);
  const auto& spatial_metric_perturbation =
      get<SpatialMetricTag>(spatial_metric_vars);

  // Project the analytic metric perturbation onto the standard orthonormal
  // polarization basis. This gives h_+ - i h_cross independently of the RWZ
  // tensor-harmonic decomposition and RWZ-function normalization.
  SpinWeighted<ComplexDataVector, -2> direct_strain{number_of_points, 0.0};
  for (size_t s = 0; s < number_of_points; ++s) {
    const double theta = get<0>(angular_coords)[s];
    const double phi = get<1>(angular_coords)[s];
    const std::array<double, 3> e_theta{
        {cos(theta) * cos(phi), cos(theta) * sin(phi), -sin(theta)}};
    const std::array<double, 3> e_phi{{-sin(phi), cos(phi), 0.0}};
    double h_theta_theta = 0.0;
    double h_theta_phi = 0.0;
    double h_phi_phi = 0.0;
    for (size_t i = 0; i < 3; ++i) {
      for (size_t j = 0; j < 3; ++j) {
        const double metric_component =
            spatial_metric_perturbation.get(i, j)[s];
        h_theta_theta +=
            metric_component * gsl::at(e_theta, i) * gsl::at(e_theta, j);
        h_theta_phi +=
            metric_component * gsl::at(e_theta, i) * gsl::at(e_phi, j);
        h_phi_phi += metric_component * gsl::at(e_phi, i) * gsl::at(e_phi, j);
      }
    }
    direct_strain.data()[s] = 0.5 * (h_theta_theta - h_phi_phi) -
                              std::complex<double>{0.0, h_theta_phi};
  }
  return Spectral::Swsh::libsharp_to_goldberg_modes(
      Spectral::Swsh::swsh_transform(l_max, 1, direct_strain), l_max);
}

void test_teukolsky_wave() {
  constexpr size_t l_max = 6;
  constexpr double amplitude = 1.0e-3;
  constexpr double radius = 10.0;
  constexpr double time = 2.4;
  constexpr double profile_radius = 8.0;
  constexpr double profile_width = 1.5;
  constexpr double tolerance = 2.0e-10;
  const Approx teukolsky_approx =
      Approx::custom().epsilon(0.0).margin(tolerance);
  const std::array<double, 3> center{{0.3, -0.4, 0.2}};
  const auto even = extract_teukolsky_wave(amplitude, 0, "even", center);
  const auto even_double =
      extract_teukolsky_wave(2.0 * amplitude, 0, "even", center);
  const auto odd = extract_teukolsky_wave(amplitude, 2, "odd", center);
  const auto odd_at_origin = extract_teukolsky_wave(
      amplitude, 2, "odd", std::array<double, 3>{{0.0, 0.0, 0.0}});

  // Evaluate the Gaussian profile and its derivatives. Substitution of the
  // Teukolsky metric (Eqs. (5)--(10) of Teukolsky 1982) into the documented
  // Moncrief definitions gives the two nonzero modes below. The square-root
  // factors convert the real Teukolsky angular functions to orthonormal Y_lm.
  const double profile_coordinate = radius - profile_radius - time;
  const double derivative_factor = -2.0 / square(profile_width);
  const double profile =
      amplitude * exp(-square(profile_coordinate) / square(profile_width));
  const double profile_1 = derivative_factor * profile_coordinate * profile;
  const double profile_2 =
      derivative_factor * (profile + profile_coordinate * profile_1);
  const double profile_3 =
      derivative_factor * (2.0 * profile_1 + profile_coordinate * profile_2);
  const double profile_4 =
      derivative_factor * (3.0 * profile_2 + profile_coordinate * profile_3);
  const std::complex<double> expected_phi_plus_20{
      0.5 * sqrt(M_PI / 5.0) *
          (profile_4 - 3.0 * profile_3 / radius +
           3.0 * profile_2 / square(radius)),
      0.0};
  const std::complex<double> expected_phi_minus_22{
      0.0, -sqrt(2.0 * M_PI / 15.0) * (profile_3 - 3.0 * profile_2 / radius +
                                       3.0 * profile_1 / square(radius))};
  const std::complex<double> imaginary_unit{0.0, 1.0};
  const size_t number_of_modes = square(l_max + 1);
  ComplexModalVector expected_even_phi_plus{number_of_modes, 0.0};
  ComplexModalVector expected_odd_phi_minus{number_of_modes, 0.0};
  ComplexModalVector expected_even_strain{number_of_modes, 0.0};
  ComplexModalVector expected_odd_strain{number_of_modes, 0.0};
  const ComplexModalVector zero_modes{number_of_modes, 0.0};
  set_mode(make_not_null(&expected_even_phi_plus), l_max, 2, 0,
           expected_phi_plus_20);
  set_mode(make_not_null(&expected_even_strain), l_max, 2, 0,
           sqrt(24.0) * expected_phi_plus_20);
  set_mode(make_not_null(&expected_odd_phi_minus), l_max, 2, 2,
           expected_phi_minus_22);
  set_mode(make_not_null(&expected_odd_phi_minus), l_max, 2, -2,
           conj(expected_phi_minus_22));
  set_mode(make_not_null(&expected_odd_strain), l_max, 2, 2,
           sqrt(24.0) * imaginary_unit * expected_phi_minus_22);
  set_mode(make_not_null(&expected_odd_strain), l_max, 2, -2,
           sqrt(24.0) * imaginary_unit * conj(expected_phi_minus_22));

  CHECK_ITERABLE_CUSTOM_APPROX(even.phi_plus, expected_even_phi_plus,
                               teukolsky_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(even.phi_minus, zero_modes, teukolsky_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(even.r_times_strain.data(), expected_even_strain,
                               teukolsky_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(odd.phi_plus, zero_modes, teukolsky_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(odd.phi_minus, expected_odd_phi_minus,
                               teukolsky_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(odd.r_times_strain.data(), expected_odd_strain,
                               teukolsky_approx);

  auto expected_even_double = expected_even_strain;
  expected_even_double *= 2.0;
  CHECK_ITERABLE_CUSTOM_APPROX(even_double.r_times_strain.data(),
                               expected_even_double, teukolsky_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(odd.r_times_strain.data(),
                               odd_at_origin.r_times_strain.data(),
                               teukolsky_approx);
}

void test_teukolsky_wave_strain_normalization() {
  constexpr size_t l_max = 6;
  constexpr double amplitude = 1.0e-3;
  constexpr double inner_radius = 40.0;
  constexpr double outer_radius = 80.0;
  // Keep the outgoing pulse at the same retarded time as the existing
  // Teukolsky test while moving the extraction sphere into the wave zone.
  constexpr double time_minus_radius = 2.4 - 10.0;
  const std::array<double, 3> center{{0.3, -0.4, 0.2}};

  const auto check_parity = [&center, inner_radius, outer_radius](
                                const int input_m, const std::string& parity) {
    const gr::Solutions::TeukolskyWave solution{
        amplitude, input_m, parity, "outgoing", center, 8.0, 1.5, false};
    const auto inner_rwz = extract_teukolsky_wave(
        solution, l_max, inner_radius, inner_radius + time_minus_radius);
    const auto outer_rwz = extract_teukolsky_wave(
        solution, l_max, outer_radius, outer_radius + time_minus_radius);
    const auto inner_direct = direct_teukolsky_strain_modes(
        solution, l_max, inner_radius, inner_radius + time_minus_radius);
    const auto outer_direct = direct_teukolsky_strain_modes(
        solution, l_max, outer_radius, outer_radius + time_minus_radius);

    const std::complex<double> inner_expected =
        inner_radius * mode(inner_direct.data(), l_max, 2, input_m);
    const std::complex<double> outer_expected =
        outer_radius * mode(outer_direct.data(), l_max, 2, input_m);
    const std::complex<double> inner_actual =
        strain_mode(inner_rwz, l_max, 2, input_m);
    const std::complex<double> outer_actual =
        strain_mode(outer_rwz, l_max, 2, input_m);
    const double inner_error = abs(inner_actual - inner_expected);
    const double outer_error = abs(outer_actual - outer_expected);
    CAPTURE(input_m);
    CAPTURE(parity);
    CAPTURE(inner_actual);
    CAPTURE(inner_expected);
    CAPTURE(outer_actual);
    CAPTURE(outer_expected);
    CAPTURE(inner_error);
    CAPTURE(outer_error);
    CHECK(abs(outer_expected) > 1.0e-6);
    CHECK(outer_error < 0.6 * inner_error);
    CHECK(outer_error < 0.015 * abs(outer_expected));
  };

  check_parity(0, "even");
  check_parity(0, "odd");
}

void test_spherical_schwarzschild_has_no_radiative_modes() {
  constexpr size_t l_max = 8;
  constexpr size_t angular_l_max = l_max + 2;
  constexpr double radius = 10.0;
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const std::array<double, 3> spin{{0.0, 0.0, 0.0}};
  const std::array<double, 3> velocity{{0.0, 0.0, 0.0}};
  const ylm::Strahlkorper<Frame::Inertial> strahlkorper{
      angular_l_max, angular_l_max, radius, center};
  const auto coords = ylm::cartesian_coords(strahlkorper);
  const gh::Solutions::WrappedGr<gr::Solutions::KerrSchild> solution{
      1.0, spin, center, velocity};
  const auto gh_vars = solution.variables(
      coords, 0.0,
      tmpl::list<gr::Tags::SpacetimeMetric<DataVector, 3>,
                 gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>>{});
  const auto rwz = gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
      get<gr::Tags::SpacetimeMetric<DataVector, 3>>(gh_vars),
      get<gh::Tags::Pi<DataVector, 3>>(gh_vars),
      get<gh::Tags::Phi<DataVector, 3>>(gh_vars), coords,
      strahlkorper.ylm_spherepack(), l_max, center, radius);
  check_all_modes_zero(rwz, 1.0e-11);
}

#ifdef SPECTRE_DEBUG
void test_input_validation() {
  constexpr size_t l_max = 3;
  const size_t number_of_modes = square(l_max + 1);
  const ComplexModalVector modes{number_of_modes, 0.0};
  const ComplexModalVector wrong_size_modes{number_of_modes - 1, 0.0};
  CHECK_THROWS_WITH(
      (gr::surfaces::regge_wheeler_zerilli_moncrief(
          wrong_size_modes, modes, modes, modes, modes, modes, modes, modes,
          modes, l_max, 2.0)),
      Catch::Matchers::ContainsSubstring("Expected h_t to have size"));
  CHECK_THROWS_WITH(
      (gr::surfaces::regge_wheeler_zerilli_moncrief(modes, modes, modes, modes,
                                                    modes, modes, modes, modes,
                                                    modes, l_max, 0.0)),
      Catch::Matchers::ContainsSubstring("extraction radius must be positive"));

  const ylm::Spherepack incomplete_spherepack{4, 3};
  const size_t size = incomplete_spherepack.physical_size();
  const tnsr::aa<DataVector, 3, Frame::Inertial> spacetime_metric{size, 0.0};
  const tnsr::aa<DataVector, 3, Frame::Inertial> pi{size, 0.0};
  const tnsr::iaa<DataVector, 3, Frame::Inertial> phi{size, 0.0};
  const tnsr::I<DataVector, 3, Frame::Inertial> coords{size, 0.0};
  CHECK_THROWS_WITH(
      (gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
          spacetime_metric, pi, phi, coords, incomplete_spherepack, 2,
          std::array<double, 3>{{0.0, 0.0, 0.0}}, 2.0)),
      Catch::Matchers::ContainsSubstring("requires m_max == l_max"));

  constexpr double radius = 2.0;
  const std::array<double, 3> center{{0.0, 0.0, 0.0}};
  const ylm::Strahlkorper<Frame::Inertial> underresolved_strahlkorper{
      4, 4, radius, center};
  const auto underresolved_data = minkowski_data(underresolved_strahlkorper);
  CHECK_THROWS_WITH(
      (gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
          underresolved_data.spacetime_metric, underresolved_data.pi,
          underresolved_data.phi, underresolved_data.coords,
          underresolved_strahlkorper.ylm_spherepack(), 3, center, radius)),
      Catch::Matchers::ContainsSubstring(
          "requires a TensorYlm grid with l_max"));

  const tnsr::aa<DataVector, 3, Frame::Inertial> wrong_size_spacetime_metric{
      get<0>(underresolved_data.coords).size() - 1, 0.0};
  CHECK_THROWS_WITH(
      (gr::surfaces::regge_wheeler_zerilli_moncrief_from_gh_vars(
          wrong_size_spacetime_metric, underresolved_data.pi,
          underresolved_data.phi, underresolved_data.coords,
          underresolved_strahlkorper.ylm_spherepack(), 2, center, radius)),
      Catch::Matchers::ContainsSubstring(
          "component 0 of spacetime_metric to have size"));
}
#endif

}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.GeneralRelativity.Surfaces."
    "ReggeWheelerZerilli",
    "[PointwiseFunctions][Unit]") {
  test_moncrief_modes();
  test_minkowski();
  test_against_spec();
  test_teukolsky_wave();
  test_teukolsky_wave_strain_normalization();
  test_spherical_schwarzschild_has_no_radiative_modes();
#ifdef SPECTRE_DEBUG
  test_input_validation();
#endif
}
