// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "PointwiseFunctions/Hydro/PrimitiveSources.hpp"
#include "Utilities/Gsl.hpp"

namespace {
namespace sources = hydro::primitive_sources;
using Vector = std::array<double, 3>;
using Metric = std::array<Vector, 3>;
const Metric identity{{{1., 0., 0.}, {0., 1., 0.}, {0., 0., 1.}}};
const Vector zero{{0., 0., 0.}};

// A parallel magnetic field has exactly zero Poynting momentum. This remains
// true for received binary64 speeds near 0.499, not only the exact-W case 0.5.
// References are rho*s/(1-3*s*s), evaluated independently from the received
// binary64 inputs in C++23 with 512-bit binary arithmetic and rounded once.
// Hexadecimal literals preserve those reference bits without instantiating
// Boost's float-conversion templates in the unit test's static analysis.
void test_parallel_field_cancellation() {
  const auto relative =
      Approx::custom()
          .epsilon(128. * std::numeric_limits<double>::epsilon())
          .scale(0.);
  std::vector<double> speeds{};
  std::vector<double> fields{};
  std::vector<double> densities{};
  std::vector<double> expected{};
  const std::array<double, 6> test_speeds{{-0.499, 0.,
                                           std::nextafter(0.499, 0.), 0.499,
                                           std::nextafter(0.499, 1.), 0.5}};
  const std::array<double, 3> test_densities{{0., 1.e-20, 1.}};
  const std::array<Vector, 6> reference_momenta{{
      {{-0x0p+0, -0x1.7491414188a67p-66, -0x1.f8ec48a1cb3e4p+0}},
      {{0x0p+0, 0x0p+0, 0x0p+0}},
      {{0x0p+0, 0x1.7491414188a62p-66, 0x1.f8ec48a1cb3ddp+0}},
      {{0x0p+0, 0x1.7491414188a67p-66, 0x1.f8ec48a1cb3e4p+0}},
      {{0x0p+0, 0x1.7491414188a6cp-66, 0x1.f8ec48a1cb3ebp+0}},
      {{0x0p+0, 0x1.79ca10c924223p-66, 0x1p+1}},
  }};
  for (size_t speed_index = 0; speed_index < test_speeds.size();
       ++speed_index) {
    const double speed = gsl::at(test_speeds, speed_index);
    for (const double field : {0., 1., std::ldexp(1., 30)}) {
      for (size_t density_index = 0; density_index < test_densities.size();
           ++density_index) {
        const double rho = gsl::at(test_densities, density_index);
        CAPTURE(speed, field, rho);
        const double reference =
            gsl::at(gsl::at(reference_momenta, speed_index), density_index);
        const Vector velocity{{speed, speed, speed}};
        const Vector magnetic_field{{field, field, field}};
        const auto result = sources::joint_sources(rho, 0., 0., velocity,
                                                   magnetic_field, identity);
        for (size_t d = 1; d < 4; ++d) {
          CHECK_ITERABLE_CUSTOM_APPROX(gsl::at(result, d), reference, relative);
        }
        speeds.push_back(speed);
        fields.push_back(field);
        densities.push_back(rho);
        expected.push_back(reference);
      }
    }
  }
  // Exercise all states in one heterogeneous batch, including empty matter,
  // tiny fluid momentum, both signs of velocity and different field strengths.
  const size_t n = speeds.size();
  DataVector rho(n);
  DataVector speed(n);
  DataVector field(n);
  for (size_t k = 0; k < n; ++k) {
    rho[k] = densities[k];
    speed[k] = speeds[k];
    field[k] = fields[k];
  }
  const DataVector zeros(n, 0.);
  const std::array<DataVector, 3> velocity{{speed, speed, speed}};
  const std::array<DataVector, 3> magnetic_field{{field, field, field}};
  std::array<std::array<DataVector, 3>, 3> metric{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t j = 0; j < 3; ++j) {
      gsl::at(gsl::at(metric, i), j) = DataVector(n, i == j ? 1. : 0.);
    }
  }
  const auto result = sources::joint_sources(rho, zeros, zeros, velocity,
                                             magnetic_field, metric);
  for (size_t k = 0; k < n; ++k) {
    CAPTURE(k, speeds[k], fields[k], densities[k]);
    for (size_t d = 1; d < 4; ++d) {
      CHECK_ITERABLE_CUSTOM_APPROX(gsl::at(result, d)[k], expected[k],
                                   relative);
    }
  }
}

void test_thermodynamics() {
  const auto relative =
      Approx::custom()
          .epsilon(128. * std::numeric_limits<double>::epsilon())
          .scale(0.);
  // Rest-frame fluid: E=rho(1+epsilon), S^i=0, trace=3p. Retain
  // contributions below the old Ccsn enthalpy-denominator floor.
  for (const double rho : {0.5e-300, 1.e-300, 2.e-300, 1.}) {
    for (const double epsilon : {-0.5, 0., 0.25}) {
      CAPTURE(rho, epsilon);
      const double pressure = 0.25 * rho;
      const auto result =
          sources::joint_sources(rho, epsilon, pressure, zero, zero, identity);
      CHECK_ITERABLE_CUSTOM_APPROX(result[0], rho * (1. + epsilon), relative);
      CHECK_ITERABLE_CUSTOM_APPROX(result[4], 3. * pressure, relative);
      for (size_t i = 1; i < 4; ++i) {
        CHECK(gsl::at(result, i) == 0.);
      }
    }
  }
  // Small kinetic stress must not be reconstructed by subtracting 1 from W^2.
  const auto slow = sources::joint_sources(2., 0., 0., Vector{{1.e-10, 0., 0.}},
                                           zero, identity);
  CHECK_ITERABLE_CUSTOM_APPROX(slow[4], 2.e-20 / (1. - 1.e-20), relative);
  // Negative epsilon is allowed; pressure can make H nonnegative.
  const auto boundary =
      sources::joint_sources(1., -2., 1., zero, zero, identity);
  CHECK(boundary[0] == -1.);
  CHECK(boundary[4] == 3.);
}

void test_vacuum() {
  // rho=p=0 retains ideal-MHD electric-field kinematics for the supplied v.
  const Vector velocity{{0.25, 0., 0.}};
  const Vector magnetic_field{{0., 2., 0.}};
  const std::array<double, 5> expected{{2.125, 1., 0., 0., 2.125}};
  for (const double epsilon : {-std::numeric_limits<double>::max(), -2., 0.,
                               std::numeric_limits<double>::max()}) {
    CAPTURE(epsilon);
    CHECK(sources::joint_sources(0., epsilon, 0., velocity, magnetic_field,
                                 identity) == expected);
    const std::array<double, 5> all_zero{};
    CHECK(sources::joint_sources(0., epsilon, 0., velocity, zero, identity) ==
          all_zero);
  }
}

void test_metric_and_batch() {
  // Constant coordinate transform x'=U x, U=[[1,1,0],[0,1,0],[0,0,1]].
  // g=U^T U; E and trace are invariant and S'=U S.
  const Metric metric{{{1., 1., 0.}, {1., 2., 0.}, {0., 0., 1.}}};
  const Vector velocity{{0.125, 0.25, 0.}};
  const Vector magnetic_field{{2., -1., 0.5}};
  const Vector flat_velocity{{0.375, 0.25, 0.}};
  const Vector flat_field{{1., -1., 0.5}};
  const auto curved =
      sources::joint_sources(1., 0.25, 0.125, velocity, magnetic_field, metric);
  const auto flat = sources::joint_sources(1., 0.25, 0.125, flat_velocity,
                                           flat_field, identity);
  CHECK_ITERABLE_APPROX(curved[0], flat[0]);
  CHECK_ITERABLE_APPROX(curved[4], flat[4]);
  CHECK_ITERABLE_APPROX(curved[1], flat[1] - flat[2]);
  CHECK_ITERABLE_APPROX(curved[2], flat[2]);
  CHECK_ITERABLE_APPROX(curved[3], flat[3]);

  const DataVector rho{1., 0., 0.5e-300};
  const DataVector epsilon{0.25, -2., -0.5};
  const DataVector pressure{0.125, 0., 0.125e-300};
  std::array<DataVector, 3> v{};
  std::array<DataVector, 3> b{};
  std::array<std::array<DataVector, 3>, 3> g{};
  for (size_t i = 0; i < 3; ++i) {
    gsl::at(v, i) = DataVector(3, gsl::at(velocity, i));
    gsl::at(b, i) = DataVector(3, gsl::at(magnetic_field, i));
    for (size_t j = 0; j < 3; ++j) {
      gsl::at(gsl::at(g, i), j) = DataVector(3, gsl::at(gsl::at(metric, i), j));
    }
  }
  const auto batch = sources::joint_sources(rho, epsilon, pressure, v, b, g);
  for (size_t k = 0; k < 3; ++k) {
    const auto scalar = sources::joint_sources(
        rho[k], epsilon[k], pressure[k], velocity, magnetic_field, metric);
    for (size_t j = 0; j < 5; ++j) {
      CHECK(gsl::at(gsl::at(batch, j), k) == gsl::at(scalar, j));
    }
  }
  // The exact-H counterexample must also reject a single invalid vector lane.
  auto invalid_rho = rho;
  auto invalid_epsilon = epsilon;
  auto invalid_pressure = pressure;
  invalid_rho[2] = std::numeric_limits<double>::denorm_min();
  invalid_epsilon[2] = -1.25;
  invalid_pressure[2] = 0.;
  CHECK_THROWS_AS(sources::joint_sources(invalid_rho, invalid_epsilon,
                                         invalid_pressure, v, b, g),
                  std::domain_error);
  invalid_pressure = pressure;
  invalid_pressure[1] = 0.1;
  CHECK_THROWS_AS(
      sources::joint_sources(rho, epsilon, invalid_pressure, v, b, g),
      std::domain_error);
  CHECK_THROWS_AS(
      sources::joint_sources(rho, DataVector(2, 0.), pressure, v, b, g),
      std::invalid_argument);
}

void test_rejections() {
  const auto run = [](double rho, double epsilon, double pressure) {
    return sources::joint_sources(rho, epsilon, pressure, zero, zero, identity);
  };
  CHECK_THROWS_AS(run(-1., 0., 0.), std::domain_error);
  CHECK_THROWS_AS(run(0., 0., 1.), std::domain_error);
  CHECK_THROWS_AS(run(1., 0., -1.), std::domain_error);
  CHECK_THROWS_AS(run(0., std::numeric_limits<double>::quiet_NaN(), 0.),
                  std::domain_error);
  CHECK_THROWS_AS(run(1., std::numeric_limits<double>::infinity(), 0.),
                  std::domain_error);
  CHECK_THROWS_AS(run(1., -2., 0.), std::domain_error);
  // Exact H is negative although the computed rho+rho*epsilon can round to
  // zero.
  const double tiny = std::numeric_limits<double>::denorm_min();
  CHECK_THROWS_AS(run(tiny, -1.25, 0.), std::domain_error);
  CHECK_NOTHROW(run(tiny, -1.25, tiny));
  CHECK_NOTHROW(run(tiny, -2., tiny));
  CHECK_THROWS_AS(
      sources::joint_sources(1., 0., 0., Vector{{1., 0., 0.}}, zero, identity),
      std::domain_error);
  auto metric = identity;
  metric[0][0] = 0.;
  CHECK_THROWS_AS(sources::joint_sources(1., 0., 0., zero, zero, metric),
                  std::domain_error);
  metric = identity;
  metric[0][1] = 0.5;
  CHECK_THROWS_AS(sources::joint_sources(1., 0., 0., zero, zero, metric),
                  std::domain_error);
  CHECK_THROWS_AS(sources::joint_sources(
                      DataVector{}, DataVector{}, DataVector{},
                      std::array<DataVector, 3>{}, std::array<DataVector, 3>{},
                      std::array<std::array<DataVector, 3>, 3>{}),
                  std::invalid_argument);
}

void test_exact_enthalpy_sign() {
  // H = rho * (1 - 2) + p changes sign exactly at p = rho. Exercise
  // subnormal densities and shifts on both sides of a multiprecision limb.
  // Adjacent received pressures must stay on their respective sides of zero.
  for (const int exponent :
       {-1070, -1022, -1011, -64, -12, -11, 0,  20,  21,  22,
        52,    53,    54,    64,  84,  85,  86, 117, 512, 1020}) {
    CAPTURE(exponent);
    const double rho = std::ldexp(1., exponent);
    CHECK(sources::exact_enthalpy_nonnegative(rho, -2., rho));
    CHECK_FALSE(
        sources::exact_enthalpy_nonnegative(rho, -2., std::nextafter(rho, 0.)));
    CHECK(sources::exact_enthalpy_nonnegative(
        rho, -2.,
        std::nextafter(rho, std::numeric_limits<double>::infinity())));
    // A fractional negative internal energy also has an exact dyadic boundary.
    const double pressure = 0.25 * rho;
    CHECK(sources::exact_enthalpy_nonnegative(rho, -1.25, pressure));
    CHECK_FALSE(sources::exact_enthalpy_nonnegative(
        rho, -1.25, std::nextafter(pressure, 0.)));
    CHECK(sources::exact_enthalpy_nonnegative(
        rho, -1.25,
        std::nextafter(pressure, std::numeric_limits<double>::infinity())));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Hydro.PrimitiveSources",
                  "[Unit][Hydro]") {
  test_parallel_field_cancellation();
  test_thermodynamics();
  test_vacuum();
  test_metric_and_batch();
  test_rejections();
  test_exact_enthalpy_sign();
}
