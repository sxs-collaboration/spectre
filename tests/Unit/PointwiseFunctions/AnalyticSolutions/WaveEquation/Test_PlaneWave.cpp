// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <array>
#include <cmath>

#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

inline tnsr::I<double, 1, Frame::Inertial> extract_point_from_coords(
    const size_t offset, const tnsr::I<DataVector, 1>& x) {
  return tnsr::I<double, 1, Frame::Inertial>{
      std::array<double, 1>{{x.get(0)[offset]}}};
}

inline tnsr::I<double, 2, Frame::Inertial> extract_point_from_coords(
    const size_t offset, const tnsr::I<DataVector, 2>& x) {
  return tnsr::I<double, 2, Frame::Inertial>{
      std::array<double, 2>{{x.get(0)[offset], x.get(1)[offset]}}};
}

inline tnsr::I<double, 3, Frame::Inertial> extract_point_from_coords(
    const size_t offset, const tnsr::I<DataVector, 3>& x) {
  return tnsr::I<double, 3, Frame::Inertial>{std::array<double, 3>{
      {x.get(0)[offset], x.get(1)[offset], x.get(2)[offset]}}};
}

template <size_t Dim>
void check_solution_x(const double kx, const double omega, const DataVector& u,
                      const ScalarWave::Solutions::PlaneWave<Dim>& pw,
                      const tnsr::I<DataVector, Dim>& x, const double t) {
  const DataVector psi = cube(u);
  const DataVector dpsi_dt = -3.0 * omega * square(u);
  const DataVector dpsi_dx = 3.0 * kx * square(u);
  const DataVector d2psi_dt2 = 6.0 * square(omega) * u;
  const DataVector d2psi_dtdx = -6.0 * omega * kx * u;
  const DataVector d2psi_dxdx = 6.0 * square(kx) * u;
  CHECK_ITERABLE_APPROX(psi, pw.psi(x, t).get());
  CHECK_ITERABLE_APPROX(dpsi_dt, pw.dpsi_dt(x, t).get());
  CHECK_ITERABLE_APPROX(d2psi_dt2, pw.d2psi_dt2(x, t).get());
  CHECK_ITERABLE_APPROX(dpsi_dx, pw.dpsi_dx(x, t).get(0));
  CHECK_ITERABLE_APPROX(d2psi_dtdx, pw.d2psi_dtdx(x, t).get(0));
  CHECK_ITERABLE_APPROX(d2psi_dxdx, pw.d2psi_dxdx(x, t).get(0, 0));
  for (size_t s = 0; s < u.size(); ++s) {
    const auto p = extract_point_from_coords(s, x);
    CHECK(approx(psi[s]) == pw.psi(p, t).get());
    CHECK(approx(dpsi_dt[s]) == pw.dpsi_dt(p, t).get());
    CHECK(approx(d2psi_dt2[s]) == pw.d2psi_dt2(p, t).get());
    CHECK(approx(dpsi_dx[s]) == pw.dpsi_dx(p, t).get(0));
    CHECK(approx(d2psi_dtdx[s]) == pw.d2psi_dtdx(p, t).get(0));
    CHECK(approx(d2psi_dxdx[s]) == pw.d2psi_dxdx(p, t).get(0, 0));
  }

  CHECK_ITERABLE_APPROX(get<ScalarWave::Psi>(pw.evolution_variables(x, t)),
                        pw.psi(x, t));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Phi<Dim>>(pw.evolution_variables(x, t)),
                        pw.dpsi_dx(x, t));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Pi>(pw.evolution_variables(x, t)),
                        Scalar<DataVector>(-1.0 * pw.dpsi_dt(x, t).get()));

  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Psi>>(pw.dt_evolution_variables(x, t)),
      pw.dpsi_dt(x, t));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Phi<Dim>>>(pw.dt_evolution_variables(x, t)),
      pw.d2psi_dtdx(x, t));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Pi>>(pw.dt_evolution_variables(x, t)),
      Scalar<DataVector>(-1.0 * pw.d2psi_dt2(x, t).get()));
}

template <size_t Dim>
void check_solution_y(const double kx, const double ky, const double omega,
                      const DataVector& u,
                      const ScalarWave::Solutions::PlaneWave<Dim>& pw,
                      const tnsr::I<DataVector, Dim>& x, const double t) {
  const DataVector dpsi_dy = 3.0 * ky * square(u);
  const DataVector d2psi_dtdy = -6.0 * omega * ky * u;
  const DataVector d2psi_dxdy = 6.0 * kx * ky * u;
  const DataVector d2psi_dydy = 6.0 * square(ky) * u;
  CHECK_ITERABLE_APPROX(dpsi_dy, pw.dpsi_dx(x, t).get(1));
  CHECK_ITERABLE_APPROX(d2psi_dtdy, pw.d2psi_dtdx(x, t).get(1));
  CHECK_ITERABLE_APPROX(d2psi_dxdy, pw.d2psi_dxdx(x, t).get(0, 1));
  CHECK_ITERABLE_APPROX(d2psi_dxdy, pw.d2psi_dxdx(x, t).get(1, 0));
  CHECK_ITERABLE_APPROX(d2psi_dydy, pw.d2psi_dxdx(x, t).get(1, 1));
  for (size_t s = 0; s < u.size(); ++s) {
    const auto p = extract_point_from_coords(s, x);
    CHECK(approx(dpsi_dy[s]) == pw.dpsi_dx(p, t).get(1));
    CHECK(approx(d2psi_dtdy[s]) == pw.d2psi_dtdx(p, t).get(1));
    CHECK(approx(d2psi_dxdy[s]) == pw.d2psi_dxdx(p, t).get(0, 1));
    CHECK(approx(d2psi_dxdy[s]) == pw.d2psi_dxdx(p, t).get(1, 0));
    CHECK(approx(d2psi_dydy[s]) == pw.d2psi_dxdx(p, t).get(1, 1));
  }

  CHECK_ITERABLE_APPROX(get<ScalarWave::Psi>(pw.evolution_variables(x, t)),
                        pw.psi(x, t));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Phi<Dim>>(pw.evolution_variables(x, t)),
                        pw.dpsi_dx(x, t));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Pi>(pw.evolution_variables(x, t)),
                        Scalar<DataVector>(-1.0 * pw.dpsi_dt(x, t).get()));

  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Psi>>(pw.dt_evolution_variables(x, t)),
      pw.dpsi_dt(x, t));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Phi<Dim>>>(pw.dt_evolution_variables(x, t)),
      pw.d2psi_dtdx(x, t));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Pi>>(pw.dt_evolution_variables(x, t)),
      Scalar<DataVector>(-1.0 * pw.d2psi_dt2(x, t).get()));
}

void check_solution_z(const double kx, const double ky, const double kz,
                      const double omega, const DataVector& u,
                      const ScalarWave::Solutions::PlaneWave<3>& pw,
                      const tnsr::I<DataVector, 3>& x, const double t) {
  const DataVector dpsi_dz = 3.0 * kz * square(u);
  const DataVector d2psi_dtdz = -6.0 * omega * kz * u;
  const DataVector d2psi_dxdz = 6.0 * kx * kz * u;
  const DataVector d2psi_dydz = 6.0 * ky * kz * u;
  const DataVector d2psi_dzdz = 6.0 * square(kz) * u;
  CHECK_ITERABLE_APPROX(dpsi_dz, pw.dpsi_dx(x, t).get(2));
  CHECK_ITERABLE_APPROX(d2psi_dtdz, pw.d2psi_dtdx(x, t).get(2));
  CHECK_ITERABLE_APPROX(d2psi_dxdz, pw.d2psi_dxdx(x, t).get(0, 2));
  CHECK_ITERABLE_APPROX(d2psi_dxdz, pw.d2psi_dxdx(x, t).get(2, 0));
  CHECK_ITERABLE_APPROX(d2psi_dydz, pw.d2psi_dxdx(x, t).get(2, 1));
  CHECK_ITERABLE_APPROX(d2psi_dydz, pw.d2psi_dxdx(x, t).get(1, 2));
  CHECK_ITERABLE_APPROX(d2psi_dzdz, pw.d2psi_dxdx(x, t).get(2, 2));
  for (size_t s = 0; s < u.size(); ++s) {
    const auto p = extract_point_from_coords(s, x);
    CHECK(approx(dpsi_dz[s]) == pw.dpsi_dx(p, t).get(2));
    CHECK(approx(d2psi_dtdz[s]) == pw.d2psi_dtdx(p, t).get(2));
    CHECK(approx(d2psi_dxdz[s]) == pw.d2psi_dxdx(p, t).get(0, 2));
    CHECK(approx(d2psi_dxdz[s]) == pw.d2psi_dxdx(p, t).get(2, 0));
    CHECK(approx(d2psi_dydz[s]) == pw.d2psi_dxdx(p, t).get(2, 1));
    CHECK(approx(d2psi_dydz[s]) == pw.d2psi_dxdx(p, t).get(1, 2));
    CHECK(approx(d2psi_dzdz[s]) == pw.d2psi_dxdx(p, t).get(2, 2));
  }

  CHECK_ITERABLE_APPROX(get<ScalarWave::Psi>(pw.evolution_variables(x, t)),
                        pw.psi(x, t));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Phi<3>>(pw.evolution_variables(x, t)),
                        pw.dpsi_dx(x, t));
  CHECK_ITERABLE_APPROX(get<ScalarWave::Pi>(pw.evolution_variables(x, t)),
                        Scalar<DataVector>(-1.0 * pw.dpsi_dt(x, t).get()));

  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Psi>>(pw.dt_evolution_variables(x, t)),
      pw.dpsi_dt(x, t));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Phi<3>>>(pw.dt_evolution_variables(x, t)),
      pw.d2psi_dtdx(x, t));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Pi>>(pw.dt_evolution_variables(x, t)),
      Scalar<DataVector>(-1.0 * pw.d2psi_dt2(x, t).get()));
}

void test_1d() {
  const double k = -1.5;
  const double center_x = 2.4;
  const double omega = std::abs(k);
  const double t = 3.1;
  const double x1 = -0.2;
  const double x2 = 8.7;
  const tnsr::I<DataVector, 1> x(DataVector({x1, x2}));
  const DataVector u(
      {k * (x1 - center_x) - omega * t, k * (x2 - center_x) - omega * t});
  const ScalarWave::Solutions::PlaneWave<1> pw(
      {{k}}, {{center_x}}, std::make_unique<MathFunctions::PowX>(3));
  check_solution_x(k, omega, u, pw, x, t);

  Parallel::register_derived_classes_with_charm<MathFunction<1>>();
  const auto deserialized_pw = serialize_and_deserialize(pw);
  check_solution_x(k, omega, u, deserialized_pw, x, t);

  test_creation<ScalarWave::Solutions::PlaneWave<1>>(
      "  WaveVector: [3.5]\n"
      "  Center: [3.5]\n"
      "  Profile:\n"
      "    PowX:\n"
      "      Power: 4");
}

void test_2d() {
  const double kx = 1.5;
  const double ky = -7.2;
  const double center_x = 2.4;
  const double center_y = -4.8;
  const double omega = std::sqrt(square(kx) + square(ky));
  const double t = 3.1;
  const double x1 = -10.2;
  const double x2 = 8.7;
  const double y1 = -1.98;
  const double y2 = 48.27;
  const tnsr::I<DataVector, 2> x{
      std::array<DataVector, 2>{{DataVector({x1, x2}), DataVector({y1, y2})}}};
  const DataVector u({kx * (x1 - center_x) + ky * (y1 - center_y) - omega * t,
                      kx * (x2 - center_x) + ky * (y2 - center_y) - omega * t});
  const ScalarWave::Solutions::PlaneWave<2> pw(
      {{kx, ky}}, {{center_x, center_y}},
      std::make_unique<MathFunctions::PowX>(3));
  check_solution_x(kx, omega, u, pw, x, t);
  check_solution_y(kx, ky, omega, u, pw, x, t);

  Parallel::register_derived_classes_with_charm<MathFunction<1>>();
  const auto deserialized_pw = serialize_and_deserialize(pw);
  check_solution_x(kx, omega, u, deserialized_pw, x, t);
  check_solution_y(kx, ky, omega, u, deserialized_pw, x, t);

  test_creation<ScalarWave::Solutions::PlaneWave<2>>(
      "  WaveVector: [-2, 3.5]\n"
      "  Center: [-2, 3.5]\n"
      "  Profile:\n"
      "    PowX:\n"
      "      Power: 4");
}

void test_3d() {
  const double kx = 1.5;
  const double ky = -7.2;
  const double kz = 2.7;
  const double center_x = 2.4;
  const double center_y = -4.8;
  const double center_z = 8.4;
  const double omega = std::sqrt(square(kx) + square(ky) + square(kz));
  const double t = 3.1;
  const double x1 = -10.2;
  const double x2 = 8.7;
  const double y1 = -1.98;
  const double y2 = 48.27;
  const double z1 = 2.2;
  const double z2 = 1.1;
  const tnsr::I<DataVector, 3> x{std::array<DataVector, 3>{
      {DataVector({x1, x2}), DataVector({y1, y2}), DataVector({z1, z2})}}};
  const DataVector u({kx * (x1 - center_x) + ky * (y1 - center_y) +
                          kz * (z1 - center_z) - omega * t,
                      kx * (x2 - center_x) + ky * (y2 - center_y) +
                          kz * (z2 - center_z) - omega * t});
  const ScalarWave::Solutions::PlaneWave<3> pw(
      {{kx, ky, kz}}, {{center_x, center_y, center_z}},
      std::make_unique<MathFunctions::PowX>(3));
  check_solution_x(kx, omega, u, pw, x, t);
  check_solution_y(kx, ky, omega, u, pw, x, t);
  check_solution_z(kx, ky, kz, omega, u, pw, x, t);

  Parallel::register_derived_classes_with_charm<MathFunction<1>>();
  const auto deserialized_pw = serialize_and_deserialize(pw);
  check_solution_x(kx, omega, u, deserialized_pw, x, t);
  check_solution_y(kx, ky, omega, u, deserialized_pw, x, t);
  check_solution_z(kx, ky, kz, omega, u, deserialized_pw, x, t);

  test_creation<ScalarWave::Solutions::PlaneWave<3>>(
      "  WaveVector: [-1, -2, 3.5]\n"
      "  Center: [-1, -2, 3.5]\n"
      "  Profile:\n"
      "    PowX:\n"
      "      Power: 4");
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.WaveEquation.PlaneWave",
    "[PointwiseFunctions][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}
