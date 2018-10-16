// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <memory>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"           // IWYU pragma: keep
#include "Evolution/Systems/ScalarWave/Tags.hpp"  // IWYU pragma: keep
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticSolutions/WaveEquation/PlaneWave.hpp"
#include "PointwiseFunctions/MathFunctions/MathFunction.hpp"
#include "PointwiseFunctions/MathFunctions/PowX.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/TestCreation.hpp"
#include "tests/Unit/TestHelpers.hpp"

// IWYU pragma: no_forward_declare MathFunction
// IWYU pragma: no_forward_declare Tensor

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

template <size_t Dim, size_t DimSolution>
void check_solution(
    const DataVector& expected_psi, const DataVector& expected_dpsi_dt,
    const DataVector& expected_d2psi_dt2, const DataVector& expected_dpsi_dlast,
    const std::array<DataVector, Dim + 1>& expected_second_derivs,
    const ScalarWave::Solutions::PlaneWave<DimSolution>& pw,
    const tnsr::I<DataVector, DimSolution>& x, const double t) noexcept {
  // expected_second_derivs is:
  // 0 -> d^2 psi / dt dx^(Dim-1)
  // 1-3 -> d^2 psi / dx^(Dim - 1) dx^i where i is in [0, 2]
  CHECK_ITERABLE_APPROX(expected_psi, pw.psi(x, t).get());
  CHECK_ITERABLE_APPROX(expected_dpsi_dt, pw.dpsi_dt(x, t).get());
  CHECK_ITERABLE_APPROX(expected_d2psi_dt2, pw.d2psi_dt2(x, t).get());
  CHECK_ITERABLE_APPROX(expected_dpsi_dlast, pw.dpsi_dx(x, t).get(Dim - 1));
  CHECK_ITERABLE_APPROX(expected_second_derivs[0],
                        pw.d2psi_dtdx(x, t).get(Dim - 1));
  for (size_t i = 0; i < Dim; ++i) {
    CHECK_ITERABLE_APPROX(gsl::at(expected_second_derivs, i + 1),
                          pw.d2psi_dxdx(x, t).get(i, Dim - 1));
    if (i < Dim - 1) {
      CHECK_ITERABLE_APPROX(gsl::at(expected_second_derivs, i + 1),
                            pw.d2psi_dxdx(x, t).get(Dim - 1, i));
    }
  }

  for (size_t s = 0; s < x.get(0).size(); ++s) {
    const auto p = extract_point_from_coords(s, x);
    CHECK_ITERABLE_APPROX(expected_psi[s], pw.psi(p, t).get());
    CHECK_ITERABLE_APPROX(expected_dpsi_dt[s], pw.dpsi_dt(p, t).get());
    CHECK_ITERABLE_APPROX(expected_d2psi_dt2[s], pw.d2psi_dt2(p, t).get());
    CHECK_ITERABLE_APPROX(expected_dpsi_dlast[s],
                          pw.dpsi_dx(p, t).get(Dim - 1));
    CHECK_ITERABLE_APPROX(expected_second_derivs[0][s],
                          pw.d2psi_dtdx(p, t).get(Dim - 1));
    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(gsl::at(expected_second_derivs, i + 1)[s],
                            pw.d2psi_dxdx(p, t).get(i, Dim - 1));
      if (i < Dim - 1) {
        CHECK_ITERABLE_APPROX(gsl::at(expected_second_derivs, i + 1)[s],
                              pw.d2psi_dxdx(p, t).get(Dim - 1, i));
      }
    }
  }

  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Psi>(
          pw.variables(x, t,
                       tmpl::list<ScalarWave::Pi, ScalarWave::Phi<DimSolution>,
                                  ScalarWave::Psi>{})),
      pw.psi(x, t));
  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Phi<DimSolution>>(
          pw.variables(x, t,
                       tmpl::list<ScalarWave::Pi, ScalarWave::Phi<DimSolution>,
                                  ScalarWave::Psi>{})),
      pw.dpsi_dx(x, t));
  CHECK_ITERABLE_APPROX(
      get<ScalarWave::Pi>(
          pw.variables(x, t,
                       tmpl::list<ScalarWave::Pi, ScalarWave::Phi<DimSolution>,
                                  ScalarWave::Psi>{})),
      Scalar<DataVector>(-1.0 * pw.dpsi_dt(x, t).get()));

  CHECK_ITERABLE_APPROX(get<Tags::dt<ScalarWave::Psi>>(pw.variables(
                            x, t,
                            tmpl::list<Tags::dt<ScalarWave::Pi>,
                                       Tags::dt<ScalarWave::Phi<DimSolution>>,
                                       Tags::dt<ScalarWave::Psi>>{})),
                        pw.dpsi_dt(x, t));
  CHECK_ITERABLE_APPROX(
      get<Tags::dt<ScalarWave::Phi<DimSolution>>>(
          pw.variables(x, t,
                       tmpl::list<Tags::dt<ScalarWave::Pi>,
                                  Tags::dt<ScalarWave::Phi<DimSolution>>,
                                  Tags::dt<ScalarWave::Psi>>{})),
      pw.d2psi_dtdx(x, t));
  CHECK_ITERABLE_APPROX(get<Tags::dt<ScalarWave::Pi>>(pw.variables(
                            x, t,
                            tmpl::list<Tags::dt<ScalarWave::Pi>,
                                       Tags::dt<ScalarWave::Phi<DimSolution>>,
                                       Tags::dt<ScalarWave::Psi>>{})),
                        Scalar<DataVector>(-1.0 * pw.d2psi_dt2(x, t).get()));
}

void test_1d() {
  const double kx = -1.5;
  const double center_x = 2.4;
  const double omega = std::abs(kx);
  const double t = 3.1;
  const double x1 = -0.2;
  const double x2 = 8.7;
  const tnsr::I<DataVector, 1> x(DataVector({x1, x2}));
  const DataVector u(
      {kx * (x1 - center_x) - omega * t, kx * (x2 - center_x) - omega * t});
  const ScalarWave::Solutions::PlaneWave<1> pw(
      {{kx}}, {{center_x}}, std::make_unique<MathFunctions::PowX>(3));
  check_solution<1>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * kx * square(u),
      std::array<DataVector, 2>{{-6.0 * omega * kx * u, 6.0 * square(kx) * u}},
      pw, x, t);

  Parallel::register_derived_classes_with_charm<MathFunction<1>>();
  const auto deserialized_pw = serialize_and_deserialize(pw);
  check_solution<1>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * kx * square(u),
      std::array<DataVector, 2>{{-6.0 * omega * kx * u, 6.0 * square(kx) * u}},
      deserialized_pw, x, t);

  const auto created_solution =
      test_creation<ScalarWave::Solutions::PlaneWave<1>>(
          "  WaveVector: [-1.5]\n"
          "  Center: [2.4]\n"
          "  Profile:\n"
          "    PowX:\n"
          "      Power: 3");
  CHECK(
      created_solution.variables(
          x, t,
          tmpl::list<ScalarWave::Pi, ScalarWave::Phi<1>, ScalarWave::Psi>{}) ==
      pw.variables(
          x, t,
          tmpl::list<ScalarWave::Pi, ScalarWave::Phi<1>, ScalarWave::Psi>{}));
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
  check_solution<1>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * kx * square(u),
      std::array<DataVector, 2>{{-6.0 * omega * kx * u, 6.0 * square(kx) * u}},
      pw, x, t);
  check_solution<2>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * ky * square(u),
      std::array<DataVector, 3>{
          {-6.0 * omega * ky * u, 6.0 * kx * ky * u, 6.0 * square(ky) * u}},
      pw, x, t);

  Parallel::register_derived_classes_with_charm<MathFunction<1>>();
  const auto deserialized_pw = serialize_and_deserialize(pw);
  check_solution<1>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * kx * square(u),
      std::array<DataVector, 2>{{-6.0 * omega * kx * u, 6.0 * square(kx) * u}},
      deserialized_pw, x, t);
  check_solution<2>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * ky * square(u),
      std::array<DataVector, 3>{
          {-6.0 * omega * ky * u, 6.0 * kx * ky * u, 6.0 * square(ky) * u}},
      deserialized_pw, x, t);

  const auto created_solution =
      test_creation<ScalarWave::Solutions::PlaneWave<2>>(
          "  WaveVector: [1.5, -7.2]\n"
          "  Center: [2.4, -4.8]\n"
          "  Profile:\n"
          "    PowX:\n"
          "      Power: 3");
  CHECK(
      created_solution.variables(
          x, t,
          tmpl::list<ScalarWave::Pi, ScalarWave::Phi<2>, ScalarWave::Psi>{}) ==
      pw.variables(
          x, t,
          tmpl::list<ScalarWave::Pi, ScalarWave::Phi<2>, ScalarWave::Psi>{}));
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
  check_solution<1>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * kx * square(u),
      std::array<DataVector, 2>{{-6.0 * omega * kx * u, 6.0 * square(kx) * u}},
      pw, x, t);
  check_solution<2>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * ky * square(u),
      std::array<DataVector, 3>{
          {-6.0 * omega * ky * u, 6.0 * kx * ky * u, 6.0 * square(ky) * u}},
      pw, x, t);
  check_solution<3>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * kz * square(u),
      std::array<DataVector, 4>{{-6.0 * omega * kz * u, 6.0 * kx * kz * u,
                                 6.0 * ky * kz * u, 6.0 * square(kz) * u}},
      pw, x, t);

  Parallel::register_derived_classes_with_charm<MathFunction<1>>();
  const auto deserialized_pw = serialize_and_deserialize(pw);
  check_solution<1>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * kx * square(u),
      std::array<DataVector, 2>{{-6.0 * omega * kx * u, 6.0 * square(kx) * u}},
      deserialized_pw, x, t);
  check_solution<2>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * ky * square(u),
      std::array<DataVector, 3>{
          {-6.0 * omega * ky * u, 6.0 * kx * ky * u, 6.0 * square(ky) * u}},
      deserialized_pw, x, t);
  check_solution<3>(
      cube(u), -3.0 * omega * square(u), 6.0 * square(omega) * u,
      3.0 * kz * square(u),
      std::array<DataVector, 4>{{-6.0 * omega * kz * u, 6.0 * kx * kz * u,
                                 6.0 * ky * kz * u, 6.0 * square(kz) * u}},
      deserialized_pw, x, t);

  const auto created_solution =
      test_creation<ScalarWave::Solutions::PlaneWave<3>>(
          "  WaveVector: [1.5, -7.2, 2.7]\n"
          "  Center: [2.4, -4.8, 8.4]\n"
          "  Profile:\n"
          "    PowX:\n"
          "      Power: 3");
  CHECK(
      created_solution.variables(
          x, t,
          tmpl::list<ScalarWave::Pi, ScalarWave::Phi<3>, ScalarWave::Psi>{}) ==
      pw.variables(
          x, t,
          tmpl::list<ScalarWave::Pi, ScalarWave::Phi<3>, ScalarWave::Psi>{}));
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.WaveEquation.PlaneWave",
    "[PointwiseFunctions][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}
