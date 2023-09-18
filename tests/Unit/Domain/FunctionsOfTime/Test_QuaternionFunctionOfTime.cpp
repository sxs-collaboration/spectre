// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/Serialization/Serialize.hpp"

namespace {
void test_serialization_versioning() {
  using QuatFoT = domain::FunctionsOfTime::QuaternionFunctionOfTime<2>;
  register_classes_with_charm<QuatFoT>();
  const std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> func(
      std::make_unique<QuatFoT>(
          0.0, std::array<DataVector, 1>{DataVector{{1.0, 0.0, 0.0, 0.0}}},
          std::array<DataVector, 3>{DataVector{3, 0.0},
                                    DataVector{0.0, 0.0, 3.78},
                                    DataVector{3, 0.0}},
          0.6));
  func->update(0.6, DataVector{3, 0.0}, 1.0);

  // Because of the implementation-defined sign, there's no way to
  // write char literals that won't cause narrowing errors.
  const auto vector_char = [](auto... values) {
    return std::vector<char>{static_cast<char>(values)...};
  };

  // After any serialization change, generate a new set of bytes with:
  // for (char c : serialize(func)) {
  //   printf("0x%02hhx, ", c);
  // }
  const auto serialization_v3 = vector_char(
      0x42, 0x0a, 0x2d, 0x34, 0x94, 0xc4, 0x33, 0x7f, 0x03, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f, 0x02, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x03, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x3d, 0x0a, 0xd7, 0xa3,
      0x70, 0x3d, 0x0e, 0x40, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xe3, 0x3f, 0x03, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xbe, 0x9f, 0x1a, 0x2f,
      0xdd, 0x24, 0x02, 0x40, 0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x3d, 0x0a, 0xd7, 0xa3, 0x70, 0x3d, 0x0e, 0x40,
      0x03, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0xf0, 0x3f, 0x02, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x04, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0xf0, 0x3f,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x33, 0x33, 0x33, 0x33, 0x33, 0x33, 0xe3, 0x3f, 0x04, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x6c, 0x9e, 0x00, 0x25, 0x11, 0x13, 0xdb, 0x3f,
      0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00,
      0x00, 0x00, 0x00, 0x00, 0x53, 0xa0, 0x09, 0xbb, 0xdd, 0xfe, 0xec, 0x3f);
  REQUIRE(serialize(func) == serialization_v3);

  std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime> deserialized{};
  deserialize(make_not_null(&deserialized), serialization_v3.data());
  CHECK(dynamic_cast<const QuatFoT&>(*func) ==
        dynamic_cast<const QuatFoT&>(*deserialized));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.QuaternionFunctionOfTime",
                  "[Unit][Domain]") {
  {
    INFO("QuaternionFunctionOfTime: Time bounds");
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        0.0, std::array<DataVector, 1>{DataVector{4, 0.0}},
        std::array<DataVector, 3>{DataVector{3, 0.0}, DataVector{3, 0.0},
                                  DataVector{3, 0.0}},
        0.5};
    CHECK(qfot.time_bounds() == std::array<double, 2>({0.0, 0.5}));
  }

  {
    INFO("QuaternionFunctionOfTime: Check output");
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        -0.1, std::array<DataVector, 1>{DataVector{{1.0, 0.0, 0.0, 0.0}}},
        std::array<DataVector, 3>{DataVector{{0.0, 0.0, -0.3}},
                                  DataVector{0.0, 0.0, 0.145},
                                  DataVector{3, 0.0}},
        0.5};

    const std::string expected_output =
        "Quaternion:\n"
        "t=-0.1: (1,0,0,0)\n"
        "Angle:\n"
        "t=-0.1: (0,0,-0.3) (0,0,0.145) (0,0,0)";

    const std::string output = get_output(qfot);
    CHECK(output == expected_output);
  }

  {
    INFO("QuaternionFunctionOfTime: Internal PiecewisePolynomial");
    DataVector init_omega{0.0, 0.0, 3.78};
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        0.0, std::array<DataVector, 1>{DataVector{{1.0, 0.0, 0.0, 0.0}}},
        std::array<DataVector, 3>{DataVector{3, 0.0}, init_omega,
                                  DataVector{3, 0.0}},
        0.5};
    domain::FunctionsOfTime::PiecewisePolynomial<2> pp{
        0.0,
        std::array<DataVector, 3>{DataVector{3, 0.0}, init_omega,
                                  DataVector{3, 0.0}},
        0.5};
    qfot.update(0.6, DataVector{3, 0.0}, 1.0);
    pp.update(0.6, DataVector{3, 0.0}, 1.0);

    CHECK(qfot.angle_func(0.4) == pp.func(0.4));
    CHECK(qfot.angle_func_and_deriv(0.4) == pp.func_and_deriv(0.4));
    CHECK(qfot.angle_func_and_2_derivs(0.4) == pp.func_and_2_derivs(0.4));
  }

  {
    INFO("QuaternionFunctionOfTime: pup, cloning, extra functions, copy/move");
    DataVector init_omega{0.0, 0.0, 1.0};
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        0.0, std::array<DataVector, 1>{DataVector{{1.0, 0.0, 0.0, 0.0}}},
        std::array<DataVector, 3>{DataVector{3, 0.0}, init_omega,
                                  DataVector{3, 0.0}},
        2.5};
    // Different expiration time to check comparison operators
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot2{
        0.0, std::array<DataVector, 1>{DataVector{{1.0, 0.0, 0.0, 0.0}}},
        std::array<DataVector, 3>{DataVector{3, 0.0}, init_omega,
                                  DataVector{3, 0.0}},
        3.0};

    auto qfot_ptr = qfot.get_clone();
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2>
        qfot_serialized_deserialized = serialize_and_deserialize(qfot);

    CHECK(qfot == qfot_serialized_deserialized);
    CHECK(qfot != qfot2);

    std::array<DataVector, 1> expected_func{
        DataVector{{cos(1.0), 0.0, 0.0, sin(1.0)}}};
    std::array<DataVector, 2> expected_func_and_deriv{
        DataVector{{cos(1.0), 0.0, 0.0, sin(1.0)}},
        DataVector{{-0.5 * sin(1.0), 0.0, 0.0, 0.5 * cos(1.0)}}};

    CHECK_ITERABLE_APPROX(qfot.quat_func(2.0), expected_func);
    CHECK_ITERABLE_APPROX(qfot.quat_func_and_deriv(2.0),
                          expected_func_and_deriv);
    CHECK_ITERABLE_APPROX(qfot.func(2.0), expected_func);
    CHECK_ITERABLE_APPROX(qfot.func_and_deriv(2.0), expected_func_and_deriv);
    CHECK_ITERABLE_APPROX(qfot_ptr->func(2.0), qfot.func(2.0));
    CHECK_ITERABLE_APPROX(qfot_ptr->func_and_deriv(2.0),
                          qfot.func_and_deriv(2.0));
    CHECK_ITERABLE_APPROX(qfot_serialized_deserialized.func(2.0),
                          qfot.func(2.0));
    CHECK_ITERABLE_APPROX(qfot_serialized_deserialized.func_and_deriv(2.0),
                          qfot.func_and_deriv(2.0));

    test_copy_semantics(qfot);
    test_move_semantics(std::move(qfot_serialized_deserialized), qfot);
  }

  {
    INFO("QuaternionFunctionOfTime: Constant omega");
    double t = 0.0;
    double expir_time = 0.5;
    const double omega_z = 1.3333;
    DataVector init_quat{1.0, 0.0, 0.0, 0.0};
    DataVector init_angle{0.0, 0.0, 0.0};
    DataVector init_omega{0.0, 0.0, omega_z};
    DataVector init_dtomega{0.0, 0.0, 0.0};

    // Construct QuaternionFunctionOfTime
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        t, std::array<DataVector, 1>{init_quat},
        std::array<DataVector, 3>{init_angle, init_omega, init_dtomega},
        expir_time};

    // Update stored PiecewisePolynomial with 0 2nd derivative so it's
    // constant. This will automatically update the stored quaternions as well
    const double time_step = 0.5;
    for (int i = 0; i < 15; i++) {
      t += time_step;
      expir_time += time_step;
      qfot.update(t, DataVector{3, 0.0}, expir_time);
    }

    // Get the quaternion and 2 derivatives at a certain time.
    double check_time = 5.398;
    const std::array<DataVector, 3> quat_func_and_2_derivs =
        qfot.quat_func_and_2_derivs(check_time);
    const std::array<DataVector, 3> quat_func_and_2_derivs2 =
        qfot.func_and_2_derivs(check_time);
    for (size_t i = 0; i < 3; i++) {
      CHECK_ITERABLE_APPROX(gsl::at(quat_func_and_2_derivs, i),
                            gsl::at(quat_func_and_2_derivs2, i));
    }

    // Analytic solution for constant omega
    // quat = ( cos(omega*t/2), 0, 0, sin(omega*t/2) )
    DataVector a_quat{{cos(0.5 * omega_z * check_time), 0.0, 0.0,
                       sin(0.5 * omega_z * check_time)}};
    DataVector a_dtquat{{-0.5 * omega_z * sin(0.5 * omega_z * check_time), 0.0,
                         0.0, 0.5 * omega_z * cos(0.5 * omega_z * check_time)}};
    DataVector a_dt2quat{
        {-0.25 * omega_z * omega_z * cos(0.5 * omega_z * check_time), 0.0, 0.0,
         -0.25 * omega_z * omega_z * sin(0.5 * omega_z * check_time)}};

    // Compare analytic solution to numerical
    Approx custom_approx = Approx::custom().epsilon(1.0e-12).scale(1.0);
    {
      INFO("  Compare quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[0], a_quat,
                                   custom_approx);
    }
    {
      INFO("  Compare derivative of quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[1], a_dtquat,
                                   custom_approx);
    }
    {
      INFO("  Compare second derivative of quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[2], a_dt2quat,
                                   custom_approx);
    }
  }

  {
    INFO("QuaternionFunctionOfTime: Linear Omega");
    double t = 0.0;
    double expir_time = 0.5;
    // phi(t) = fac1 * t^2 + fac2 * t + fac3
    const double fac1 = 0.25;
    const double fac2 = 0.5;
    const double fac3 = 0.0;
    // omega(t) = 2*fac1 * t + fac2
    // dtomega(t) = 2 * fac1 (constant)
    // dt2omega(t) = 0.0

    DataVector init_quat{1.0, 0.0, 0.0, 0.0};
    DataVector init_angle{3, 0.0};
    DataVector init_omega{{0.0, 0.0, fac2}};
    DataVector init_dtomega{{0.0, 0.0, 2.0 * fac1}};
    // Construct QuaternionFunctionOfTime
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        t, std::array<DataVector, 1>{init_quat},
        std::array<DataVector, 3>{init_angle, init_omega, init_dtomega},
        expir_time};

    // Update internal PiecewisePolynomial with constant 2nd derivative so
    // omega is linear. This will automatically update the stored quaternions as
    // well
    const double time_step = 0.5;
    for (int i = 0; i < 15; i++) {
      t += time_step;
      expir_time += time_step;
      qfot.update(t, DataVector{{0.0, 0.0, 2.0 * fac1}}, expir_time);
    }

    // Get the quaternion and 2 derivatives at a certain time.
    double check_time = 5.398;
    const std::array<DataVector, 3> quat_func_and_2_derivs =
        qfot.quat_func_and_2_derivs(check_time);
    const std::array<DataVector, 3> quat_func_and_2_derivs2 =
        qfot.func_and_2_derivs(check_time);
    for (size_t i = 0; i < 3; i++) {
      CHECK_ITERABLE_APPROX(gsl::at(quat_func_and_2_derivs, i),
                            gsl::at(quat_func_and_2_derivs2, i));
    }

    // phi(t) = fac1 * t^2 + fac2 * t + fac3
    const double phi =
        fac1 * check_time * check_time + fac2 * check_time + fac3;
    // omega(t) = 2*fac1 * t + fac2
    const double omega = 2 * fac1 * check_time + fac2;
    const double dtomega = 2 * fac1;

    DataVector a_quat{{cos(0.5 * phi), 0.0, 0.0, sin(0.5 * phi)}};
    DataVector a_dtquat{
        {-0.5 * a_quat[3] * omega, 0.0, 0.0, 0.5 * a_quat[0] * omega}};
    DataVector a_dt2quat{{-0.5 * (a_dtquat[3] * omega + a_quat[3] * dtomega),
                          0.0, 0.0,
                          0.5 * (a_dtquat[0] * omega + a_quat[0] * dtomega)}};

    // Compare analytic solution to numerical
    Approx custom_approx = Approx::custom().epsilon(5.0e-12).scale(1.0);
    {
      INFO("  Compare quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[0], a_quat,
                                   custom_approx);
    }
    {
      INFO("  Compare derivative of quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[1], a_dtquat,
                                   custom_approx);
    }
    {
      INFO("  Compare second derivative of quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[2], a_dt2quat,
                                   custom_approx);
    }
  }

  {
    INFO("QuaternionFunctionOfTime: Quadratic Omega");
    double t = 0.0;
    double expir_time = 0.5;
    // phi(t) = fac1 * t^3 + fac2 * t^2 + fac3 * t + fac4;
    const double fac1 = 0.2;
    const double fac2 = 0.3;
    const double fac3 = 0.4;
    const double fac4 = 0.0;
    // omega(t) = 3*fac1 * t^2 + 2*fac2 * t + fac3
    // dtomega(t) = 6*fac1 * t + 2* fac2
    // dt2omega(t) = 6 * fac1 (constant)

    DataVector init_quat{1.0, 0.0, 0.0, 0.0};
    DataVector init_angle{3, 0.0};
    DataVector init_omega{{0.0, 0.0, fac3}};
    DataVector init_dtomega{{0.0, 0.0, 2.0 * fac2}};
    DataVector init_dt2omega{{0.0, 0.0, 6.0 * fac1}};
    // Construct QuaternionFunctionOfTime
    domain::FunctionsOfTime::QuaternionFunctionOfTime<3> qfot{
        t, std::array<DataVector, 1>{init_quat},
        std::array<DataVector, 4>{init_angle, init_omega, init_dtomega,
                                  init_dt2omega},
        expir_time};

    // Update PiecewisePolynomial with constant 3rd derivative so omega is
    // quadratic. This will automatically update the stored quaternions as
    // well
    const double time_step = 0.5;
    for (int i = 0; i < 15; i++) {
      t += time_step;
      expir_time += time_step;
      qfot.update(t, DataVector{{0.0, 0.0, 6.0 * fac1}}, expir_time);
    }

    // Get the quaternion and 2 derivatives at a certain time.
    double check_time = 5.398;
    const std::array<DataVector, 3> quat_func_and_2_derivs =
        qfot.quat_func_and_2_derivs(check_time);
    const std::array<DataVector, 3> quat_func_and_2_derivs2 =
        qfot.func_and_2_derivs(check_time);
    for (size_t i = 0; i < 3; i++) {
      CHECK_ITERABLE_APPROX(gsl::at(quat_func_and_2_derivs, i),
                            gsl::at(quat_func_and_2_derivs2, i));
    }

    // phi(t) = fac1 * t^3 + fac2 * t^2 + fac3 * t + fac4;
    const double phi = fac1 * check_time * check_time * check_time +
                       fac2 * check_time * check_time + fac3 * check_time +
                       fac4;
    // omega(t) = 3*fac1 * t^2 + 2*fac2 * t + fac3
    // dtomega(t) = 6*fac1 * t + 2* fac2
    // dt2omega(t) = 6 * fac1 (constant)
    const double omega =
        3 * fac1 * check_time * check_time + 2 * fac2 * check_time + fac3;
    const double dtomega = 6 * fac1 * check_time + 2 * fac2;

    DataVector a_quat{{cos(0.5 * phi), 0.0, 0.0, sin(0.5 * phi)}};
    DataVector a_dtquat{
        {-0.5 * a_quat[3] * omega, 0.0, 0.0, 0.5 * a_quat[0] * omega}};
    DataVector a_dt2quat{{-0.5 * (a_dtquat[3] * omega + a_quat[3] * dtomega),
                          0.0, 0.0,
                          0.5 * (a_dtquat[0] * omega + a_quat[0] * dtomega)}};

    // Compare analytic solution to numerical
    Approx custom_approx = Approx::custom().epsilon(5.0e-12).scale(1.0);
    {
      INFO("  Compare quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[0], a_quat,
                                   custom_approx);
    }
    {
      INFO("  Compare derivative of quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[1], a_dtquat,
                                   custom_approx);
    }
    {
      INFO("  Compare second derivative of quaternion");
      CHECK_ITERABLE_CUSTOM_APPROX(quat_func_and_2_derivs[2], a_dt2quat,
                                   custom_approx);
    }
  }
  {
    INFO("QuaternionFunctionOfTime: No updates");
    const double initial_time = 0.0;
    const double final_time = 100.0;

    DataVector init_quat{1.0, 0.0, 0.0, 0.0};
    // We use zero because we are only concerned about checking the quaternion
    // at late times when it hasn't been updated
    DataVector three_zero{3, 0.0};
    // Construct QuaternionFunctionOfTime
    domain::FunctionsOfTime::QuaternionFunctionOfTime<3> qfot{
        initial_time, std::array<DataVector, 1>{init_quat},
        std::array<DataVector, 4>{three_zero, three_zero, three_zero,
                                  three_zero},
        std::numeric_limits<double>::infinity()};

    const std::array<DataVector, 3> quat_func_and_2_derivs =
        qfot.func_and_2_derivs(final_time);

    DataVector four_zero{4, 0.0};
    {
      INFO("  Compare quaternion");
      CHECK_ITERABLE_APPROX(quat_func_and_2_derivs[0], init_quat);
    }
    {
      INFO("  Compare derivative of quaternion");
      CHECK_ITERABLE_APPROX(quat_func_and_2_derivs[1], four_zero);
    }
    {
      INFO("  Compare second derivative of quaternion");
      CHECK_ITERABLE_APPROX(quat_func_and_2_derivs[2], four_zero);
    }
  }

  test_serialization_versioning();
}
