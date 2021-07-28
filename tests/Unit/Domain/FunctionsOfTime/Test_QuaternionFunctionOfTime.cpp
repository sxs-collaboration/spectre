// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"
#include "Framework/TestHelpers.hpp"
#include "Utilities/Gsl.hpp"

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.QuaternionFunctionOfTime",
                  "[Unit][Domain]") {
  {
    INFO("QuaternionFunctionOfTime: Expiration time and time bounds");
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        0.0, std::array<DataVector, 1>{DataVector{4, 0.0}},
        std::array<DataVector, 3>{DataVector{3, 0.0}, DataVector{3, 0.0},
                                  DataVector{3, 0.0}},
        0.5};
    CHECK(qfot.time_bounds() == std::array<double, 2>({0.0, 0.5}));
    qfot.reset_expiration_time(0.6);
    CHECK(qfot.time_bounds() == std::array<double, 2>({0.0, 0.6}));
  }

  {
    INFO("QuaternionFunctionOfTime: Internal PiecewisePolynomial");
    DataVector init_omega{0.0, 0.0, 3.78};
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        0.0, std::array<DataVector, 1>{DataVector{{1.0, 0.0, 0.0, 0.0}}},
        std::array<DataVector, 3>{init_omega, DataVector{3, 0.0},
                                  DataVector{3, 0.0}},
        0.5};
    domain::FunctionsOfTime::PiecewisePolynomial<2> pp{
        0.0,
        std::array<DataVector, 3>{init_omega, DataVector{3, 0.0},
                                  DataVector{3, 0.0}},
        0.5};
    qfot.update(0.6, DataVector{3, 0.0}, 1.0);
    pp.update(0.6, DataVector{3, 0.0}, 1.0);

    CHECK(qfot.omega_func(0.4) == pp.func(0.4));
    CHECK(qfot.omega_func_and_deriv(0.4) == pp.func_and_deriv(0.4));
    CHECK(qfot.omega_func_and_2_derivs(0.4) == pp.func_and_2_derivs(0.4));
  }

  {
    INFO("QuaternionFunctionOfTime: pup, cloning, extra functions");
    DataVector init_omega{0.0, 0.0, 1.0};
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        0.0, std::array<DataVector, 1>{DataVector{{1.0, 0.0, 0.0, 0.0}}},
        std::array<DataVector, 3>{init_omega, DataVector{3, 0.0},
                                  DataVector{3, 0.0}},
        2.5};

    auto qfot_ptr = qfot.get_clone();
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2>
        qfot_serialized_deserialized = serialize_and_deserialize(qfot);

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
  }

  {
    INFO("QuaternionFunctionOfTime: Constant omega");
    double t = 0.0;
    double expir_time = 0.5;
    const double omega_z = 1.3333;
    DataVector init_quat{1.0, 0.0, 0.0, 0.0};
    DataVector init_omega{0.0, 0.0, omega_z};
    DataVector init_dtomega{0.0, 0.0, 0.0};
    DataVector init_dt2omega{0.0, 0.0, 0.0};

    // Construct QuaternionFunctionOfTime
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        t, std::array<DataVector, 1>{init_quat},
        std::array<DataVector, 3>{init_omega, init_dtomega, init_dt2omega},
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
    Approx custom_approx = Approx::custom().epsilon(1e-12).scale(1.0);
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
    DataVector init_omega{{0.0, 0.0, fac2}};
    DataVector init_dtomega{{0.0, 0.0, 2 * fac1}};
    DataVector init_dt2omega{3, 0.0};
    // Construct QuaternionFunctionOfTime
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        t, std::array<DataVector, 1>{init_quat},
        std::array<DataVector, 3>{init_omega, init_dtomega, init_dt2omega},
        expir_time};

    // Update internal PiecewisePolynomial with constant 2nd derivative so
    // it's linear. This will automatically update the stored quaternions as
    // well
    const double time_step = 0.5;
    for (int i = 0; i < 15; i++) {
      t += time_step;
      expir_time += time_step;
      qfot.update(t, DataVector{{0.0, 0.0, 0.0}}, expir_time);
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
    Approx custom_approx = Approx::custom().epsilon(5e-12).scale(1.0);
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
    DataVector init_omega{{0.0, 0.0, fac3}};
    DataVector init_dtomega{{0.0, 0.0, 2 * fac2}};
    DataVector init_dt2omega{{0.0, 0.0, 6 * fac1}};
    // Construct QuaternionFunctionOfTime
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        t, std::array<DataVector, 1>{init_quat},
        std::array<DataVector, 3>{init_omega, init_dtomega, init_dt2omega},
        expir_time};

    // Update PiecewisePolynomial with constant 2nd derivative so it's
    // quadratic. This will automatically update the stored quaternions as
    // well
    const double time_step = 0.5;
    for (int i = 0; i < 15; i++) {
      t += time_step;
      expir_time += time_step;
      qfot.update(t, DataVector{{0.0, 0.0, 6 * fac1}}, expir_time);
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
    Approx custom_approx = Approx::custom().epsilon(1e-12).scale(1.0);
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
}
