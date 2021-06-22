// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"

SPECTRE_TEST_CASE("Unit.Domain.FunctionsOfTime.QuaternionFuntionOfTime",
                  "[Unit][Domain]") {
  INFO("Test quaternion function of time") {
    double t = 0.0;
    double expir_time = 0.5;
    const double omega_z = 1.3333;
    DataVector init_omega{0.0, 0.0, omega_z};
    // Construct PiecewisePolynomial
    domain::FunctionsOfTime::PiecewisePolynomial<2> pp{
        t,
        std::array<DataVector, 3>{
            {init_omega, DataVector{3, 0.0}, DataVector{3, 0.0}}},
        expir_time};

    // Update PiecewisePolynomial with 0 2nd derivative so it's constant
    const double increment = 0.5;
    for (int i = 0; i < 15; i++) {
      t += increment;
      expir_time += increment;
      pp.update(t, DataVector{3, 0.0}, expir_time);
    }
    t = 0.0;
    expir_time = 0.5;
    DataVector init_quat{1.0, 0.0, 0.0, 0.0};
    // Construct QuaternionFunctionOfTime
    domain::FunctionsOfTime::QuaternionFunctionOfTime<2> qfot{
        t, std::array<DataVector, 1>{init_quat}, make_not_null(&pp),
        expir_time};

    // Get the quaternion and 2 derivatives at a certain time. This call will
    // automatically update the stored quaternions to match the stored omegas
    double check_time = 5.398;
    const std::array<DataVector, 3> quat_func_and_2_derivs =
        qfot.func_and_2_derivs(check_time);

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
    Approx custom_approx = Approx::custom().epsilon(5e-13).scale(1.0);
    check_iterable_approx<DataVector>::apply(quat_func_and_2_derivs[0], a_quat,
                                             custom_approx);
    check_iterable_approx<DataVector>::apply(quat_func_and_2_derivs[1],
                                             a_dtquat, custom_approx);
    check_iterable_approx<DataVector>::apply(quat_func_and_2_derivs[2],
                                             a_dt2quat, custom_approx);
  }
}
