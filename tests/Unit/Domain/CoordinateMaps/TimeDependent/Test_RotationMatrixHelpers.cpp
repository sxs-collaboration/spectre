// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Matrix.hpp"
#include "Domain/CoordinateMaps/TimeDependent/RotationMatrixHelpers.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/QuaternionFunctionOfTime.hpp"

namespace {
void check_rotation_matrix_helpers_3d(
    const std::array<DataVector, 3>& init_angle,
    const Matrix& expected_rot_matrix, const Matrix& expected_rot_matrix_deriv,
    const double check_time) {
  using QuatFoft = domain::FunctionsOfTime::QuaternionFunctionOfTime<2>;

  const double t = 0.0;
  const std::array<DataVector, 1> init_quat{DataVector{{1.0, 0.0, 0.0, 0.0}}};

  QuatFoft quat_fot(t, init_quat, init_angle, t + 5.0);

  const Matrix rot_matrix = rotation_matrix<3>(check_time, quat_fot);
  const Matrix rot_matrix_deriv =
      rotation_matrix_deriv<3>(check_time, quat_fot);

  Approx custom_approx = Approx::custom().epsilon(5e-12).scale(1.0);
  {
    INFO("Check rotation matrix");
    CHECK_MATRIX_CUSTOM_APPROX(rot_matrix, expected_rot_matrix, custom_approx);
  }
  {
    INFO("Check rotation matrix deriv");
    CHECK_MATRIX_CUSTOM_APPROX(rot_matrix_deriv, expected_rot_matrix_deriv,
                               custom_approx);
  }
}

void check_rotation_matrix_helpers_2d(const double init_omega,
                                      const Matrix& expected_rot_matrix,
                                      const Matrix& expected_rot_matrix_deriv,
                                      const double check_time) {
  using FoT = domain::FunctionsOfTime::PiecewisePolynomial<2>;

  const double t = 0.0;
  const std::array<DataVector, 3> init_angle_func{{{0.0}, {init_omega}, {0.0}}};

  FoT function_of_time(t, init_angle_func, t + 5.0);

  const Matrix rot_matrix = rotation_matrix<2>(check_time, function_of_time);
  const Matrix rot_matrix_deriv =
      rotation_matrix_deriv<2>(check_time, function_of_time);

  {
    INFO("Check rotation matrix");
    CHECK_MATRIX_APPROX(rot_matrix, expected_rot_matrix);
  }
  {
    INFO("Check rotation matrix deriv");
    CHECK_MATRIX_APPROX(rot_matrix_deriv, expected_rot_matrix_deriv);
  }
}

SPECTRE_TEST_CASE(
    "Unit.Domain.CoordinateMaps.TimeDependent.RotationMatrixHelpers",
    "[Unit][Domain]") {
  {
    INFO("Dim = 2");
    const double omega = 2.0;
    const double check_time = 1.0;

    const std::array<DataVector, 3> init_angle{
        DataVector{0.0}, DataVector{omega}, DataVector{0.0}};

    const double theta = omega * check_time;
    const Matrix expected_rot_matrix_z{{cos(theta), -sin(theta)},
                                       {sin(theta), cos(theta)}};
    const Matrix expected_rot_matrix_deriv_z{
        {-2.0 * sin(theta), -2.0 * cos(theta)},
        {2.0 * cos(theta), -2.0 * sin(theta)}};

    check_rotation_matrix_helpers_2d(omega, expected_rot_matrix_z,
                                     expected_rot_matrix_deriv_z, check_time);
  }
  {
    INFO("Dim = 3");
    {
      INFO("Rotation about z")
      const double omega = 2.0;
      const double check_time = 1.0;

      const std::array<DataVector, 3> init_angle{DataVector{3, 0.0},
                                                 DataVector{{0.0, 0.0, omega}},
                                                 DataVector{3, 0.0}};

      const double theta = omega * check_time;
      const Matrix expected_rot_matrix_z{{cos(theta), -sin(theta), 0.0},
                                         {sin(theta), cos(theta), 0.0},
                                         {0.0, 0.0, 1.0}};
      const Matrix expected_rot_matrix_deriv_z{
          {-2.0 * sin(theta), -2.0 * cos(theta), 0.0},
          {2.0 * cos(theta), -2.0 * sin(theta), 0.0},
          {0.0, 0.0, 0.0}};

      check_rotation_matrix_helpers_3d(init_angle, expected_rot_matrix_z,
                                       expected_rot_matrix_deriv_z, check_time);
    }

    {
      INFO("Rotation about x")
      const double omega = 2.0;
      const double check_time = 1.0;

      const std::array<DataVector, 3> init_angle{DataVector{3, 0.0},
                                                 DataVector{{omega, 0.0, 0.0}},
                                                 DataVector{3, 0.0}};

      const double theta = omega * check_time;
      const Matrix expected_rot_matrix_x{{1.0, 0.0, 0.0},
                                         {0.0, cos(theta), -sin(theta)},
                                         {0.0, sin(theta), cos(theta)}};
      const Matrix expected_rot_matrix_deriv_x{
          {0.0, 0.0, 0.0},
          {0.0, -2.0 * sin(theta), -2.0 * cos(theta)},
          {0.0, 2.0 * cos(theta), -2.0 * sin(theta)}};

      check_rotation_matrix_helpers_3d(init_angle, expected_rot_matrix_x,
                                       expected_rot_matrix_deriv_x, check_time);
    }

    {
      INFO("Rotation about y")
      const double omega = 2.0;
      const double check_time = 1.0;

      const std::array<DataVector, 3> init_angle{DataVector{3, 0.0},
                                                 DataVector{{0.0, omega, 0.0}},
                                                 DataVector{3, 0.0}};

      const double theta = omega * check_time;
      const Matrix expected_rot_matrix_y{{cos(theta), 0.0, sin(theta)},
                                         {0.0, 1.0, 0.0},
                                         {-sin(theta), 0.0, cos(theta)}};
      const Matrix expected_rot_matrix_deriv_y{
          {-2.0 * sin(theta), 0.0, 2.0 * cos(theta)},
          {0.0, 0.0, 0.0},
          {-2.0 * cos(theta), 0.0, -2.0 * sin(theta)}};

      check_rotation_matrix_helpers_3d(init_angle, expected_rot_matrix_y,
                                       expected_rot_matrix_deriv_y, check_time);
    }
  }
}
}  // namespace
