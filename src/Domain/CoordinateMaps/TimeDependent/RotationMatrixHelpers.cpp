// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/TimeDependent/RotationMatrixHelpers.hpp"

#include "DataStructures/DataVector.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void add_bilinear_term(const gsl::not_null<Matrix*> rot_matrix,
                       const DataVector& q1, const DataVector& q2,
                       const double coef = 1.0) {
  (*rot_matrix)(0, 0) +=
      coef * (q1[0] * q2[0] + q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]);
  (*rot_matrix)(1, 1) +=
      coef * (q1[0] * q2[0] + q1[2] * q2[2] - q1[1] * q2[1] - q1[3] * q2[3]);
  (*rot_matrix)(2, 2) +=
      coef * (q1[0] * q2[0] + q1[3] * q2[3] - q1[1] * q2[1] - q1[2] * q2[2]);

  (*rot_matrix)(0, 1) += coef * (2.0 * (q1[1] * q2[2] - q1[0] * q2[3]));
  (*rot_matrix)(0, 2) += coef * (2.0 * (q1[0] * q2[2] + q1[1] * q2[3]));
  (*rot_matrix)(1, 0) += coef * (2.0 * (q1[1] * q2[2] + q1[0] * q2[3]));
  (*rot_matrix)(1, 2) += coef * (2.0 * (q1[2] * q2[3] - q1[0] * q2[1]));
  (*rot_matrix)(2, 0) += coef * (2.0 * (q1[1] * q2[3] - q1[0] * q2[2]));
  (*rot_matrix)(2, 1) += coef * (2.0 * (q1[0] * q2[1] + q1[2] * q2[3]));
}
}  // namespace

template <size_t Dim>
Matrix rotation_matrix(const double t,
                       const domain::FunctionsOfTime::FunctionOfTime& fot) {
  static_assert(
      Dim == 2 or Dim == 3,
      "Rotation matrices can only be constructed in 2 or 3 dimensions.");
  Matrix rotation_matrix{Dim, Dim, 0.0};

  if constexpr (Dim == 2) {
    const double rotation_angle = fot.func(t)[0][0];
    rotation_matrix(0, 0) = cos(rotation_angle);
    rotation_matrix(0, 1) = -sin(rotation_angle);
    rotation_matrix(1, 0) = sin(rotation_angle);
    rotation_matrix(1, 1) = cos(rotation_angle);
  } else {
    const DataVector quat = fot.func(t)[0];
    add_bilinear_term(make_not_null(&rotation_matrix), quat, quat);
  }

  return rotation_matrix;
}

template <size_t Dim>
Matrix rotation_matrix_deriv(
    const double t, const domain::FunctionsOfTime::FunctionOfTime& fot) {
  static_assert(
      Dim == 2 or Dim == 3,
      "Rotation matrices can only be constructed in 2 or 3 dimensions.");
  Matrix rotation_matrix_deriv{Dim, Dim, 0.0};

  if constexpr (Dim == 2) {
    const std::array<DataVector, 2> angle_and_deriv = fot.func_and_deriv(t);
    const double rotation_angle = angle_and_deriv[0][0];
    const double rotation_angular_velocity = angle_and_deriv[1][0];
    rotation_matrix_deriv(0, 0) =
        -rotation_angular_velocity * sin(rotation_angle);
    rotation_matrix_deriv(0, 1) =
        -rotation_angular_velocity * cos(rotation_angle);
    rotation_matrix_deriv(1, 0) =
        rotation_angular_velocity * cos(rotation_angle);
    rotation_matrix_deriv(1, 1) =
        -rotation_angular_velocity * sin(rotation_angle);
  } else {
    const std::array<DataVector, 2> quat_and_deriv = fot.func_and_deriv(t);
    add_bilinear_term(make_not_null(&rotation_matrix_deriv), quat_and_deriv[0],
                      quat_and_deriv[1]);
    add_bilinear_term(make_not_null(&rotation_matrix_deriv), quat_and_deriv[1],
                      quat_and_deriv[0]);
  }

  return rotation_matrix_deriv;
}

template Matrix rotation_matrix<2>(
    const double, const domain::FunctionsOfTime::FunctionOfTime&);
template Matrix rotation_matrix<3>(
    const double, const domain::FunctionsOfTime::FunctionOfTime&);
template Matrix rotation_matrix_deriv<2>(
    const double, const domain::FunctionsOfTime::FunctionOfTime&);
template Matrix rotation_matrix_deriv<3>(
    const double, const domain::FunctionsOfTime::FunctionOfTime&);
