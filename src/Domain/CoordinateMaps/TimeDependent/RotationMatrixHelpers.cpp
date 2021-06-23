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

Matrix get_rotation_matrix(
    const double t,
    const domain::FunctionsOfTime::FunctionOfTime& fot) noexcept {
  std::array<DataVector, 1> quat = fot.func(t);
  Matrix rot_matrix{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  add_bilinear_term(make_not_null(&rot_matrix), quat[0], quat[0]);

  return rot_matrix;
}

Matrix get_rotation_matrix_deriv(
    const double t,
    const domain::FunctionsOfTime::FunctionOfTime& fot) noexcept {
  std::array<DataVector, 2> quat_and_deriv = fot.func_and_deriv(t);
  Matrix rot_matrix_deriv{{0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}, {0.0, 0.0, 0.0}};

  add_bilinear_term(make_not_null(&rot_matrix_deriv), quat_and_deriv[0],
                    quat_and_deriv[1]);
  add_bilinear_term(make_not_null(&rot_matrix_deriv), quat_and_deriv[1],
                    quat_and_deriv[0]);

  return rot_matrix_deriv;
}
