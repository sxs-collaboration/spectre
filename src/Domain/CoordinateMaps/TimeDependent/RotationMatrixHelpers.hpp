// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines rotation matrices using quaternions.

#pragma once

#include <array>

#include "DataStructures/Matrix.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
/// \endcond

Matrix get_rotation_matrix(double t,
                           const domain::FunctionsOfTime::FunctionOfTime& fot);

Matrix get_rotation_matrix_deriv(
    double t, const domain::FunctionsOfTime::FunctionOfTime& fot);
