// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines rotation matrices using quaternions.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/Matrix.hpp"

/// \cond
namespace domain {
namespace FunctionsOfTime {
class FunctionOfTime;
}  // namespace FunctionsOfTime
}  // namespace domain
/// \endcond

template <size_t Dim>
Matrix rotation_matrix(double t,
                       const domain::FunctionsOfTime::FunctionOfTime& fot);

template <size_t Dim>
Matrix rotation_matrix_deriv(
    double t, const domain::FunctionsOfTime::FunctionOfTime& fot);
