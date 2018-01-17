// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <vector>

#include "DataStructures/DataVector.hpp"

/// \ingroup ControlSystemGroup
/// Base class for FunctionsOfTime
class FunctionOfTime {
 public:
  FunctionOfTime() = default;
  FunctionOfTime(FunctionOfTime&&) noexcept = default;
  FunctionOfTime& operator=(FunctionOfTime&&) noexcept = default;
  FunctionOfTime(const FunctionOfTime&) = delete;
  FunctionOfTime& operator=(const FunctionOfTime&) = delete;
  virtual ~FunctionOfTime() = default;

  virtual std::array<double, 2> time_bounds() const noexcept = 0;

  virtual std::array<DataVector, 1> func(double t) const noexcept = 0;
  virtual std::array<DataVector, 2> func_and_deriv(double t) const noexcept = 0;
  virtual std::array<DataVector, 3> func_and_2_derivs(double t) const
      noexcept = 0;
};
