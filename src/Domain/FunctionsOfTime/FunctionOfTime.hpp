// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <memory>
#include <pup.h>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "Parallel/CharmPupable.hpp"

namespace domain {
/// \ingroup ComputationalDomainGroup
/// \brief Contains functions of time to support the dual frame system.
namespace FunctionsOfTime {
/// \ingroup ComputationalDomainGroup
/// \brief Base class for FunctionsOfTime
class FunctionOfTime : public PUP::able {
 public:
  FunctionOfTime() = default;
  FunctionOfTime(FunctionOfTime&&) noexcept = default;
  FunctionOfTime& operator=(FunctionOfTime&&) noexcept = default;
  FunctionOfTime(const FunctionOfTime&) = default;
  FunctionOfTime& operator=(const FunctionOfTime&) = default;
  ~FunctionOfTime() override = default;

  virtual auto get_clone() const noexcept
      -> std::unique_ptr<FunctionOfTime> = 0;

  /// Returns the domain of validity of the function.
  /// For FunctionsOfTime that allow a small amount of time extrapolation,
  /// `time_bounds` tells you the bounds including the allowed extrapolation
  /// interval.
  virtual std::array<double, 2> time_bounds() const noexcept = 0;

  virtual std::array<DataVector, 1> func(double t) const noexcept = 0;
  virtual std::array<DataVector, 2> func_and_deriv(double t) const noexcept = 0;
  virtual std::array<DataVector, 3> func_and_2_derivs(double t) const
      noexcept = 0;

  WRAPPED_PUPable_abstract(FunctionOfTime);  // NOLINT
};
}  // namespace FunctionsOfTime
}  // namespace domain
