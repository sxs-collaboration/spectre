// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "ControlSystem/FunctionOfTimeUpdater.hpp"
#include "DataStructures/DataVector.hpp"

// IWYU pragma: no_forward_declare FunctionOfTimeUpdater

/// \cond
namespace FunctionsOfTime {
template <size_t DerivOrder>
class PiecewisePolynomial;
}  // namespace FunctionsOfTime
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace TestHelpers {
namespace ControlErrors {
// This is a simple case of a ControlError for testing purposes only
template <size_t DerivOrder>
class Translation {
 public:
  explicit Translation(DataVector target_coords) noexcept;

  Translation(Translation&&) noexcept = default;
  Translation& operator=(Translation&&) noexcept = default;
  Translation(const Translation&) = delete;
  Translation& operator=(const Translation&) = delete;
  ~Translation() = default;

  // computes the error in the translation map and provides it to the
  // FunctionOfTimeUpdater
  void operator()(
      gsl::not_null<FunctionOfTimeUpdater<DerivOrder>*> updater,
      const FunctionsOfTime::PiecewisePolynomial<DerivOrder>& f_of_t,
      double time, const DataVector& coords) noexcept;

 private:
  DataVector target_coords_;
};
}  // namespace ControlErrors
}  // namespace TestHelpers
