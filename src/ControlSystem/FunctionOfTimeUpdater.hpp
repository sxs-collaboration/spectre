// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "ControlSystem/Averager.hpp"
#include "ControlSystem/Controller.hpp"
#include "ControlSystem/PiecewisePolynomial.hpp"
#include "ControlSystem/TimescaleTuner.hpp"

// IWYU pragma: no_forward_declare FunctionsOfTime::PiecewisePolynomial

/// \cond
class DataVector;
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

/// \ingroup ControlSystemGroup
/// Responsible for updating the FunctionOfTime map parameters.
/// `modify`: updates the FunctionOfTime map parameters, changing the maps.
/// `measure`: provides updated information to the ControlSystem, without
/// modifying the maps.
///
/// `modify` computes the control signal, which relies on \f$Q\f$ and its
/// derivatives. The Averager is responsible for numerically computing the
/// derivatives, and as such, requires sufficient data (the number of points
/// being dependent on the `DerivOrder`). This data is provided to the Averager
/// through `measure`. We do not `modify` at every `measure` (even once the
/// Averager has sufficient data to compute derivatives) since `modify` requires
/// updating the maps, which could potentially hold up other processes that rely
/// on these maps.
template <size_t DerivOrder>
class FunctionOfTimeUpdater {
 public:
  FunctionOfTimeUpdater(Averager<DerivOrder>&& averager,
                        Controller<DerivOrder>&& controller,
                        TimescaleTuner&& timescale_tuner) noexcept;

  FunctionOfTimeUpdater(FunctionOfTimeUpdater&&) noexcept = default;
  FunctionOfTimeUpdater& operator=(FunctionOfTimeUpdater&&) noexcept = default;
  FunctionOfTimeUpdater(const FunctionOfTimeUpdater&) = delete;
  FunctionOfTimeUpdater& operator=(const FunctionOfTimeUpdater&) = delete;
  ~FunctionOfTimeUpdater() = default;

  /// Provides \f$Q(t)\f$ to the averager, which internally updates the averaged
  /// values of \f$Q\f$ and derivatives stored by the averager
  void measure(double time, const DataVector& raw_q) noexcept;

  /// Computes the control signal, updates the FunctionOfTime and updates the
  /// TimescaleTuner
  void modify(
      gsl::not_null<FunctionsOfTime::PiecewisePolynomial<DerivOrder>*> f_of_t,
      double time) noexcept;

 private:
  Averager<DerivOrder> averager_;
  Controller<DerivOrder> controller_;
  TimescaleTuner timescale_tuner_;
};
