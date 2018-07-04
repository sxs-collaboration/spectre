// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>

#include "DataStructures/DataVector.hpp"

/// \ingroup ControlSystemGroup
/// A PND (proportional to Q and N derivatives of Q) controller that computes
/// the control signal:
/// \f[ U(t) = \sum_{k=0}^{N} a_{k} \frac{d^kQ}{dt^k} \f]
/// where N is specified by the template parameter `DerivOrder`.
///
/// If an averager is used for `q_and_derivs` (as we typically do), there is an
/// induced time offset, \f$\Delta t\f$, due to the time-weighted averaging.
/// Therefore, the `q_and_derivs` that we have in hand are at some time
/// \f$t_{0}\f$. However, we desire `q_and_derivs` at the current time
/// \f$t = t_{0} + \Delta t\f$ to determine the appropriate control
/// signal. We accomplish this by Taylor expanding
/// \f$Q(t_{0} + \Delta t)\f$. The averager allows for averaging of
/// \f$Q\f$ and its derivatives OR to not average \f$Q\f$ while still averaging
/// the derivatives (the derivatives are always averaged in order to reduce
/// noise due to numerical differentiation). When they are both averaged, the
/// time offset will be identical for \f$Q\f$ and the derivatives,
/// i.e. `q_time_offset` = `deriv_time_offset`. If an unaveraged \f$Q\f$ is
/// used, then the time offset associated with \f$Q\f$ is zero,
/// i.e. `q_time_offset`=0. and the derivative time offset, `deriv_time_offset`,
/// remains non-zero.
template <size_t DerivOrder>
class Controller {
 public:
  DataVector operator()(
      const DataVector& timescales,
      const std::array<DataVector, DerivOrder + 1>& q_and_derivs,
      double q_time_offset, double deriv_time_offset) const noexcept;
};
