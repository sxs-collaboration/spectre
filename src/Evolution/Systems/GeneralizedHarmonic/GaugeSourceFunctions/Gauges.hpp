// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"

namespace gh {
/// \brief Gauge conditions for generalized harmonic evolution systems.
namespace gauges {
/// \brief Base class for GH gauge conditions.
///
/// Derived class must have a `void gauge_and_spacetime_derivative` function
/// that takes as `not_null` arguments \f$H_a\f$ and \f$\partial_b H_a\f$.
/// Additional arguments can be added that are needed to compute the gauge
/// condition. The `gh::gauges::dispatch()` function must also
/// be updated to correctly detect and forward to the gauge condition. The
/// header file must also be included in `Factory.hpp` and the gauge condition
/// added to the `all_gauges` type alias in `Factory.hpp`.
class GaugeCondition : public PUP::able {
 public:
  GaugeCondition() = default;
  GaugeCondition(const GaugeCondition&) = default;
  GaugeCondition& operator=(const GaugeCondition&) = default;
  GaugeCondition(GaugeCondition&&) = default;
  GaugeCondition& operator=(GaugeCondition&&) = default;
  ~GaugeCondition() override = default;

  explicit GaugeCondition(CkMigrateMessage* msg);

  WRAPPED_PUPable_abstract(GaugeCondition);  // NOLINT

  virtual std::unique_ptr<GaugeCondition> get_clone() const = 0;
};
}  // namespace gauges
}  // namespace gh
