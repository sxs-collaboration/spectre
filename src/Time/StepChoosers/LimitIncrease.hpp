// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <utility>

#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace StepChoosers {
/// Limits step increase to a constant ratio.
class LimitIncrease : public StepChooser<StepChooserUse::Slab>,
                      public StepChooser<StepChooserUse::LtsStep> {
 public:
  /// \cond
  LimitIncrease() = default;
  explicit LimitIncrease(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(LimitIncrease);  // NOLINT
  /// \endcond

  struct Factor {
    using type = double;
    static constexpr Options::String help{"Factor to allow increase by"};
    static type lower_bound() { return 1.0; }
  };

  static constexpr Options::String help{
      "Limits step increase to a constant ratio."};
  using options = tmpl::list<Factor>;

  explicit LimitIncrease(double factor);

  using argument_tags = tmpl::list<>;

  std::pair<TimeStepRequest, bool> operator()(double last_step) const;

  bool uses_local_data() const override;
  bool can_be_delayed() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  double factor_ = std::numeric_limits<double>::signaling_NaN();
};
}  // namespace StepChoosers
