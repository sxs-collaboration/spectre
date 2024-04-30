// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <limits>
#include <pup.h>
#include <utility>

#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace StepChoosers {
/// Suggests increasing the step size by a constant ratio.
template <typename StepChooserUse>
class Increase : public StepChooser<StepChooserUse> {
 public:
  /// \cond
  Increase() = default;
  explicit Increase(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Increase);  // NOLINT
  /// \endcond

  struct Factor {
    using type = double;
    static constexpr Options::String help{"Factor to increase by"};
    static type lower_bound() { return 1.0; }
  };

  static constexpr Options::String help{"Suggests a constant factor increase."};
  using options = tmpl::list<Factor>;

  explicit Increase(const double factor) : factor_(factor) {}

  using argument_tags = tmpl::list<>;

  std::pair<TimeStepRequest, bool> operator()(const double last_step) const {
    return {{.size_goal = last_step * factor_}, true};
  }

  bool uses_local_data() const override { return false; }
  bool can_be_delayed() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | factor_; }

 private:
  double factor_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename StepChooserUse>
PUP::able::PUP_ID Increase<StepChooserUse>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
