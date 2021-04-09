// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <limits>
#include <pup.h>
#include <utility>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace StepChoosers {
/// Suggests increasing the step size by a constant ratio.
template <typename StepChooserUse>
class Increase : public StepChooser<StepChooserUse> {
 public:
  /// \cond
  Increase() = default;
  explicit Increase(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Increase);  // NOLINT
  /// \endcond

  struct Factor {
    using type = double;
    static constexpr Options::String help{"Factor to increase by"};
    static type lower_bound() noexcept { return 1.0; }
  };

  static constexpr Options::String help{"Suggests a constant factor increase."};
  using options = tmpl::list<Factor>;

  explicit Increase(const double factor) noexcept : factor_(factor) {}

  using argument_tags = tmpl::list<>;
  using return_tags = tmpl::list<>;

  template <typename Metavariables>
  std::pair<double, bool> operator()(
      const double last_step_magnitude,
      const Parallel::GlobalCache<Metavariables>& /*cache*/) const noexcept {
    return std::make_pair(last_step_magnitude * factor_, true);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override { p | factor_; }

 private:
  double factor_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename StepChooserUse>
PUP::able::PUP_ID Increase<StepChooserUse>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
