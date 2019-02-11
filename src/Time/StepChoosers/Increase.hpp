// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <limits>
#include <pup.h>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class ConstGlobalCache;
}  // namespace Parallel
namespace Tags {
struct TimeStep;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
template <typename StepChooserRegistrars>
class Increase;

namespace Registrars {
using Increase = Registration::Registrar<StepChoosers::Increase>;
}  // namespace Registrars

/// Suggests increasing the step size by a constant ratio.
template <typename StepChooserRegistrars = tmpl::list<Registrars::Increase>>
class Increase : public StepChooser<StepChooserRegistrars> {
 public:
  /// \cond
  Increase() = default;
  explicit Increase(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Increase);  // NOLINT
  /// \endcond

  struct Factor {
    using type = double;
    static constexpr OptionString help{"Factor to increase by"};
    static type lower_bound() noexcept { return 1.0; }
  };

  static constexpr OptionString help{"Suggests a constant factor increase."};
  using options = tmpl::list<Factor>;

  explicit Increase(const double factor) noexcept : factor_(factor) {}

  using argument_tags = tmpl::list<Tags::TimeStep>;

  template <typename Metavariables>
  double operator()(const TimeDelta& current_step,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/)
      const noexcept {
    return std::abs(current_step.value()) * factor_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override { p | factor_; }

 private:
  double factor_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename StepChooserRegistrars>
PUP::able::PUP_ID Increase<StepChooserRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace StepChoosers
