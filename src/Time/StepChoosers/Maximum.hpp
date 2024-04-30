// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <limits>
#include <pup.h>
#include <utility>

#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace StepChoosers {

/// Sets a constant step size limit.
template <typename StepChooserUse>
class Maximum : public StepChooser<StepChooserUse> {
 public:
  /// \cond
  Maximum() = default;
  explicit Maximum(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Maximum);  // NOLINT
  /// \endcond

  static constexpr Options::String help{"Sets a constant step size limit."};

  explicit Maximum(const double value) : value_(value) {}

  using argument_tags = tmpl::list<>;

  std::pair<TimeStepRequest, bool> operator()(const double last_step) const {
    return {{.size = std::copysign(value_, last_step)}, true};
  }

  bool uses_local_data() const override { return false; }
  bool can_be_delayed() const override { return true; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | value_; }

 private:
  double value_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename StepChooserUse>
PUP::able::PUP_ID Maximum<StepChooserUse>::my_PUP_ID = 0;  // NOLINT
/// \endcond

namespace Maximum_detail {
double parse_options(const Options::Option& options);
}  // namespace Maximum_detail
}  // namespace StepChoosers

template <typename StepChooserUse>
struct Options::create_from_yaml<StepChoosers::Maximum<StepChooserUse>> {
  template <typename Metavariables>
  static StepChoosers::Maximum<StepChooserUse> create(
      const Options::Option& options) {
    return StepChoosers::Maximum<StepChooserUse>(
        StepChoosers::Maximum_detail::parse_options(options));
  }
};
