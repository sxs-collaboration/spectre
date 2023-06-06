// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <limits>
#include <pup.h>
#include <utility>

#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"  // IWYU pragma: keep
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace StepChoosers {

/// Suggests a constant step size.
template <typename StepChooserUse>
class Constant : public StepChooser<StepChooserUse> {
 public:
  /// \cond
  Constant() = default;
  explicit Constant(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Constant);  // NOLINT
  /// \endcond

  static constexpr Options::String help{"Suggests a constant step size."};

  explicit Constant(const double value) : value_(value) {
    ASSERT(value_ > 0., "Requested step magnitude should be positive.");
  }

  using argument_tags = tmpl::list<>;

  std::pair<double, bool> operator()(
      const double /*last_step_magnitude*/) const {
    return std::make_pair(value_, true);
  }

  bool uses_local_data() const override { return false; }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | value_; }

 private:
  double value_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename StepChooserUse>
PUP::able::PUP_ID Constant<StepChooserUse>::my_PUP_ID = 0;  // NOLINT
/// \endcond

namespace Constant_detail {
double parse_options(const Options::Option& options);
}  // namespace Constant_detail
}  // namespace StepChoosers

template <typename StepChooserUse>
struct Options::create_from_yaml<
    StepChoosers::Constant<StepChooserUse>> {
  template <typename Metavariables>
  static StepChoosers::Constant<StepChooserUse> create(
      const Options::Option& options) {
    return StepChoosers::Constant<StepChooserUse>(
        StepChoosers::Constant_detail::parse_options(options));
  }
};
