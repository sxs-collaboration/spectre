// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "Options/String.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
/// Suggests step sizes to place steps at specific times.
///
/// The suggestion provided depends on the current time, so it should
/// be applied immediately, rather than delayed several slabs.
class StepToTimes : public StepChooser<StepChooserUse::Slab> {
 public:
  /// \cond
  StepToTimes() = default;
  explicit StepToTimes(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(StepToTimes);  // NOLINT
  /// \endcond

  struct Times {
    using type = std::unique_ptr<TimeSequence<double>>;
    static constexpr Options::String help{"Times to force steps at"};
  };

  static constexpr Options::String help =
      "Suggests step sizes to place steps at specific times.\n"
      "\n"
      "The suggestion provided depends on the current time, so it should\n"
      "be applied immediately, rather than delayed several slabs.";
  using options = tmpl::list<Times>;

  explicit StepToTimes(std::unique_ptr<TimeSequence<double>> times);

  using argument_tags = tmpl::list<::Tags::Time>;

  std::pair<TimeStepRequest, bool> operator()(double now,
                                              double last_step) const;

  bool uses_local_data() const override;
  bool can_be_delayed() const override;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::unique_ptr<TimeSequence<double>> times_;
};
}  // namespace StepChoosers
