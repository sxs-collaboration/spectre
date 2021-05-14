// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/TimeSequence.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeStepId;
namespace Tags {
struct Time;
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace DenseTriggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at specified times.
class Times : public DenseTrigger {
 public:
  /// \cond
  Times() = default;
  explicit Times(CkMigrateMessage* const msg) noexcept : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Times);  // NOLINT
  /// \endcond

  static constexpr Options::String help{"Trigger at specified times."};

  explicit Times(std::unique_ptr<TimeSequence<double>> times) noexcept;

  using is_triggered_argument_tags = tmpl::list<Tags::TimeStepId, Tags::Time>;

  Result is_triggered(const TimeStepId& time_step_id,
                      const double time) const noexcept;

  using is_ready_argument_tags = tmpl::list<>;

  static bool is_ready() noexcept;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override;

 private:
  std::unique_ptr<TimeSequence<double>> times_{};
};
}  // namespace DenseTriggers

template <>
struct Options::create_from_yaml<DenseTriggers::Times> {
  template <typename Metavariables>
  static DenseTriggers::Times create(const Option& options) {
    return DenseTriggers::Times(
        options.parse_as<std::unique_ptr<TimeSequence<double>>>());
  }
};
