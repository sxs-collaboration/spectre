// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>

#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "Time/TimeSequence.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class TimeStepId;
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
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
  explicit Times(CkMigrateMessage* const msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Times);  // NOLINT
  /// \endcond

  static constexpr Options::String help{"Trigger at specified times."};

  explicit Times(std::unique_ptr<TimeSequence<double>> times);

  using is_triggered_argument_tags = tmpl::list<Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      const double time) const {
    return is_triggered_impl(time);
  }

  using next_check_time_argument_tags =
      tmpl::list<Tags::TimeStepId, Tags::Time>;

  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      const TimeStepId& time_step_id, const double time) const {
    return next_check_time_impl(time_step_id, time);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override;

 private:
  std::optional<bool> is_triggered_impl(const double time) const;

  std::optional<double> next_check_time_impl(const TimeStepId& time_step_id,
                                             const double time) const;

  std::unique_ptr<TimeSequence<double>> times_{};
};
}  // namespace DenseTriggers

template <>
struct Options::create_from_yaml<DenseTriggers::Times> {
  template <typename Metavariables>
  static DenseTriggers::Times create(const Option& options) {
    return DenseTriggers::Times(
        options
            .parse_as<std::unique_ptr<TimeSequence<double>>, Metavariables>());
  }
};
