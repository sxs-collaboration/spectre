// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <optional>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/TimeSequence.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct Time;
}  // namespace Tags
/// \endcond

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at particular times.
///
/// \warning This trigger will only fire if it is actually checked at
/// the times specified.  The StepToTimes StepChooser can be useful
/// for this.
class Times : public Trigger {
 public:
  /// \cond
  Times() = default;
  explicit Times(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Times);  // NOLINT
  /// \endcond

  static constexpr Options::String help{"Trigger at particular times."};

  explicit Times(std::unique_ptr<TimeSequence<double>> times)
      : times_(std::move(times)) {}

  using argument_tags = tmpl::list<Tags::Time>;

  bool operator()(const double now) const {
    return times_->times_near(now)[1] == std::optional(now);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | times_; }

 private:
  std::unique_ptr<TimeSequence<double>> times_;
};
}  // namespace Triggers

template <>
struct Options::create_from_yaml<Triggers::Times> {
  template <typename Metavariables>
  static Triggers::Times create(const Option& options) {
    return Triggers::Times(
        options
            .parse_as<std::unique_ptr<TimeSequence<double>>, Metavariables>());
  }
};
