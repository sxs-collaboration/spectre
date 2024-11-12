// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <memory>
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
struct StepNumberWithinSlab;
}  // namespace Tags
/// \endcond

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at specified numbers of slabs after the simulation start.
class StepsWithinSlab : public Trigger {
 public:
  /// \cond
  StepsWithinSlab() = default;
  explicit StepsWithinSlab(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(StepsWithinSlab);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
    "Trigger at specified numbers of steps within each slab."};

  explicit StepsWithinSlab(
      std::unique_ptr<TimeSequence<uint64_t>> steps_within_slab)
      : steps_within_slab_(std::move(steps_within_slab)) {}

  using argument_tags = tmpl::list<Tags::StepNumberWithinSlab>;

  bool operator()(const uint64_t step_within_slab) const {
    return steps_within_slab_->times_near(step_within_slab)[1] ==
           step_within_slab;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | steps_within_slab_; }

 private:
  std::unique_ptr<TimeSequence<uint64_t>> steps_within_slab_{};
};
}  // namespace Triggers

template <>
struct Options::create_from_yaml<Triggers::StepsWithinSlab> {
  template <typename Metavariables>
  static Triggers::StepsWithinSlab create(const Option& options) {
    return Triggers::StepsWithinSlab(
        options.parse_as<std::unique_ptr<TimeSequence<uint64_t>>,
                         Metavariables>());
  }
};
