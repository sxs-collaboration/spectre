// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Time/TimeSequence.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at specified numbers of slabs after the simulation start.
class Slabs : public Trigger {
 public:
  /// \cond
  Slabs() = default;
  explicit Slabs(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Slabs);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
    "Trigger at specified numbers of slabs after the simulation start."};

  explicit Slabs(std::unique_ptr<TimeSequence<uint64_t>> slabs)
      : slabs_(std::move(slabs)) {}

  using argument_tags = tmpl::list<Tags::TimeStepId>;

  bool operator()(const TimeStepId& time_id) const {
    if (not time_id.is_at_slab_boundary() or time_id.slab_number() < 0) {
      return false;
    }
    const auto unsigned_slab =
        static_cast<std::uint64_t>(time_id.slab_number());
    return slabs_->times_near(unsigned_slab)[1] == unsigned_slab;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | slabs_; }

 private:
  std::unique_ptr<TimeSequence<uint64_t>> slabs_{};
};
}  // namespace Triggers

template <>
struct Options::create_from_yaml<Triggers::Slabs> {
  template <typename Metavariables>
  static Triggers::Slabs create(const Option& options) {
    return Triggers::Slabs(
        options.parse_as<std::unique_ptr<TimeSequence<uint64_t>>,
                         Metavariables>());
  }
};
