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
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeStepId;
}  // namespace Tags
/// \endcond

namespace Triggers {
template <typename TriggerRegistrars>
class Slabs;

namespace Registrars {
using Slabs = Registration::Registrar<Triggers::Slabs>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at specified numbers of slabs after the simulation start.
template <typename TriggerRegistrars = tmpl::list<Registrars::Slabs>>
class Slabs : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  Slabs() = default;
  explicit Slabs(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Slabs);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
    "Trigger at specified numbers of slabs after the simulation start."};

  explicit Slabs(std::unique_ptr<TimeSequence<uint64_t>> slabs) noexcept
      : slabs_(std::move(slabs)) {}

  using argument_tags = tmpl::list<Tags::TimeStepId>;

  bool operator()(const TimeStepId& time_id) const noexcept {
    if (not time_id.is_at_slab_boundary() or time_id.slab_number() < 0) {
      return false;
    }
    const auto unsigned_slab =
        static_cast<std::uint64_t>(time_id.slab_number());
    return slabs_->times_near(unsigned_slab)[1] == unsigned_slab;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept override {  // NOLINT
    p | slabs_;
  }

 private:
  std::unique_ptr<TimeSequence<uint64_t>> slabs_{};
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID Slabs<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers

template <typename TriggerRegistrars>
struct Options::create_from_yaml<Triggers::Slabs<TriggerRegistrars>> {
  template <typename Metavariables>
  static Triggers::Slabs<TriggerRegistrars> create(const Option& options) {
    return Triggers::Slabs<TriggerRegistrars>(
        options.parse_as<std::unique_ptr<TimeSequence<uint64_t>>>());
  }
};
