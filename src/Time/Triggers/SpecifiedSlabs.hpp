// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <pup.h>
#include <unordered_set>
#include <vector>

#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Registration.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
struct TimeId;
}  // namespace Tags
/// \endcond

namespace Triggers {
template <typename TriggerRegistrars>
class SpecifiedSlabs;

namespace Registrars {
using SpecifiedSlabs = Registration::Registrar<Triggers::SpecifiedSlabs>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at specified numbers of slabs after the simulation start.
template <typename TriggerRegistrars = tmpl::list<Registrars::SpecifiedSlabs>>
class SpecifiedSlabs : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  SpecifiedSlabs() = default;
  explicit SpecifiedSlabs(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SpecifiedSlabs);  // NOLINT
  /// \endcond

  struct Slabs {
    // No observing negative slabs.  That's the initialization phase.
    using type = std::vector<uint64_t>;
    static constexpr OptionString help{
      "List of slab numbers on which to trigger."};
  };

  using options = tmpl::list<Slabs>;
  static constexpr OptionString help{
    "Trigger at specified numbers of slabs after the simulation start."};

  explicit SpecifiedSlabs(const std::vector<uint64_t>& slabs) noexcept
      : slabs_(slabs.begin(), slabs.end()) {}

  using argument_tags = tmpl::list<Tags::TimeId>;

  bool operator()(const TimeId& time_id) const noexcept {
    if (not time_id.is_at_slab_boundary()) {
      return false;
    }
    return slabs_.count(static_cast<uint64_t>(time_id.slab_number())) == 1;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | slabs_;
  }

 private:
  std::unordered_set<uint64_t> slabs_;
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID SpecifiedSlabs<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers
