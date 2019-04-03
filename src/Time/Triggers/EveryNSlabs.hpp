// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstdint>
#include <pup.h>

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
class EveryNSlabs;

namespace Registrars {
using EveryNSlabs = Registration::Registrar<Triggers::EveryNSlabs>;
}  // namespace Registrars

/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger every N time slabs after a given offset.
template <typename TriggerRegistrars = tmpl::list<Registrars::EveryNSlabs>>
class EveryNSlabs : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  EveryNSlabs() = default;
  explicit EveryNSlabs(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(EveryNSlabs);  // NOLINT
  /// \endcond

  struct N {
    using type = uint64_t;
    static constexpr OptionString help{"How frequently to trigger."};
    static type lower_bound() noexcept { return 1; }
  };
  struct Offset {
    using type = uint64_t;
    static constexpr OptionString help{"First slab to trigger on."};
    static type default_value() noexcept { return 0; }
  };

  using options = tmpl::list<N, Offset>;
  static constexpr OptionString help{
    "Trigger every N time slabs after a given offset."};

  EveryNSlabs(const uint64_t interval, const uint64_t offset) noexcept
      : interval_(interval), offset_(offset) {}

  using argument_tags = tmpl::list<Tags::TimeId>;

  bool operator()(const TimeId& time_id) const noexcept {
    if (not time_id.is_at_slab_boundary()) {
      return false;
    }
    const auto slab_number = static_cast<uint64_t>(time_id.slab_number());
    return slab_number >= offset_ and (slab_number - offset_) % interval_ == 0;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | interval_;
    p | offset_;
  }

 private:
  uint64_t interval_{0};
  uint64_t offset_{0};
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID EveryNSlabs<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers
