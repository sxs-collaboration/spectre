// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <unordered_set>
#include <vector>

#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Options/Options.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Trigger at specified numbers of slabs after the simulation start.
template <typename KnownTriggers>
class SpecifiedSlabs : public Trigger<KnownTriggers> {
 public:
  /// \cond
  SpecifiedSlabs() = default;
  explicit SpecifiedSlabs(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(SpecifiedSlabs);  // NOLINT
  /// \endcond

  struct Slabs {
    using type = std::vector<size_t>;
    static constexpr OptionString help{
      "List of slab numbers on which to trigger."};
  };

  using options = tmpl::list<Slabs>;
  static constexpr OptionString help{
    "Trigger at specified numbers of slabs after the simulation start."};

  explicit SpecifiedSlabs(const std::vector<size_t>& slabs) noexcept
      : slabs_(slabs.begin(), slabs.end()) {}

  using argument_tags = tmpl::list<Tags::TimeId>;

  bool operator()(const TimeId& time_id) const noexcept {
    return slabs_.count(time_id.slab_number) == 1;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | slabs_;
  }

 private:
  std::unordered_set<size_t> slabs_;
};

/// \cond
template <typename KnownTriggers>
PUP::able::PUP_ID SpecifiedSlabs<KnownTriggers>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers
