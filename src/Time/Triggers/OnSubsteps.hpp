// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <pup_stl.h>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Options/Options.hpp"
#include "Options/ParseOptions.hpp"
#include "Options/String.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace db {
template <typename TagsList>
class DataBox;
}  // namespace db
/// \endcond

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// \ingroup TimeGroup
/// Check a trigger on substeps, as well as full steps.  Primarily for
/// debugging.
///
/// In LTS mode, only substeps of the first step in each slab will be
/// checked. Such substeps may not be aligned across the domain.
///
/// The observation value on a substep is set to the start time of the
/// step plus $10^6$ times the substep number.
class OnSubsteps : public Trigger {
 public:
  /// \cond
  OnSubsteps() = default;
  explicit OnSubsteps(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(OnSubsteps);  // NOLINT
  /// \endcond

  static constexpr Options::String help =
      "Check a trigger on substeps in addition to steps.";

  explicit OnSubsteps(std::unique_ptr<Trigger> trigger)
      : trigger_(std::move(trigger)) {}

  using argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const db::DataBox<DbTags>& box) const {
    // This trigger doesn't actually do anything.  All the special
    // logic is in the RunEventsAndTriggers action.  Just forward
    // along.
    return trigger_->is_triggered(box);
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { p | trigger_; }

 private:
  std::unique_ptr<Trigger> trigger_;
};
}  // namespace Triggers

template <>
struct Options::create_from_yaml<Triggers::OnSubsteps> {
  template <typename Metavariables>
  static Triggers::OnSubsteps create(const Option& options) {
    return Triggers::OnSubsteps(
        options.parse_as<std::unique_ptr<Trigger>, Metavariables>());
  }
};
