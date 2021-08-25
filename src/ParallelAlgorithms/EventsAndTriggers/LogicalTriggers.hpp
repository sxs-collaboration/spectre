// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup_stl.h>
#include <vector>

#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "ParallelAlgorithms/EventsAndTriggers/Trigger.hpp"
#include "Utilities/TMPL.hpp"

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// Always triggers.
class Always : public Trigger {
 public:
  /// \cond
  explicit Always(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Always);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr Options::String help = {"Always trigger."};

  Always() = default;

  using argument_tags = tmpl::list<>;

  bool operator()() const noexcept { return true; }
};

/// \ingroup EventsAndTriggersGroup
/// Negates another trigger.
class Not : public Trigger {
 public:
  /// \cond
  Not() = default;
  explicit Not(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Not);  // NOLINT
  /// \endcond

  static constexpr Options::String help = {"Negates another trigger."};

  explicit Not(std::unique_ptr<Trigger> negated_trigger) noexcept
      : negated_trigger_(std::move(negated_trigger)) {}

  using argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const db::DataBox<DbTags>& box) const noexcept {
    return not negated_trigger_->is_triggered(box);
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | negated_trigger_;
  }

 private:
  std::unique_ptr<Trigger> negated_trigger_;
};

/// \ingroup EventsAndTriggersGroup
/// Short-circuiting logical AND of other triggers.
class And : public Trigger {
 public:
  /// \cond
  And() = default;
  explicit And(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(And);  // NOLINT
  /// \endcond

  static constexpr Options::String help = {
      "Short-circuiting logical AND of other triggers."};

  explicit And(std::vector<std::unique_ptr<Trigger>> combined_triggers) noexcept
      : combined_triggers_(std::move(combined_triggers)) {}

  using argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const db::DataBox<DbTags>& box) const noexcept {
    for (auto& trigger : combined_triggers_) {
      if (not trigger->is_triggered(box)) {
        return false;
      }
    }
    return true;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | combined_triggers_;
  }

 private:
  std::vector<std::unique_ptr<Trigger>> combined_triggers_;
};

/// \ingroup EventsAndTriggersGroup
/// Short-circuiting logical OR of other triggers.
class Or : public Trigger {
 public:
  /// \cond
  Or() = default;
  explicit Or(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Or);  // NOLINT
  /// \endcond

  static constexpr Options::String help = {
      "Short-circuiting logical OR of other triggers."};

  explicit Or(std::vector<std::unique_ptr<Trigger>> combined_triggers) noexcept
      : combined_triggers_(std::move(combined_triggers)) {}

  using argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const db::DataBox<DbTags>& box) const noexcept {
    for (auto& trigger : combined_triggers_) {
      if (trigger->is_triggered(box)) {
        return true;
      }
    }
    return false;
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | combined_triggers_;
  }

 private:
  std::vector<std::unique_ptr<Trigger>> combined_triggers_;
};

/// A list of all the logical triggers.
using logical_triggers = tmpl::list<Always, And, Not, Or>;
}  // namespace Triggers

template <>
struct Options::create_from_yaml<Triggers::Not> {
  template <typename Metavariables>
  static Triggers::Not create(const Options::Option& options) {
    return Triggers::Not(
        options.parse_as<std::unique_ptr<Trigger>, Metavariables>());
  }
};

template <>
struct Options::create_from_yaml<Triggers::And> {
  template <typename Metavariables>
  static Triggers::And create(const Options::Option& options) {
    return Triggers::And(
        options
            .parse_as<std::vector<std::unique_ptr<Trigger>>, Metavariables>());
  }
};

template <>
struct Options::create_from_yaml<Triggers::Or> {
  template <typename Metavariables>
  static Triggers::Or create(const Options::Option& options) {
    return Triggers::Or(
        options
            .parse_as<std::vector<std::unique_ptr<Trigger>>, Metavariables>());
  }
};
