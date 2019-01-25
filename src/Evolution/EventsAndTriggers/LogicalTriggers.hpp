// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>

#include "Evolution/EventsAndTriggers/Trigger.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace Triggers {
/// \ingroup EventsAndTriggersGroup
/// Always triggers.  This trigger is automatically registered.
template <typename TriggerRegistrars = tmpl::list<>>
class Always : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  explicit Always(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Always);  // NOLINT
  /// \endcond

  using options = tmpl::list<>;
  static constexpr OptionString help = {"Always trigger."};

  Always() = default;

  using argument_tags = tmpl::list<>;

  bool operator()() const noexcept { return true; }
};

/// \ingroup EventsAndTriggersGroup
/// Negates another trigger.  This trigger is automatically registered.
template <typename TriggerRegistrars>
class Not : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  Not() = default;
  explicit Not(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Not);  // NOLINT
  /// \endcond

  static constexpr OptionString help = {"Negates another trigger."};

  explicit Not(
      std::unique_ptr<Trigger<TriggerRegistrars>> negated_trigger) noexcept
      : negated_trigger_(std::move(negated_trigger)) {}

  using argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const db::DataBox<DbTags>& box) noexcept {
    return not negated_trigger_->is_triggered(box);
  }

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p) noexcept {  // NOLINT
    p | negated_trigger_;
  }

 private:
  std::unique_ptr<Trigger<TriggerRegistrars>> negated_trigger_;
};

/// \ingroup EventsAndTriggersGroup
/// Short-circuiting logical AND of other triggers.  This trigger is
/// automatically registered.
template <typename TriggerRegistrars>
class And : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  And() = default;
  explicit And(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(And);  // NOLINT
  /// \endcond

  static constexpr OptionString help = {
      "Short-circuiting logical AND of other triggers."};

  explicit And(std::vector<std::unique_ptr<Trigger<TriggerRegistrars>>>
                   combined_triggers) noexcept
      : combined_triggers_(std::move(combined_triggers)) {}

  using argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const db::DataBox<DbTags>& box) noexcept {
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
  std::vector<std::unique_ptr<Trigger<TriggerRegistrars>>> combined_triggers_;
};

/// \ingroup EventsAndTriggersGroup
/// Short-circuiting logical OR of other triggers.  This trigger is
/// automatically registered.
template <typename TriggerRegistrars>
class Or : public Trigger<TriggerRegistrars> {
 public:
  /// \cond
  Or() = default;
  explicit Or(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Or);  // NOLINT
  /// \endcond

  static constexpr OptionString help = {
      "Short-circuiting logical OR of other triggers."};

  explicit Or(std::vector<std::unique_ptr<Trigger<TriggerRegistrars>>>
                  combined_triggers) noexcept
      : combined_triggers_(std::move(combined_triggers)) {}

  using argument_tags = tmpl::list<Tags::DataBox>;

  template <typename DbTags>
  bool operator()(const db::DataBox<DbTags>& box) noexcept {
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
  std::vector<std::unique_ptr<Trigger<TriggerRegistrars>>> combined_triggers_;
};

/// \cond
template <typename TriggerRegistrars>
PUP::able::PUP_ID Always<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
template <typename TriggerRegistrars>
PUP::able::PUP_ID Not<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
template <typename TriggerRegistrars>
PUP::able::PUP_ID And<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
template <typename TriggerRegistrars>
PUP::able::PUP_ID Or<TriggerRegistrars>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers

template <typename TriggerRegistrars>
struct create_from_yaml<Triggers::Not<TriggerRegistrars>> {
  template <typename Metavariables>
  static Triggers::Not<TriggerRegistrars> create(const Option& options) {
    return Triggers::Not<TriggerRegistrars>(
        options.parse_as<std::unique_ptr<Trigger<TriggerRegistrars>>>());
  }
};

template <typename TriggerRegistrars>
struct create_from_yaml<Triggers::And<TriggerRegistrars>> {
  template <typename Metavariables>
  static Triggers::And<TriggerRegistrars> create(const Option& options) {
    return Triggers::And<TriggerRegistrars>(
        options.parse_as<
            std::vector<std::unique_ptr<Trigger<TriggerRegistrars>>>>());
  }
};

template <typename TriggerRegistrars>
struct create_from_yaml<Triggers::Or<TriggerRegistrars>> {
  template <typename Metavariables>
  static Triggers::Or<TriggerRegistrars> create(const Option& options) {
    return Triggers::Or<TriggerRegistrars>(
        options.parse_as<
            std::vector<std::unique_ptr<Trigger<TriggerRegistrars>>>>());
  }
};
