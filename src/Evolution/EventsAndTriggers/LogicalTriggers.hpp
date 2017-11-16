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
/// Always triggers.
template <typename KnownTriggers>
class Always : public Trigger<KnownTriggers> {
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
/// Negates another trigger
template <typename KnownTriggers>
class Not : public Trigger<KnownTriggers> {
 public:
  /// \cond
  Not() = default;
  explicit Not(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Not);  // NOLINT
  /// \endcond

  static constexpr OptionString help = {"Negates another trigger."};

  explicit Not(std::unique_ptr<Trigger<KnownTriggers>> negated_trigger) noexcept
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
  std::unique_ptr<Trigger<KnownTriggers>> negated_trigger_;
};

/// \ingroup EventsAndTriggersGroup
/// Short-circuiting logical AND of other triggers.
template <typename KnownTriggers>
class And : public Trigger<KnownTriggers> {
 public:
  /// \cond
  And() = default;
  explicit And(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(And);  // NOLINT
  /// \endcond

  static constexpr OptionString help = {
      "Short-circuiting logical AND of other triggers."};

  explicit And(std::vector<std::unique_ptr<Trigger<KnownTriggers>>>
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
  std::vector<std::unique_ptr<Trigger<KnownTriggers>>> combined_triggers_;
};

/// \ingroup EventsAndTriggersGroup
/// Short-circuiting logical OR of other triggers.
template <typename KnownTriggers>
class Or : public Trigger<KnownTriggers> {
 public:
  /// \cond
  Or() = default;
  explicit Or(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Or);  // NOLINT
  /// \endcond

  static constexpr OptionString help = {
      "Short-circuiting logical OR of other triggers."};

  explicit Or(std::vector<std::unique_ptr<Trigger<KnownTriggers>>>
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
  std::vector<std::unique_ptr<Trigger<KnownTriggers>>> combined_triggers_;
};

/// \cond
template <typename KnownTriggers>
PUP::able::PUP_ID Always<KnownTriggers>::my_PUP_ID = 0;  // NOLINT
template <typename KnownTriggers>
PUP::able::PUP_ID Not<KnownTriggers>::my_PUP_ID = 0;  // NOLINT
template <typename KnownTriggers>
PUP::able::PUP_ID And<KnownTriggers>::my_PUP_ID = 0;  // NOLINT
template <typename KnownTriggers>
PUP::able::PUP_ID Or<KnownTriggers>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace Triggers

template <typename KnownTriggers>
struct create_from_yaml<Triggers::Not<KnownTriggers>> {
  static Triggers::Not<KnownTriggers> create(const Option& options) {
    return Triggers::Not<KnownTriggers>(
        options.parse_as<std::unique_ptr<Trigger<KnownTriggers>>>());
  }
};

template <typename KnownTriggers>
struct create_from_yaml<Triggers::And<KnownTriggers>> {
  static Triggers::And<KnownTriggers> create(const Option& options) {
    return Triggers::And<KnownTriggers>(
        options
            .parse_as<std::vector<std::unique_ptr<Trigger<KnownTriggers>>>>());
  }
};

template <typename KnownTriggers>
struct create_from_yaml<Triggers::Or<KnownTriggers>> {
  static Triggers::Or<KnownTriggers> create(const Option& options) {
    return Triggers::Or<KnownTriggers>(
        options
            .parse_as<std::vector<std::unique_ptr<Trigger<KnownTriggers>>>>());
  }
};
