// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <pup.h>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Parallel {
template <typename Metavariables>
class GlobalCache;
}  // namespace Parallel
/// \endcond

namespace TestHelpers::DenseTriggers {
class TestTrigger : public DenseTrigger {
 public:
  /// \cond
  TestTrigger() = default;
  explicit TestTrigger(CkMigrateMessage* const msg) noexcept
      : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TestTrigger);  // NOLINT
  /// \endcond

  struct IsReady {
    using type = bool;
    constexpr static Options::String help = "IsReady";
  };

  struct IsTriggered {
    using type = bool;
    constexpr static Options::String help = "IsTriggered";
  };

  struct NextCheck {
    using type = double;
    constexpr static Options::String help = "NextCheck";
  };

  using options = tmpl::list<IsReady, IsTriggered, NextCheck>;
  constexpr static Options::String help = "help";

  TestTrigger(const bool is_ready_arg, const bool is_triggered,
              const double next_check) noexcept
      : is_ready_(is_ready_arg),
        is_triggered_(is_triggered),
        next_check_(next_check) {}

  using is_triggered_argument_tags = tmpl::list<>;
  Result is_triggered() const noexcept {
    CHECK(is_ready_);
    return {is_triggered_, next_check_};
  }

  using is_ready_argument_tags = tmpl::list<>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/) const noexcept {
    return is_ready_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept override {
    DenseTrigger::pup(p);
    p | is_ready_;
    p | is_triggered_;
    p | next_check_;
  }

 private:
  bool is_ready_;
  bool is_triggered_;
  double next_check_;
};

template <typename Label>
class BoxTrigger : public DenseTrigger {
 public:
  /// \cond
  BoxTrigger() = default;
  explicit BoxTrigger(CkMigrateMessage* const msg) noexcept
      : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BoxTrigger);  // NOLINT
  /// \endcond

  static std::string name() noexcept {
    return "BoxTrigger<" + Options::name<Label>() + ">";
  }

  using options = tmpl::list<>;
  constexpr static Options::String help = "help";

  struct IsTriggered : db::SimpleTag {
    using type = bool;
  };

  struct NextCheck : db::SimpleTag {
    using type = double;
  };

  struct IsReady : db::SimpleTag {
    using type = bool;
  };

  using is_triggered_argument_tags =
      tmpl::list<IsReady, IsTriggered, NextCheck>;
  Result is_triggered(const bool is_ready_arg, const bool is_triggered,
                      const double next_check) const noexcept {
    CHECK(is_ready_arg);
    return {is_triggered, next_check};
  }

  using is_ready_argument_tags = tmpl::list<IsReady>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  bool is_ready(Parallel::GlobalCache<Metavariables>& /*cache*/,
                const ArrayIndex& /*array_index*/,
                const Component* const /*meta*/,
                const bool is_ready_arg) const noexcept {
    return is_ready_arg;
  }
  void pup(PUP::er& p) noexcept { DenseTrigger::pup(p); }
};

/// \cond
template <typename Label>
PUP::able::PUP_ID BoxTrigger<Label>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace TestHelpers::DenseTriggers
