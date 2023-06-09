// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <ios>
#include <optional>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "Evolution/EventsAndDenseTriggers/DenseTrigger.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/MakeString.hpp"
#include "Utilities/PrettyType.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
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
  explicit TestTrigger(CkMigrateMessage* const msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(TestTrigger);  // NOLINT
  /// \endcond

  struct NotReady {
    template <typename T>
    static std::string format(const std::optional<T>& arg) {
      if (arg.has_value()) {
        return MakeString{} << std::boolalpha << arg;
      } else {
        return "NotReady";
      }
    }
  };

  struct IsTriggered {
    using type = Options::Auto<bool, NotReady>;
    constexpr static Options::String help = "IsTriggered";
  };

  struct NextCheck {
    using type = Options::Auto<double, NotReady>;
    constexpr static Options::String help = "NextCheck";
  };

  using options = tmpl::list<IsTriggered, NextCheck>;
  constexpr static Options::String help = "help";

  TestTrigger(const std::optional<bool>& is_triggered,
              const std::optional<double>& next_check)
      : is_triggered_(is_triggered), next_check_(next_check) {}

  using is_triggered_argument_tags = tmpl::list<>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/) const {
    return is_triggered_;
  }

  using next_check_time_argument_tags = tmpl::list<>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/) const {
    return next_check_;
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    DenseTrigger::pup(p);
    p | is_triggered_;
    p | next_check_;
  }

 private:
  std::optional<bool> is_triggered_;
  std::optional<double> next_check_;
};

template <typename Label>
class BoxTrigger : public DenseTrigger {
 public:
  /// \cond
  BoxTrigger() = default;
  explicit BoxTrigger(CkMigrateMessage* const msg) : DenseTrigger(msg) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BoxTrigger);  // NOLINT
  /// \endcond

  static std::string name() {
    return "BoxTrigger<" + pretty_type::name<Label>() + ">";
  }

  using options = tmpl::list<>;
  constexpr static Options::String help = "help";

  struct IsTriggered : db::SimpleTag {
    using type = std::optional<bool>;
  };

  struct NextCheck : db::SimpleTag {
    using type = std::optional<double>;
  };

  using is_triggered_argument_tags = tmpl::list<IsTriggered>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<bool> is_triggered(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      const std::optional<bool>& is_triggered) const {
    return is_triggered;
  }

  using next_check_time_argument_tags = tmpl::list<NextCheck>;
  template <typename Metavariables, typename ArrayIndex, typename Component>
  std::optional<double> next_check_time(
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const Component* /*component*/,
      const std::optional<double>& next_check) const {
    return next_check;
  }

  void pup(PUP::er& p) { DenseTrigger::pup(p); }
};

/// \cond
template <typename Label>
PUP::able::PUP_ID BoxTrigger<Label>::my_PUP_ID = 0;  // NOLINT
/// \endcond
}  // namespace TestHelpers::DenseTriggers
