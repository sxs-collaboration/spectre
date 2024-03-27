// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>
#include <string>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Options/String.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::size::States {
class Initial : public State {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "A temporary state for the beginning of a simulation. This is state 0 in "
      "SpEC."};
  Initial() = default;
  std::string name() const override { return "Initial"; }
  size_t number() const override { return 0; }
  std::unique_ptr<State> get_clone() const override;
  std::string update(const gsl::not_null<Info*> info,
                     const StateUpdateArgs& update_args,
                     const CrossingTimeInfo& crossing_time_info) const override;
  double control_error(
      const Info& info,
      const ControlErrorArgs& control_error_args) const override;

  WRAPPED_PUPable_decl_template(Initial);  // NOLINT
  explicit Initial(CkMigrateMessage* const /*msg*/) {}
};
}  // namespace control_system::size::States
