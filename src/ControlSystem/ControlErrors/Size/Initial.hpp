// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace control_system::size::States {
class Initial : public State {
 public:
  Initial() = default;
  std::string name() const override { return "Initial"; }
  size_t number() const override { return 0; }
  std::unique_ptr<State> get_clone() const override;
  void update(const gsl::not_null<Info*> info,
              const StateUpdateArgs& update_args,
              const CrossingTimeInfo& crossing_time_info) const override;
  double control_error(
      const Info& info,
      const ControlErrorArgs& control_error_args) const override;

  WRAPPED_PUPable_decl_template(Initial); // NOLINT
  explicit Initial(CkMigrateMessage* const /*msg*/) {}
};
}  // namespace control_system::size::States
