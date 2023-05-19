// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace control_system::size::States {
class AhSpeed : public State {
 public:
  AhSpeed() = default;
  std::string name() const override { return "AhSpeed"; }
  size_t number() const override { return 1; }
  std::unique_ptr<State> get_clone() const override;
  void update(const gsl::not_null<Info*> info,
              const StateUpdateArgs& update_args,
              const CrossingTimeInfo& crossing_time_info) const override;
  /// The return value is Q from Eq. 92 of \cite Hemberger2012jz.
  double control_error(
      const Info& info,
      const ControlErrorArgs& control_error_args) const override;

  WRAPPED_PUPable_decl_template(AhSpeed); // NOLINT
  explicit AhSpeed(CkMigrateMessage* const /*msg*/) {}
};
}  // namespace control_system::size::States
