// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "ControlSystem/ControlErrors/Size/Info.hpp"
#include "ControlSystem/ControlErrors/Size/State.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"

namespace control_system::size::States {
class DeltaR : public State {
 public:
  DeltaR() = default;
  std::string name() const override { return "DeltaR"; }
  size_t number() const override { return 2; }
  std::unique_ptr<State> get_clone() const override;
  void update(const gsl::not_null<Info*> info,
              const StateUpdateArgs& update_args,
              const CrossingTimeInfo& crossing_time_info) const override;
  /// The return value is Q from Eq. 96 of \cite Hemberger2012jz.
  double control_error(
      const Info& info,
      const ControlErrorArgs& control_error_args) const override;

  WRAPPED_PUPable_decl_template(DeltaR); // NOLINT
  explicit DeltaR(CkMigrateMessage* const /*msg*/) {}
};
}  // namespace control_system::size::States
