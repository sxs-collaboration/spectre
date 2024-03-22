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
class DeltaRDriftOutward : public State {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Controls the velocity of the excision surface to maintain a constant "
      "separation between the excision surface and the horizon surface with a "
      "small outward radial velocity. This is state 5 in SpEC."};
  DeltaRDriftOutward() = default;
  std::string name() const override { return "DeltaRDriftOutward"; }
  size_t number() const override { return 5; }
  std::unique_ptr<State> get_clone() const override;
  std::string update(const gsl::not_null<Info*> info,
                     const StateUpdateArgs& update_args,
                     const CrossingTimeInfo& crossing_time_info) const override;
  /// The return value is Q from Eq. 96 of \cite Hemberger2012jz, plus
  /// an outward velocity term.
  double control_error(
      const Info& info,
      const ControlErrorArgs& control_error_args) const override;

  WRAPPED_PUPable_decl_template(DeltaRDriftOutward);  // NOLINT
  explicit DeltaRDriftOutward(CkMigrateMessage* const /*msg*/) {}
};
}  // namespace control_system::size::States
