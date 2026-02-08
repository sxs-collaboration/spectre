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
class DeltaRNoDrift : public SPECTRE_CHARM_DERIVED(DeltaRNoDrift, State) {
 public:
  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Controls the velocity of the excision surface to maintain a constant "
      "separation between the excision surface and the horizon surface. "
      "This is a transition from DeltaRDriftInward to DeltaR, and otherwise "
      "acts very much like state DeltaR (e.g. it has the same control error). "
      "This is state 4 in SpEC."};
  DeltaRNoDrift() = default;
  std::string name() const override { return "DeltaRNoDrift"; }
  size_t number() const override { return 4; }
  std::unique_ptr<State> get_clone() const override;
  std::string update(gsl::not_null<Info*> info,
                     const StateUpdateArgs& update_args,
                     const CrossingTimeInfo& crossing_time_info) const override;
  /// The return value is Q from Eq. 96 of \cite Hemberger2012jz.
  double control_error(
      const Info& info,
      const ControlErrorArgs& control_error_args) const override;

  WRAPPED_PUPable_decl_template(DeltaRNoDrift);  // NOLINT
};
}  // namespace control_system::size::States
