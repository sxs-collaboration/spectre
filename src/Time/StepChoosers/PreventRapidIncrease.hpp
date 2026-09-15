// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <pup.h>

#include "Options/String.hpp"
#include "Time/History.hpp"
#include "Time/StepChoosers/StepChooser.hpp"
#include "Time/TimeStepRequest.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct HistoryEvolvedVariables;
}  // namespace Tags
/// \endcond

namespace StepChoosers {
namespace PreventRapidIncrease_detail {
template <typename T>
bool limit_from_history(const TimeSteppers::ConstUntypedHistory<T>& history,
                        double last_step);
}  // namespace PreventRapidIncrease_detail
/// Limits the time step to prevent multistep integrator instabilities.
///
/// Avoids instabilities due to rapid increases in the step size by
/// preventing the step size from increasing if any step in the
/// time-stepper history increased.  If there have been recent step
/// size increases, the new size bound is the size of the most recent
/// step, otherwise no restriction is imposed.
/// @{
template <typename System,
          typename = tmpl::conditional_t<
              tt::is_a_v<tmpl::list, typename System::variables_tag>,
              typename System::variables_tag,
              tmpl::list<typename System::variables_tag>>>
class PreventRapidIncrease;

template <typename System, typename... VariablesTags>
class PreventRapidIncrease<System, tmpl::list<VariablesTags...>>
    : public StepChooser<StepChooserUse::Slab>,
      public StepChooser<StepChooserUse::LtsStep> {
 public:
  /// \cond
  PreventRapidIncrease() = default;
  explicit PreventRapidIncrease(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(PreventRapidIncrease);  // NOLINT
  /// \endcond

  static constexpr Options::String help{
      "Limits the time step to prevent multistep integrator instabilities."};
  using options = tmpl::list<>;

  using argument_tags =
      tmpl::list<::Tags::HistoryEvolvedVariables<VariablesTags>...>;

  TimeStepRequest operator()(
      const ::TimeSteppers::History<typename VariablesTags::type>&... histories,
      const double last_step) const {
    if ((... or PreventRapidIncrease_detail::limit_from_history(
                    histories.untyped(), last_step))) {
      return {.size = last_step};
    } else {
      return {};
    }
  }

  bool uses_local_data() const override { return false; }
  bool can_be_delayed() const override { return true; }
  bool must_set_step_size() const override { return true; }

  void pup(PUP::er& p) override {
    StepChooser<StepChooserUse::Slab>::pup(p);
    StepChooser<StepChooserUse::LtsStep>::pup(p);
  }
};
/// @}

/// \cond
template <typename System, typename... VariablesTags>
PUP::able::PUP_ID PreventRapidIncrease<
    System, tmpl::list<VariablesTags...>>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
}  // namespace StepChoosers
