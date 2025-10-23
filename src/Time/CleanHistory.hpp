// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
class TimeStepper;
namespace Tags {
template <typename StepperInterface>
struct TimeStepper;
template <typename Tag>
struct HistoryEvolvedVariables;
}  // namespace Tags
namespace TimeSteppers {
template <typename Vars>
class History;
}  // namespace TimeSteppers
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

/// \ingroup TimeGroup
/// \brief Clean time stepper history after a substep
/// @{
template <typename System,
          template <typename> typename CacheTagPrefix = std::type_identity_t,
          typename = tmpl::conditional_t<
              tt::is_a_v<tmpl::list, typename System::variables_tag>,
              typename System::variables_tag,
              tmpl::list<typename System::variables_tag>>>
struct CleanHistory;

template <typename System, template <typename> typename CacheTagPrefix,
          typename... VariablesTags>
struct CleanHistory<System, CacheTagPrefix, tmpl::list<VariablesTags...>> {
  using return_tags =
      tmpl::list<Tags::HistoryEvolvedVariables<VariablesTags>...>;
  using argument_tags =
      tmpl::list<CacheTagPrefix<Tags::TimeStepper<TimeStepper>>>;

  static void apply(
      const gsl::not_null<
          TimeSteppers::History<typename VariablesTags::type>*>... histories,
      const TimeStepper& time_stepper);
};
/// @}
