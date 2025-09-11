// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Utilities/TMPL.hpp"

/// \cond
class ImexTimeStepper;
template <typename TagsList>
class Variables;
namespace Tags {
template <typename StepperInterface>
struct TimeStepper;
}  // namespace Tags
namespace TimeSteppers {
template <typename Vars>
class History;
}  // namespace TimeSteppers
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
namespace imex::Tags {
template <typename ImplicitSector>
struct ImplicitHistory;
}  // namespace imex::Tags
/// \endcond

namespace imex {
/// Mutator to clean history objects for each sector after an IMEX substep.
template <typename System,
          typename ImplicitSectors = typename System::implicit_sectors>
struct CleanHistory;

/// \copydoc CleanHistory
template <typename System, typename... ImplicitSectors>
struct CleanHistory<System, tmpl::list<ImplicitSectors...>> {
  using return_tags =
      tmpl::list<imex::Tags::ImplicitHistory<ImplicitSectors>...>;
  using argument_tags = tmpl::list<::Tags::TimeStepper<ImexTimeStepper>>;

  static void apply(
      const gsl::not_null<TimeSteppers::History<
          Variables<typename ImplicitSectors::tensors>>*>... histories,
      const ImexTimeStepper& time_stepper);
};
}  // namespace imex
