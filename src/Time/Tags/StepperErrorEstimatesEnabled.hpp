// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace Tags {
/// \ingroup TimeGroup
/// \brief A tag that is true if the time stepper should be run in
/// error estimation mode.
///
/// \details Estimates will not actually be produced for a variable
/// unless `StepperErrorTolerances` provides tolerances, but in
/// split-variable systems some time steppers require extra steps if
/// any variable requires error estimates.
struct StepperErrorEstimatesEnabled : db::SimpleTag {
  using type = bool;
};
}  // namespace Tags
