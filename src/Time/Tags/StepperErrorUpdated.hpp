// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace Tags {
/// \ingroup TimeGroup
/// \brief Tag indicating whether the stepper error has been updated on the
/// current step
struct StepperErrorUpdated : db::SimpleTag {
  using type = bool;
};
}  // namespace Tags
