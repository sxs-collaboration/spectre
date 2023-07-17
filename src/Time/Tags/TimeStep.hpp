// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/Time.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for step size
struct TimeStep : db::SimpleTag {
  using type = ::TimeDelta;
};
}  // namespace Tags
