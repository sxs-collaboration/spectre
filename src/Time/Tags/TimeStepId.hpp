// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Time/TimeStepId.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup TimeGroup
/// \brief Tag for ::TimeStepId for the algorithm state
struct TimeStepId : db::SimpleTag {
  using type = ::TimeStepId;
  template <typename Tag>
  using step_prefix = typename Tags::dt<Tag>;
};
}  // namespace Tags
