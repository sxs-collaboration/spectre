// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "Time/AdaptiveSteppingDiagnostics.hpp"

namespace Tags {
struct AdaptiveSteppingDiagnostics : db::SimpleTag {
  using type = ::AdaptiveSteppingDiagnostics;
};
}  // namespace Tags
