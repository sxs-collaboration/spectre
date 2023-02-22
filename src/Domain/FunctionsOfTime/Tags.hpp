// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace domain::Tags {
/// Tag to retrieve the FunctionsOfTime from the GlobalCache.
struct FunctionsOfTime : db::BaseTag {};
}  // namespace domain::Tags
