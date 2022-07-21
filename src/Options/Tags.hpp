// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"

namespace Options::Tags {
/// Option parser tag to retrieve the YAML source and all applied
/// overlays.  This tag can be requested without providing it as a
/// template parameter to the Parser.
struct InputSource {
  using type = std::vector<std::string>;
};
}  // namespace Options::Tags
