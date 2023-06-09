// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>

#include "DataStructures/DataBox/Tag.hpp"
#include "Utilities/PrettyType.hpp"

namespace mem_monitor {
/*!
 * Gives full subfile path to the dat file for the memory monitor of a
 * parallel component
 */
template <typename ParallelComponent>
std::string subfile_name() {
  return "/MemoryMonitors/" + pretty_type::name<ParallelComponent>();
}

namespace Tags {
/*!
 * \brief Tag to hold memory usage of parallel components before it is written
 * to disk.
 *
 * \details The types in the unordered_map are as follows:
 *
 * \code
 * std::unordered_map<ComponentName, std::unordered_map<%Time,
 *      std::unordered_map<Node/Proc, Memory in MB>>>
 * \endcode
 */
struct MemoryHolder : db::SimpleTag {
  using type = std::unordered_map<
      std::string, std::unordered_map<double, std::unordered_map<int, double>>>;
};
}  // namespace Tags
}  // namespace mem_monitor
