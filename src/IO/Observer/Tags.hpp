// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "IO/Observer/ArrayComponentId.hpp"

namespace observers {
/// \ingroup ObserversGroup
/// %Tags used on the observer parallel component
namespace Tags {
/// The number of events registered with the observer.
struct NumberOfEvents : db::SimpleTag {
  using type =
      std::unordered_map<ArrayComponentId, std::unique_ptr<std::atomic_int>>;
  static std::string name() noexcept { return "NumberOfEvents"; }
};

/// All the ids of all the components registered to an observer for doing
/// reduction observations.
struct ReductionArrayComponentIds : db::SimpleTag {
  using type = std::unordered_set<ArrayComponentId>;
  static std::string name() noexcept { return "ReductionArrayComponentIds"; }
};

/// All the ids of all the components registered to an observer for doing
/// volume observations.
struct VolumeArrayComponentIds : db::SimpleTag {
  using type = std::unordered_set<ArrayComponentId>;
  static std::string name() noexcept { return "VolumeArrayComponentIds"; }
};

/// Volume tensor data to be written to disk.
struct TensorData : db::SimpleTag {
  static std::string name() noexcept { return "TensorData"; }
  using type =
      std::unordered_map<std::string,
                         std::vector<std::pair<std::string, DataVector>>>;
};
}  // namespace Tags
}  // namespace observers
