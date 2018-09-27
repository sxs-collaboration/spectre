// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <lrtslock.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Options/Options.hpp"

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
      std::unordered_map<observers::ObservationId,
                         std::unordered_map<observers::ArrayComponentId,
                                            ExtentsAndTensorVolumeData>>;
};

/// The number of observer components that have contributed data at the
/// observation ids.
struct VolumeObserversContributed : db::SimpleTag {
  static std::string name() noexcept { return "VolumeObserversContributed"; }
  using type = std::unordered_map<observers::ObservationId, size_t>;
};

/// Node lock used when needing to lock the H5 file on disk.
struct VolumeFileLock : db::SimpleTag {
  static std::string name() noexcept { return "VolumeFileLock"; }
  using type = CmiNodeLock;
};

/// Node lock used when needing to lock the H5 file on disk.
struct ReductionFileLock : db::SimpleTag {
  static std::string name() noexcept { return "ReductionFileLock"; }
  using type = CmiNodeLock;
};
}  // namespace Tags

namespace OptionTags {
/// \ingroup ObserversGroup
/// The name of the H5 file on disk to which all volume data is written.
struct VolumeFileName {
  using type = std::string;
  static constexpr OptionString help = {
      "Name of the volume data file without extension"};
  static type default_value() noexcept { return "./VolumeData"; }
};
}  // namespace OptionTags
}  // namespace observers
