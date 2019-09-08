// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <atomic>
#include <converse.h>
#include <cstddef>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/TensorData.hpp"
#include "IO/Observer/ArrayComponentId.hpp"
#include "IO/Observer/ObservationId.hpp"
#include "Options/Options.hpp"
#include "Parallel/NodeLock.hpp"
#include "Parallel/Reduction.hpp"

namespace observers {
/// \ingroup ObserversGroup
/// %Tags used on the observer parallel component
namespace Tags {
/// The number of events registered with the observer.
struct NumberOfEvents : db::SimpleTag {
  using type =
      std::unordered_map<ArrayComponentId, std::unique_ptr<std::atomic_int>>;
};

/// All the ids of all the components registered to an observer for doing
/// reduction observations.
struct ReductionArrayComponentIds : db::SimpleTag {
  using type = std::unordered_set<ArrayComponentId>;
};

/// All the ids of all the components registered to an observer for doing
/// volume observations.
struct VolumeArrayComponentIds : db::SimpleTag {
  using type = std::unordered_set<ArrayComponentId>;
};

/// Volume tensor data to be written to disk.
struct TensorData : db::SimpleTag {
  using type =
      std::unordered_map<observers::ObservationId,
                         std::unordered_map<observers::ArrayComponentId,
                                            ElementVolumeData>>;
};

/// \cond
template <class... ReductionDatums>
struct ReductionDataNames;
/// \endcond

/// Reduction data to be written to disk.
template <class... ReductionDatums>
struct ReductionData : db::SimpleTag {
  using type = std::unordered_map<observers::ObservationId,
                                  Parallel::ReductionData<ReductionDatums...>>;
  using names_tag = ReductionDataNames<ReductionDatums...>;
};

/// Names of the reduction data to be written to disk.
template <class... ReductionDatums>
struct ReductionDataNames : db::SimpleTag {
  using type =
      std::unordered_map<observers::ObservationId, std::vector<std::string>>;
  using data_tag = ReductionData<ReductionDatums...>;
};

/// The number of observer components that have registered on each node
/// for volume output.
/// The key of the map is the `observation_type_hash` of the `ObservationId`.
/// The set contains all the processing elements it has registered on.
struct VolumeObserversRegistered : db::SimpleTag {
  using type = std::unordered_map<size_t, std::set<size_t>>;
};

/// The number of observer components that have contributed data at the
/// observation ids.
struct VolumeObserversContributed : db::SimpleTag {
  using type = std::unordered_map<observers::ObservationId, size_t>;
};

/// The number of observer components that have registered.
/// The key of the map is the `observation_type_hash` of the `ObservationId`.
/// The set contains all the processing elements it has registered on.
///
/// The idea is to keep track of how many processing elements have
/// called the registration function (even if some processing elements
/// call it multiple times).  This number is the number of times the
/// Observer group will call the local ObserverWriter nodegroup during
/// a reduction.
struct ReductionObserversRegistered : db::SimpleTag {
  using type = std::unordered_map<size_t, std::set<size_t>>;
};

/// The number of ObserverWriter nodegroups that have registered.
/// The key of the map is the `observation_type_hash` of the `ObservationId`.
/// The set contains all the nodes that have been registered.
struct ReductionObserversRegisteredNodes : db::SimpleTag {
  using type = std::unordered_map<size_t, std::set<size_t>>;
};

/// The number of observer components that have contributed data at the
/// observation ids.
struct ReductionObserversContributed : db::SimpleTag {
  using type = std::unordered_map<observers::ObservationId, size_t>;
};

/// Node lock used when needing to read/write to H5 files on disk.
///
/// The reason for only having one lock for all files is that we currently don't
/// require a thread-safe HDF5 installation. In the future we will need to
/// experiment with different HDF5 configurations.
struct H5FileLock : db::SimpleTag {
  using type = Parallel::NodeLock;
};
}  // namespace Tags

/// \ingroup ObserversGroup
/// Option tags related to recording data
namespace OptionTags {
/// \ingroup OptionGroupsGroup
/// Groups option tags related to recording data, e.g. file names.
struct Group {
  static std::string name() { return "Observers"; }
  static constexpr OptionString help = {"Options for recording data"};
};

/// \ingroup ObserversGroup
/// The name of the H5 file on disk to which all volume data is written.
struct VolumeFileName {
  using type = std::string;
  static constexpr OptionString help = {
      "Name of the volume data file without extension"};
  using group = Group;
};

/// \ingroup ObserversGroup
/// The name of the H5 file on disk to which all reduction data is written.
struct ReductionFileName {
  using type = std::string;
  static constexpr OptionString help = {
      "Name of the reduction data file without extension"};
  using group = Group;
};
}  // namespace OptionTags

namespace Tags {
struct VolumeFileName : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<::observers::OptionTags::VolumeFileName>;

  static constexpr bool pass_metavariables = false;
  static std::string create_from_options(
      const std::string& volume_file_name) noexcept {
    return volume_file_name;
  }
};

struct ReductionFileName : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<::observers::OptionTags::ReductionFileName>;

  static constexpr bool pass_metavariables = false;
  static std::string create_from_options(
      const std::string& reduction_file_name) noexcept {
    return reduction_file_name;
  }
};
}  // namespace Tags
}  // namespace observers
