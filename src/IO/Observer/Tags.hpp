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
/// \brief The set of `ArrayComponentId`s that will contribute to each
/// `ObservationId` for reduction.
///
/// This is set during registration is only read from later on in the
/// simulation, except for possibly during migration of an `ArrayComponentId`
/// component.
struct ObservationsRegistered : db::SimpleTag {
  using type =
      std::unordered_map<ObservationKey, std::unordered_set<ArrayComponentId>>;
};

/// \brief The set of `ArrayComponentId` that have contributed to each
/// `ObservationId` for reductions
///
/// The tag is used both on the `Observer` and on the `ObserverWriter`
/// components since all we need to do is keep track of array component IDs in
/// both cases.
struct ReductionsContributed : db::SimpleTag {
  using type =
      std::unordered_map<ObservationId, std::unordered_set<ArrayComponentId>>;
};

/// \brief The set of nodes that have contributed to each `ObservationId` for
/// writing reduction data
///
/// This is used on node 0 (or whichever node has been designated as the one to
/// write the reduction files). The `unordered_set` is the node IDs that have
/// contributed so far.
struct NodeReductionsContributedForWriting : db::SimpleTag {
  using type = std::unordered_map<ObservationId, std::unordered_set<size_t>>;
};

/// \brief The set of nodes that are registered with each
/// `ObservationIdRegistrationKey` for writing reduction data
///
/// The set contains all the nodes that have been registered.
///
/// We need to keep track of this separately from the local reductions on the
/// node that are contributing so we need a separate tag. Since nodes are easily
/// indexed by an unsigned integer, we use `size_t` to keep track of them.
struct ReductionObserversRegisteredNodes : db::SimpleTag {
  using type = std::unordered_map<ObservationKey, std::set<size_t>>;
};

/// \brief Lock used when contributing reduction data.
///
/// A separate lock from the node lock of the nodegroup is used in order to
/// allow other cores to contribute volume data, write to disk, etc.
struct ReductionDataLock : db::SimpleTag {
  using type = Parallel::NodeLock;
};

/// \brief The set of `ArrayComponentId` that have contributed to each
/// `ObservationId` for volume observation
///
/// The tag is used both on the `Observer` and on the `ObserverWriter`
/// components since all we need to do is keep track of array component IDs in
/// both cases.
struct VolumesContributed : db::SimpleTag {
  using type =
      std::unordered_map<ObservationId, std::unordered_set<ArrayComponentId>>;
};

/// \brief Lock used when contributing volume data.
///
/// A separate lock from the node lock of the nodegroup is used in order to
/// allow other cores to contribute reduction data, write to disk, etc.
struct VolumeDataLock : db::SimpleTag {
  using type = Parallel::NodeLock;
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
///
/// If we have `M` `ArrayComponentIds` registered with a given `Observer`
/// for a given `ObservationId`, then we expect the `Observer` to receive
/// `M` contributions with the given `ObservationId`. We combine the reduction
/// data as we receive it, so we only need one copy for each `ObservationId`.
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
/// Groups option tags related to recording data, e.g. file names.
struct Group {
  static std::string name() { return "Observers"; }
  static constexpr OptionString help = {"Options for recording data"};
};

/// The name of the H5 file on disk to which all volume data is written.
struct VolumeFileName {
  using type = std::string;
  static constexpr OptionString help = {
      "Name of the volume data file without extension"};
  using group = Group;
};

/// The name of the H5 file on disk to which all reduction data is written.
struct ReductionFileName {
  using type = std::string;
  static constexpr OptionString help = {
      "Name of the reduction data file without extension"};
  using group = Group;
};
}  // namespace OptionTags

namespace Tags {
/// \brief The name of the HDF5 file on disk into which volume data is written.
///
/// By volume data we mean any data that is not written once across all nodes.
/// For example, data on a 2d surface written from a 3d simulation is considered
/// volume data, while an integral over the entire (or a subset of the) domain
/// is considered reduction data.
struct VolumeFileName : db::SimpleTag {
  using type = std::string;
  using option_tags = tmpl::list<::observers::OptionTags::VolumeFileName>;

  static constexpr bool pass_metavariables = false;
  static std::string create_from_options(
      const std::string& volume_file_name) noexcept {
    return volume_file_name;
  }
};

/// \brief The name of the HDF5 file on disk into which reduction data is
/// written.
///
/// By reduction data we mean any data that is written once across all nodes.
/// For example, an integral over the entire (or a subset of the) domain
/// is considered reduction data, while data on a 2d surface written from a 3d
/// simulation is considered volume data.
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
