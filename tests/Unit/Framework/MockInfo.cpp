// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/MockInfo.hpp"

#include <pup.h>
#include <pup_stl.h>
#include <unordered_map>
#include <utility>
#include <vector>

#include "Utilities/ErrorHandling/Error.hpp"

namespace ActionTesting {
MockInfo::MockInfo(
    const NodeId node_id, const LocalCoreId local_core_id,
    std::unordered_map<NodeId, std::unordered_map<LocalCoreId, GlobalCoreId>>
        mock_global_cores,
    std::unordered_map<GlobalCoreId, std::pair<NodeId, LocalCoreId>>
        mock_nodes_and_local_cores)
    : mock_node_(node_id.value),
      mock_local_core_(local_core_id.value),
      mock_global_cores_(std::move(mock_global_cores)),
      mock_nodes_and_local_cores_(std::move(mock_nodes_and_local_cores)) {}

MockInfo::MockInfo(const std::vector<int>& procs_per_node, const int my_node,
                   const int my_local_rank)
    : mock_node_(my_node),
      mock_local_core_(my_local_rank),
      mock_global_cores_([&procs_per_node]() {
        std::unordered_map<NodeId,
                           std::unordered_map<LocalCoreId, GlobalCoreId>>
            mock_global_cores{};
        int global_core = 0;
        for (int node = 0; std::cmp_less(node, procs_per_node.size()); ++node) {
          std::unordered_map<LocalCoreId, GlobalCoreId> global_cores;
          for (int local_core = 0;
               local_core < procs_per_node[static_cast<size_t>(node)];
               ++local_core, ++global_core) {
            global_cores.insert(
                {LocalCoreId{local_core}, GlobalCoreId{global_core}});
          }
          mock_global_cores.insert({NodeId{node}, global_cores});
        }
        return mock_global_cores;
      }()),
      mock_nodes_and_local_cores_([&procs_per_node]() {
        std::unordered_map<GlobalCoreId, std::pair<NodeId, LocalCoreId>>
            mock_nodes_and_local_cores{};
        int global_core = 0;
        for (int node = 0; std::cmp_less(node, procs_per_node.size()); ++node) {
          for (int local_core = 0;
               local_core < procs_per_node[static_cast<size_t>(node)];
               ++local_core, ++global_core) {
            mock_nodes_and_local_cores.insert(
                {GlobalCoreId{global_core},
                 std::make_pair(NodeId{node}, LocalCoreId{local_core})});
          }
        }
        return mock_nodes_and_local_cores;
      }()) {}

MockInfo::MockInfo(CkMigrateMessage* msg) : ::sys::Info(msg) {}

auto MockInfo::get_clone() const -> std::unique_ptr<::sys::Info> {
  return std::make_unique<MockInfo>(*this);
}

void MockInfo::pup(PUP::er& p) {
  p | mock_node_;
  p | mock_local_core_;
  p | mock_global_cores_;
  p | mock_nodes_and_local_cores_;
}

int MockInfo::number_of_procs() const {
  return static_cast<int>(mock_nodes_and_local_cores_.size());
}

int MockInfo::my_proc() const {
  return mock_global_cores_.at(NodeId{mock_node_})
      .at(LocalCoreId{mock_local_core_})
      .value;
}

int MockInfo::number_of_nodes() const {
  return static_cast<int>(mock_global_cores_.size());
}

int MockInfo::my_node() const { return mock_node_; }

int MockInfo::procs_on_node([[maybe_unused]] const int node_index) const {
  return static_cast<int>(mock_global_cores_.at(NodeId{node_index}).size());
}

int MockInfo::my_local_rank() const { return mock_local_core_; }

int MockInfo::first_proc_on_node([[maybe_unused]] const int node_index) const {
  return mock_global_cores_.at(NodeId{node_index}).at(LocalCoreId{0}).value;
}

int MockInfo::node_of([[maybe_unused]] const int proc_index) const {
  return mock_nodes_and_local_cores_.at(GlobalCoreId{proc_index}).first.value;
}

int MockInfo::local_rank_of([[maybe_unused]] const int proc_index) const {
  return mock_nodes_and_local_cores_.at(GlobalCoreId{proc_index}).second.value;
}

PUP::able::PUP_ID MockInfo::my_PUP_ID = 0;
}  // namespace ActionTesting
