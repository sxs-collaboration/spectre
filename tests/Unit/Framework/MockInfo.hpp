// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>
#include <unordered_map>

#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/System/Info.hpp"

namespace ActionTesting {
/// Wraps a int representing the node number.  This is so the user
/// can write things like `emplace_array_component(NodeId{3},...)`  instead of
/// `emplace_array_component(3,...)`.
struct NodeId {
  int value;
  void pup(PUP::er& p) { p | value; }  // NOLINT
};

inline bool operator==(const NodeId& lhs, const NodeId& rhs) {
  return lhs.value == rhs.value;
}

inline bool operator!=(const NodeId& lhs, const NodeId& rhs) {
  return not(lhs == rhs);
}

/// Wraps a int representing the local core number. This is so the
/// user can write things like
/// `emplace_array_component(NodeId{3},LocalCoreId{2},...)`  instead of
/// `emplace_array_component(3,2,...)`.
///
/// The local core number is unique for each core on the same node,
/// but cores on different nodes can have the same local core number.
/// For example, if there are 3 nodes with 2
/// cores each, then the cores on the first node have local core numbers
/// 0 and 1, the cores on the second node also have local core numbers
/// 0 and 1, and so on.
struct LocalCoreId {
  int value;
  void pup(PUP::er& p) { p | value; }  // NOLINT
};

inline bool operator==(const LocalCoreId& lhs, const LocalCoreId& rhs) {
  return lhs.value == rhs.value;
}

inline bool operator!=(const LocalCoreId& lhs, const LocalCoreId& rhs) {
  return not(lhs == rhs);
}

/// Wraps a int representing the global core number.
///
/// The global core number is unique for each core, even if the cores
/// are on different nodes.  For example, if there are 3 nodes with 2
/// cores each, the global core number goes from 0 through 5 to label
/// each of the 6 cores, and no two cores have the same global core
/// number.
struct GlobalCoreId {
  int value;
  void pup(PUP::er& p) { p | value; }  // NOLINT
};

inline bool operator==(const GlobalCoreId& lhs, const GlobalCoreId& rhs) {
  return lhs.value == rhs.value;
}

inline bool operator!=(const GlobalCoreId& lhs, const GlobalCoreId& rhs) {
  return not(lhs == rhs);
}
}  // namespace ActionTesting

namespace std {
template <>
struct hash<ActionTesting::NodeId> {
  int operator()(const ActionTesting::NodeId& t) const { return t.value; }
};

template <>
struct hash<ActionTesting::LocalCoreId> {
  int operator()(const ActionTesting::LocalCoreId& t) const { return t.value; }
};

template <>
struct hash<ActionTesting::GlobalCoreId> {
  int operator()(const ActionTesting::GlobalCoreId& t) const { return t.value; }
};
}  // namespace std

namespace ActionTesting {
/// Low-level system information such as number of nodes and processors for the
/// mock runtime system.
class MockInfo final : public ::sys::Info {
 public:
  MockInfo() = default;
  MockInfo(MockInfo&&) = default;
  MockInfo& operator=(MockInfo&&) = default;
  MockInfo(const MockInfo&) = default;
  MockInfo& operator=(const MockInfo&) = default;
  ~MockInfo() override = default;

  MockInfo(
      NodeId node_id, LocalCoreId local_core_id,
      std::unordered_map<NodeId, std::unordered_map<LocalCoreId, GlobalCoreId>>
          mock_global_cores,
      std::unordered_map<GlobalCoreId, std::pair<NodeId, LocalCoreId>>
          mock_nodes_and_local_cores);

  explicit MockInfo(const std::vector<int>& procs_per_node, int my_node = 0,
                    int my_local_rank = 0);

  explicit MockInfo(CkMigrateMessage* msg);

  WRAPPED_PUPable_decl_base_template(::sys::Info, MockInfo);

  auto get_clone() const -> std::unique_ptr<::sys::Info> override;

  void pup(PUP::er& p) override;

  /// \brief Number of processing elements.
  int number_of_procs() const override;

  /// \brief %Index of my processing element.
  int my_proc() const override;

  /// \brief Number of nodes.
  int number_of_nodes() const override;

  /// \brief %Index of my node.
  int my_node() const override;

  /// \brief Number of processing elements on the given node.
  int procs_on_node(int node_index) const override;

  /// \brief The local index of my processing element on my node.
  int my_local_rank() const override;

  /// \brief %Index of first processing element on the given node.
  int first_proc_on_node(int node_index) const override;

  /// \brief %Index of the node for the given processing element.
  int node_of(int proc_index) const override;

  /// \brief The local index for the given processing element on its node.
  int local_rank_of(int proc_index) const override;

 private:
  int mock_node_{0};
  int mock_local_core_{0};
  // mock_global_cores[node][local_core] is the global_core.
  std::unordered_map<NodeId, std::unordered_map<LocalCoreId, GlobalCoreId>>
      mock_global_cores_;
  // mock_nodes_and_local_cores_[global_core] is the pair node,local_core.
  std::unordered_map<GlobalCoreId, std::pair<NodeId, LocalCoreId>>
      mock_nodes_and_local_cores_;
};
}  // namespace ActionTesting
