// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"

namespace sys {
/// \ingroup UtilitiesGroup
/// \brief Low-level system information such as number of nodes and processors
class Info : public PUP::able {
 public:
  Info() = default;
  Info(Info&&) = default;
  Info& operator=(Info&&) = default;
  Info(const Info&) = default;
  Info& operator=(const Info&) = default;
  ~Info() override = default;

  explicit Info(CkMigrateMessage* msg) : PUP::able(msg) {}

  WRAPPED_PUPable_abstract(Info);  // NOLINT

  virtual auto get_clone() const -> std::unique_ptr<::sys::Info> = 0;

  /// \brief Number of processing elements.
  virtual int number_of_procs() const = 0;

  /// \brief %Index of my processing element.
  virtual int my_proc() const = 0;

  /// \brief Number of nodes.
  virtual int number_of_nodes() const = 0;

  /// \brief %Index of my node.
  virtual int my_node() const = 0;

  /// \brief Number of processing elements on the given node.
  virtual int procs_on_node(int node_index) const = 0;

  /// \brief The local index of my processing element on my node. This is in the
  /// interval 0, ..., procs_on_node(my_node()) - 1.
  virtual int my_local_rank() const = 0;

  /// \brief %Index of first processing element on the given node.
  virtual int first_proc_on_node(int node_index) const = 0;

  /// \brief %Index of the node for the given processing element.
  virtual int node_of(int proc_index) const = 0;

  /// \brief The local index for the given processing element on its node.
  virtual int local_rank_of(int proc_index) const = 0;
};
}  // namespace sys
