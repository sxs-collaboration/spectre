// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions that provide low-level system information such as number
/// of nodes and processors, in a way that they are mockable in ActionTesting.

#pragma once

#include <memory>
#include <pup.h>

#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/System/Info.hpp"

/// Functionality for parallelization.
///
/// The functions in namespace `Parallel` that return information on
/// nodes and cores are templated on DistribObject.  Actions should
/// use these functions rather than the raw charm++ versions (in the
/// sys namespace in Utilities/System/ParallelInfo.hpp) so that the
/// mocking framework will see the mocked cores and nodes.
namespace Parallel {
/// Low-level system information such as number of nodes and processors for the
/// Charm++ runtime system.
class Info final : public ::sys::Info {
 public:
  Info() = default;
  Info(Info&&) = default;
  Info& operator=(Info&&) = default;
  Info(const Info&) = default;
  Info& operator=(const Info&) = default;
  ~Info() override = default;

  explicit Info(CkMigrateMessage* msg);
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(Info);  // NOLINT

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
};

/*!
 * \ingroup ParallelGroup
 * \brief Number of processing elements.
 */
template <typename DistribObject>
int number_of_procs(const DistribObject& distributed_object) {
  return distributed_object.number_of_procs();
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of my processing element.
 */
template <typename DistribObject>
int my_proc(const DistribObject& distributed_object) {
  return distributed_object.my_proc();
}

/*!
 * \ingroup ParallelGroup
 * \brief Number of nodes.
 */
template <typename DistribObject>
int number_of_nodes(const DistribObject& distributed_object) {
  return distributed_object.number_of_nodes();
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of my node.
 */
template <typename DistribObject>
int my_node(const DistribObject& distributed_object) {
  return distributed_object.my_node();
}

/*!
 * \ingroup ParallelGroup
 * \brief Number of processing elements on the given node.
 */
template <typename DistribObject>
int procs_on_node(const int node_index,
                  const DistribObject& distributed_object) {
  return distributed_object.procs_on_node(node_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief The local index of my processing element on my node.
 * This is in the interval 0, ..., procs_on_node(my_node() - 1.
 */
template <typename DistribObject>
int my_local_rank(const DistribObject& distributed_object) {
  return distributed_object.my_local_rank();
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of first processing element on the given node.
 */
template <typename DistribObject>
int first_proc_on_node(const int node_index,
                       const DistribObject& distributed_object) {
  return distributed_object.first_proc_on_node(node_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of the node for the given processing element.
 */
template <typename DistribObject>
int node_of(const int proc_index, const DistribObject& distributed_object) {
  return distributed_object.node_of(proc_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief The local index for the given processing element on its node.
 */
template <typename DistribObject>
int local_rank_of(const int proc_index,
                  const DistribObject& distributed_object) {
  return distributed_object.local_rank_of(proc_index);
}
}  // namespace Parallel
