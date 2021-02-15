// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions that provide low-level system information such as number
/// of nodes and processors, in a way that they are mockable in ActionTesting.

#pragma once

/// Functionality for parallelization.
///
/// The functions in namespace `Parallel` that return information on
/// nodes and cores are templated on DistribObject.  Actions should
/// use these functions rather than the raw charm++ versions (in the
/// sys namespace in Utilities/System/ParalleInfo.hpp) so that the
/// mocking framework will see the mocked cores and nodes.
namespace Parallel {

/*!
 * \ingroup ParallelGroup
 * \brief Number of processing elements.
 */
template <typename DistribObject>
int number_of_procs(const DistribObject& distributed_object) noexcept {
  return distributed_object.number_of_procs();
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of my processing element.
 */
template <typename DistribObject>
int my_proc(const DistribObject& distributed_object) noexcept {
  return distributed_object.my_proc();
}

/*!
 * \ingroup ParallelGroup
 * \brief Number of nodes.
 */
template <typename DistribObject>
int number_of_nodes(const DistribObject& distributed_object) noexcept {
  return distributed_object.number_of_nodes();
}

 /*!
  * \ingroup ParallelGroup
  * \brief %Index of my node.
  */
template <typename DistribObject>
int my_node(const DistribObject& distributed_object) noexcept {
  return distributed_object.my_node();
}

/*!
 * \ingroup ParallelGroup
 * \brief Number of processing elements on the given node.
 */
template <typename DistribObject>
int procs_on_node(const int node_index,
                  const DistribObject& distributed_object) noexcept {
  return distributed_object.procs_on_node(node_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief The local index of my processing element on my node.
 * This is in the interval 0, ..., procs_on_node(my_node()) - 1.
 */
template <typename DistribObject>
int my_local_rank(const DistribObject& distributed_object) noexcept {
  return distributed_object.my_local_rank();
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of first processing element on the given node.
 */
template <typename DistribObject>
int first_proc_on_node(const int node_index,
                       const DistribObject& distributed_object) noexcept {
  return distributed_object.first_proc_on_node(node_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of the node for the given processing element.
 */
template <typename DistribObject>
int node_of(const int proc_index,
            const DistribObject& distributed_object) noexcept {
  return distributed_object.node_of(proc_index);
}

/*!
 * \ingroup ParallelGroup
 * \brief The local index for the given processing element on its node.
 */
template <typename DistribObject>
int local_rank_of(const int proc_index,
                  const DistribObject& distributed_object) noexcept {
  return distributed_object.local_rank_of(proc_index);
}

}  // namespace Parallel
