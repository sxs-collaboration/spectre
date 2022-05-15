// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions that provide low-level system information such as number
/// of nodes and processors, in a way that they are mockable in ActionTesting.

#pragma once

#include <cstddef>

/// Functionality for parallelization.
///
/// The functions in namespace `Parallel` that return information on
/// nodes and cores are templated on DistribObject.  Actions should
/// use these functions rather than the raw charm++ versions (in the
/// sys namespace in Utilities/System/ParallelInfo.hpp) so that the
/// mocking framework will see the mocked cores and nodes.
namespace Parallel {

/*!
 * \ingroup ParallelGroup
 * \brief Number of processing elements.
 */
template <typename T, typename DistribObject>
T number_of_procs(const DistribObject& distributed_object) {
  return static_cast<T>(distributed_object.number_of_procs());
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of my processing element.
 */
template <typename T, typename DistribObject>
T my_proc(const DistribObject& distributed_object) {
  return static_cast<T>(distributed_object.my_proc());
}

/*!
 * \ingroup ParallelGroup
 * \brief Number of nodes.
 */
template <typename T, typename DistribObject>
T number_of_nodes(const DistribObject& distributed_object) {
  return static_cast<T>(distributed_object.number_of_nodes());
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of my node.
 */
template <typename T, typename DistribObject>
T my_node(const DistribObject& distributed_object) {
  return static_cast<T>(distributed_object.my_node());
}

/*!
 * \ingroup ParallelGroup
 * \brief Number of processing elements on the given node.
 */
template <typename T, typename R, typename DistribObject>
T procs_on_node(const R node_index, const DistribObject& distributed_object) {
  return static_cast<T>(
      distributed_object.procs_on_node(static_cast<int>(node_index)));
}

/*!
 * \ingroup ParallelGroup
 * \brief The local index of my processing element on my node.
 * This is in the interval 0, ..., procs_on_node(my_node()) - 1.
 */
template <typename T, typename DistribObject>
T my_local_rank(const DistribObject& distributed_object) {
  return static_cast<T>(distributed_object.my_local_rank());
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of first processing element on the given node.
 */
template <typename T, typename R, typename DistribObject>
T first_proc_on_node(const R node_index,
                     const DistribObject& distributed_object) {
  return static_cast<T>(
      distributed_object.first_proc_on_node(static_cast<int>(node_index)));
}

/*!
 * \ingroup ParallelGroup
 * \brief %Index of the node for the given processing element.
 */
template <typename T, typename R, typename DistribObject>
T node_of(const R proc_index, const DistribObject& distributed_object) {
  return static_cast<T>(
      distributed_object.node_of(static_cast<int>(proc_index)));
}

/*!
 * \ingroup ParallelGroup
 * \brief The local index for the given processing element on its node.
 */
template <typename T, typename R, typename DistribObject>
T local_rank_of(const R proc_index, const DistribObject& distributed_object) {
  return static_cast<T>(
      distributed_object.local_rank_of(static_cast<int>(proc_index)));
}
}  // namespace Parallel
