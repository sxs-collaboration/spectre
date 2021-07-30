// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions that provide low-level system information such as number
/// of nodes and processors.  When working with distributed objects, one should
/// use the corresponding functions in Parallel/Info.hpp instead of the
/// low-level functions here.

#pragma once

#include <charm++.h>

namespace sys {
/*!
 * \ingroup UtilitiesGroup
 * \brief Number of processing elements.
 */
inline int number_of_procs() { return CkNumPes(); }

/*!
 * \ingroup UtilitiesGroup
 * \brief %Index of my processing element.
 */
inline int my_proc() { return CkMyPe(); }

/*!
 * \ingroup UtilitiesGroup
 * \brief Number of nodes.
 */
inline int number_of_nodes() { return CkNumNodes(); }

/*!
 * \ingroup UtilitiesGroup
 * \brief %Index of my node.
 */
inline int my_node() { return CkMyNode(); }

/*!
 * \ingroup UtilitiesGroup
 * \brief Number of processing elements on the given node.
 */
inline int procs_on_node(const int node_index) {
  // When using the verbs-linux-x86_64 non-SMP build of Charm++ these
  // functions have unused-parameter warnings. This is remedied by
  // casting the integer to a void which results in no extra assembly
  // code being generated. We use this instead of pragmas because we
  // would require one pragma for GCC and one for clang, which would
  // result in code duplication. Commenting out the variable
  // nodeIndex gives compilation failures on most Charm++ builds since
  // they actually use the variable. Charm++ plz...
  static_cast<void>(node_index);
  return CkNodeSize(node_index);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief The local index of my processing element on my node.
 * This is in the interval 0, ..., procs_on_node(my_node()) - 1.
 */
inline int my_local_rank() { return CkMyRank(); }

/*!
 * \ingroup UtilitiesGroup
 * \brief %Index of first processing element on the given node.
 */
inline int first_proc_on_node(const int node_index) {
  static_cast<void>(node_index);
  return CkNodeFirst(node_index);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief %Index of the node for the given processing element.
 */
inline int node_of(const int proc_index) {
  static_cast<void>(proc_index);
  return CkNodeOf(proc_index);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief The local index for the given processing element on its node.
 */
inline int local_rank_of(const int proc_index) {
  static_cast<void>(proc_index);
  return CkRankOf(proc_index);
}

/*!
 * \ingroup UtilitiesGroup
 * \brief The elapsed wall time in seconds.
 */
inline double wall_time() { return CkWallTimer(); }
}  // namespace sys
