// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions that provide low-level system information such as number
/// of nodes and processors.  When working with distributed objects, one should
/// use the corresponding functions in Parallel/Info.hpp instead of the
/// low-level functions here.

#pragma once

#include <string>

namespace sys {
/*!
 * \ingroup UtilitiesGroup
 * \brief Number of processing elements.
 */
int number_of_procs();

/*!
 * \ingroup UtilitiesGroup
 * \brief %Index of my processing element.
 */
int my_proc();

/*!
 * \ingroup UtilitiesGroup
 * \brief Number of nodes.
 */
int number_of_nodes();

/*!
 * \ingroup UtilitiesGroup
 * \brief %Index of my node.
 */
int my_node();

/*!
 * \ingroup UtilitiesGroup
 * \brief Number of processing elements on the given node.
 */
int procs_on_node(int node_index);

/*!
 * \ingroup UtilitiesGroup
 * \brief The local index of my processing element on my node.
 * This is in the interval 0, ..., procs_on_node(my_node()) - 1.
 */
int my_local_rank();

/*!
 * \ingroup UtilitiesGroup
 * \brief %Index of first processing element on the given node.
 */
int first_proc_on_node(int node_index);

/*!
 * \ingroup UtilitiesGroup
 * \brief %Index of the node for the given processing element.
 */
int node_of(int proc_index);

/*!
 * \ingroup UtilitiesGroup
 * \brief The local index for the given processing element on its node.
 */
int local_rank_of(int proc_index);

/*!
 * \ingroup UtilitiesGroup
 * \brief The elapsed wall time in seconds.
 */
double wall_time();

/// @{
/// \ingroup UtilitiesGroup
/// \brief Format the wall time in DD-HH:MM:SS format.
///
/// If the walltime is shorter than a day, omit the `DD-` part.
std::string pretty_wall_time(double total_seconds);

std::string pretty_wall_time();
/// @}
}  // namespace sys
