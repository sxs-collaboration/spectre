// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines type traits related to Charm++ types

#pragma once

#include <charm++.h>

namespace Parallel {

/// \ingroup Parallel
/// Check if `T` is a Charm++ proxy for an array chare
template <typename T>
struct is_array_proxy : std::is_base_of<CProxy_ArrayElement, T>::type {};

/// \ingroup Parallel
/// Check if `T` is a Charm++ proxy for a chare
template <typename T>
struct is_chare_proxy : std::is_base_of<CProxy_Chare, T>::type {};

/// \ingroup Parallel
/// Check if `T` is a Charm++ proxy for a group chare
template <typename T>
struct is_group_proxy : std::is_base_of<CProxy_IrrGroup, T>::type {};

/// \ingroup Parallel
/// Check if `T` is a Charm++ proxy for a node group chare
template <typename T>
struct is_node_group_proxy : std::is_base_of<CProxy_NodeGroup, T>::type {};

} // namespace Parallel
