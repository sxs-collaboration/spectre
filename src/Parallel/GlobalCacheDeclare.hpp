// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Forward-declares CProxy_GlobalCache which MutableGlobalCache needs, but
/// GlobalCache is defined after MutableGlobalCache. Also forward declares
/// ResourceInfo which the GlobalCache has an entry method for.

#pragma once

/// \cond
namespace Parallel {
template <typename Metavariables>
struct ResourceInfo;
template <class Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond
