// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Forward-declares CProxy_GlobalCache which MutableGlobalCache needs, but
/// GlobalCache is defined after MutableGlobalCache.

#pragma once

/// \cond
namespace Parallel {
template <class Metavariables>
class CProxy_GlobalCache;
}  // namespace Parallel
/// \endcond
