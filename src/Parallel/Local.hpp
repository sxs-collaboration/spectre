// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>

#include "Parallel/TypeTraits.hpp"

namespace Parallel {

/// Wrapper for calling Charm++'s `.ckLocal()` on a proxy
///
/// The Proxy must be to a Charm++ singleton chare or to a Charm++ array element
/// chare (i.e., the proxy obtained by indexing into a Charm++ array chare).
///
/// The function returns a pointer to the chare if it exists on the local
/// processor, and NULL if it does not. See the Charm++ documentation.
/// It is the responsibility of the user to check the result pointer is valid.
template <typename Proxy>
auto* local(Proxy&& proxy) noexcept {
  // It only makes sense to call .ckLocal() on some kinds of proxies
  static_assert(is_chare_proxy<std::decay_t<Proxy>>::value or
                is_array_element_proxy<std::decay_t<Proxy>>::value or
                is_array_proxy<std::decay_t<Proxy>>::value);
  if constexpr (is_array_proxy<std::decay_t<Proxy>>::value) {
    // The array case should be a single-element array serving as a singleton
    //
    // TODO for code review: can we check this somehow? Using an int as the
    // array_index is a good start, because most of our array chares use a
    // different array index type, but this isn't robust...
    return proxy[0].ckLocal();
  } else {
    return proxy.ckLocal();
  }
}

/// Wrapper for calling Charm++'s `.ckLocalBranch()` on a proxy
///
/// The Proxy must be to a Charm++ group chare or nodegroup chare.
///
/// The function returns a pointer to the local group/nodegroup chare.
template <typename Proxy>
auto* local_branch(Proxy&& proxy) noexcept {
  // It only makes sense to call .ckLocalBranch() on some kinds of proxies
  static_assert(is_group_proxy<std::decay_t<Proxy>>::value or
                is_node_group_proxy<std::decay_t<Proxy>>::value);
  return proxy.ckLocalBranch();
}

}  // namespace Parallel
