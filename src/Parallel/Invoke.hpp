// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "Parallel/TypeTraits.hpp"
#include "Utilities/Requires.hpp"

namespace Parallel {
namespace detail {
// Allow 64 inline entry method calls before we fall back to Charm++. This is
// done to avoid blowing the stack.
inline bool max_inline_entry_methods_reached() noexcept {
  thread_local size_t approx_stack_depth = 0;
#ifndef SPECTRE_PROFILING
  approx_stack_depth++;
  if (approx_stack_depth < 64) {
    return false;
  }
  approx_stack_depth = 0;
#endif
  return true;
}

template <typename T, typename = std::void_t<>>
struct has_ckLocal_method : std::false_type {};

template <typename T>
struct has_ckLocal_method<T, std::void_t<decltype(std::declval<T>().ckLocal())>>
    : std::true_type {};

// Any class template that disables its ckLocal() function for some
// template parameters (via static_assert or if constexpr) should
// define a type alias ckLocal_enabled to be std::true_type for
// cases where ckLocal() is enabled, and std::false_type for cases
// where ckLocal() is disabled.  If ckLocal_enabled isn't defined,
// then ckLocal() is understood to be enabled.
template<typename T, typename = std::void_t<>>
struct ckLocal_enabled {
  using type = std::true_type;
};

template <typename T>
struct ckLocal_enabled<T, std::void_t<typename T::ckLocal_enabled>> {
  using type = typename T::ckLocal_enabled;
};

template <typename T>
using ckLocal_exists_and_is_enabled_t =
    std::conditional_t<has_ckLocal_method<T>::value,
                       typename ckLocal_enabled<T>::type, std::false_type>;

}  // namespace detail

// @{
/*!
 * \ingroup ParallelGroup
 * \brief Send the data `args...` to the algorithm running on `proxy`, and tag
 * the message with the identifier `temporal_id`.
 *
 * If the algorithm was previously disabled, set `enable_if_disabled` to true to
 * enable the algorithm on the parallel component.
 */
template <typename ReceiveTag, typename Proxy, typename ReceiveDataType,
          Requires<detail::ckLocal_exists_and_is_enabled_t<
              std::decay_t<Proxy>>::value> = nullptr>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data,
                  const bool enable_if_disabled = false) noexcept {
  auto* obj = proxy.ckLocal();
  // Only elide the Charm++ RTS if the object is local and if we won't blow the
  // stack by having too many recursive function calls.
  if (obj != nullptr and not detail::max_inline_entry_methods_reached()) {
    obj->template receive_data<ReceiveTag>(
        std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
        enable_if_disabled);
  } else {
    proxy.template receive_data<ReceiveTag>(
        std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
        enable_if_disabled);
  }
}

template <typename ReceiveTag, typename Proxy, typename ReceiveDataType,
          Requires<not detail::ckLocal_exists_and_is_enabled_t<
              std::decay_t<Proxy>>::value> = nullptr>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data,
                  const bool enable_if_disabled = false) noexcept {
  proxy.template receive_data<ReceiveTag>(
      std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
      enable_if_disabled);
}
// @}

// @{
/*!
 * \ingroup ParallelGroup
 * \brief Invoke a simple action on `proxy`
 */
template <typename Action, typename Proxy>
void simple_action(Proxy&& proxy) noexcept {
  proxy.template simple_action<Action>();
}

template <typename Action, typename Proxy, typename Arg0, typename... Args>
void simple_action(Proxy&& proxy, Arg0&& arg0, Args&&... args) noexcept {
  proxy.template simple_action<Action>(
      std::tuple<std::decay_t<Arg0>, std::decay_t<Args>...>(
          std::forward<Arg0>(arg0), std::forward<Args>(args)...));
}
// @}

// @{
/*!
 * \ingroup ParallelGroup
 * \brief Invoke a threaded action on `proxy`, where the proxy must be a
 * nodegroup.
 */
template <typename Action, typename Proxy>
void threaded_action(Proxy&& proxy) noexcept {
  proxy.template threaded_action<Action>();
}

template <typename Action, typename Proxy, typename Arg0, typename... Args>
void threaded_action(Proxy&& proxy, Arg0&& arg0, Args&&... args) noexcept {
  proxy.template threaded_action<Action>(
      std::tuple<std::decay_t<Arg0>, std::decay_t<Args>...>(
          std::forward<Arg0>(arg0), std::forward<Args>(args)...));
}
// @}
}  // namespace Parallel
