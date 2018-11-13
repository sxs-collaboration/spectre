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
  approx_stack_depth++;
  if (approx_stack_depth < 64) {
    return false;
  }
  approx_stack_depth = 0;
  return true;
}

template <typename T, typename = cpp17::void_t<>>
struct has_ckLocal_method : std::false_type {};

template <typename T>
struct has_ckLocal_method<T,
                          cpp17::void_t<decltype(std::declval<T>().ckLocal())>>
    : std::true_type {};
}  // namespace detail

// @{
/*!
 * \ingroup ParallelGroup
 * \brief Send the data `args...` to the algorithm running on `proxy`, and tag
 * the message with the identifier `temporal_id`.
 *
 * If the algorithm was previously disabled, set `enable_if_disabled` to true to
 * enable the algorithm on the parallel component.
 *
 * \note The reason there are two separate functions is because Charm++ does not
 * allow defaulted arguments for group and nodegroup chares.
 */
template <
    typename ReceiveTag, typename Proxy, typename ReceiveDataType,
    Requires<detail::has_ckLocal_method<std::decay_t<Proxy>>::value> = nullptr>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data,
                  const bool enable_if_disabled) noexcept {
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

template <
    typename ReceiveTag, typename Proxy, typename ReceiveDataType,
    Requires<detail::has_ckLocal_method<std::decay_t<Proxy>>::value> = nullptr>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data) noexcept {
  auto* obj = proxy.ckLocal();
  // Only elide the Charm++ RTS if the object is local and if we won't blow the
  // stack by having too many recursive function calls.
  if (obj != nullptr and not detail::max_inline_entry_methods_reached()) {
    obj->template receive_data<ReceiveTag>(
        std::move(temporal_id), std::forward<ReceiveDataType>(receive_data));
  } else {
    proxy.template receive_data<ReceiveTag>(
        std::move(temporal_id), std::forward<ReceiveDataType>(receive_data));
  }
}

template <typename ReceiveTag, typename Proxy, typename ReceiveDataType,
          Requires<not detail::has_ckLocal_method<std::decay_t<Proxy>>::value> =
              nullptr>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data,
                  const bool enable_if_disabled) noexcept {
  proxy.template receive_data<ReceiveTag>(
      std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
      enable_if_disabled);
}

template <typename ReceiveTag, typename Proxy, typename ReceiveDataType,
          Requires<not detail::has_ckLocal_method<std::decay_t<Proxy>>::value> =
              nullptr>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data) noexcept {
  proxy.template receive_data<ReceiveTag>(
      std::move(temporal_id), std::forward<ReceiveDataType>(receive_data));
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
