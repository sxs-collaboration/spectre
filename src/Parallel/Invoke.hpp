// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <utility>

namespace Parallel {
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
template <typename ReceiveTag, typename Proxy, typename ReceiveDataType>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data,
                  const bool enable_if_disabled) noexcept {
  proxy.template receive_data<ReceiveTag>(
      std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
      enable_if_disabled);
}

template <typename ReceiveTag, typename Proxy, typename ReceiveDataType>
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
