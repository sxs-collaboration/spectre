// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <type_traits>
#include <utility>

#include "Parallel/MaxInlineMethodsReached.hpp"
#include "Parallel/TypeTraits.hpp"

namespace Parallel {
namespace detail {
template <typename T, typename = std::void_t<>>
struct has_ckLocal_method : std::false_type {};

template <typename T>
struct has_ckLocal_method<T, std::void_t<decltype(std::declval<T>().ckLocal())>>
    : std::true_type {};
}  // namespace detail

/*!
 * \ingroup ParallelGroup
 * \brief Send the data `args...` to the algorithm running on `proxy`, and tag
 * the message with the identifier `temporal_id`.
 *
 * If the algorithm was previously disabled, set `enable_if_disabled` to true to
 * enable the algorithm on the parallel component.
 */
template <typename ReceiveTag, typename Proxy, typename ReceiveDataType>
void receive_data(Proxy&& proxy, typename ReceiveTag::temporal_id temporal_id,
                  ReceiveDataType&& receive_data,
                  const bool enable_if_disabled = false) noexcept {
  // Both branches of this if statement call into the charm proxy system, but
  // because we specify [inline] for array chares, we must specify the
  // `ReceiveDataType` explicitly in the template arguments when dispatching to
  // array chares.
  // This is required because of inconsistent overloads in the generated charm++
  // files when [inline] is specified vs when it is not. The [inline] version
  // generates a function signature with template parameters associated with
  // each function argument, ensuring a redundant template parameter for the
  // `ReceiveDataType`. Then, that template parameter cannot be inferred so must
  // be explicitly specified.
  if constexpr (detail::has_ckLocal_method<std::decay_t<Proxy>>::value) {
    proxy.template receive_data<ReceiveTag, std::decay_t<ReceiveDataType>>(
        std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
        enable_if_disabled);
  } else {
    proxy.template receive_data<ReceiveTag>(
        std::move(temporal_id), std::forward<ReceiveDataType>(receive_data),
        enable_if_disabled);
  }
}

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

/*!
 * \ingroup ParallelGroup
 * \brief Invoke a local synchronous action on `proxy`
 */
template <typename Action, typename Proxy, typename... Args>
decltype(auto) local_synchronous_action(Proxy&& proxy,
                                        Args&&... args) noexcept {
  return proxy.ckLocalBranch()->template local_synchronous_action<Action>(
      std::forward<Args>(args)...);
}

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
