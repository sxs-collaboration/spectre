// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <charm++.h>
#include <tuple>

#include "Parallel/PupStlCpp11.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Parallel {
/*!
 * \ingroup ParallelGroup
 * \brief Used for reducing heterogeneous collection of types in a single
 * reduction call
 *
 * For each `ReductionData` you must write a custom reducer to reduce the data.
 * Here is an example of such a function:
 *
 * \snippet Test_AlgorithmReduction.cpp custom_reduce_function
 *
 * A variable number of `CkReductionMsg*`s are passed to the reducer, and so it
 * is necessary to loop over each message, contributing the data into a new
 * `ReductionData` (called `reduced` in the above example). The
 * `Parallel::get()` function is used to retrieve elements from the
 * `ReductionData` in a manner that is completely analogous to retrieving
 * elements from a `std::tuple` using `std::get`. Once the new `ReductionData`
 * has been filled, a new `CkReductionMsg*` must be returned using
 * `Parallel::new_reduction_msg()`.
 *
 * The custom reducer must be registered with Charm++, which requires two steps.
 * First, you must add:
 *
 * \snippet Test_AlgorithmReduction.cpp custom_reduce_register_function
 *
 * to the translation unit (cpp file) that you will be compiling the reducer
 * into. Next, in order to be able to use the reducer, the translation unit that
 * sets up the Charm++ related code must have a function pointer to the register
 * function you write in the vector called `charm_init_node_funcs`. For example:
 *
 * \snippet Test_AlgorithmReduction.cpp custom_reduce_register
 */
template <class... Ts>
struct ReductionData {
  template <class Arg0, class... Args,
            Requires<not cpp17::is_same_v<std::decay_t<Arg0>, ReductionData>> =
                nullptr>
  explicit ReductionData(Arg0&& arg0, Args&&... args)
      : data(std::forward<Arg0>(arg0), std::forward<Args>(args)...) {}

  ReductionData() = default;
  ReductionData(const ReductionData& /*rhs*/) = default;
  ReductionData& operator=(const ReductionData& /*rhs*/) = default;
  ReductionData(ReductionData&& /*rhs*/) = default;
  ReductionData& operator=(ReductionData&& /*rhs*/) = default;
  ~ReductionData() = default;

  explicit ReductionData(CkReductionMsg* const message) noexcept {
    PUP::fromMem creator(message->getData());
    creator | *this;
  }

  /// \cond
  // clang-tidy: non-const reference
  void pup(PUP::er& p) noexcept { p | data; }  // NOLINT

  std::unique_ptr<char[]> packed() noexcept;

  size_t size() noexcept;

 private:
  // clang-tidy: false positive redundant declaration
  template <size_t Index, class... Us>
  friend auto get(ReductionData<Us...>& reduction_data) noexcept  // NOLINT
      -> decltype(std::get<Index>(reduction_data.data));
  // clang-tidy: false positive redundant declaration
  template <size_t Index, class... Us>
  friend auto get(  // NOLINT
      const ReductionData<Us...>& reduction_data) noexcept
      -> decltype(std::get<Index>(reduction_data.data));
  // clang-tidy: false positive redundant declaration
  template <size_t Index, class... Us>
  friend auto get(ReductionData<Us...>&& reduction_data) noexcept  // NOLINT
      -> decltype(std::get<Index>(reduction_data.data));
  // clang-tidy: false positive redundant declaration
  template <size_t Index, class... Us>
  friend auto get(  // NOLINT
      const ReductionData<Us...>&& reduction_data) noexcept
      -> decltype(std::get<Index>(reduction_data.data));

  std::tuple<Ts...> data;
  /// \endcond
};

/// \cond
template <class... Ts>
std::unique_ptr<char[]> ReductionData<Ts...>::packed() noexcept {
  auto result = std::make_unique<char[]>(size());
  PUP::toMem packer(result.get());
  packer | *this;
  return result;
}

template <class... Ts>
size_t ReductionData<Ts...>::size() noexcept {
  PUP::sizer size_pup;
  size_pup | *this;
  return size_pup.size();
}
/// \endcond

/*!
 * \ingroup ParallelGroup
 * \brief Retrieve the `Index`th element from a `ReductionData<Ts...>`, similar
 * to `std::get` for `std::tuple`s
 *
 * \note Also available for rvalue and const references
 */
template <size_t Index, class... Us>
auto get(ReductionData<Us...>& reduction_data) noexcept
    -> decltype(std::get<Index>(reduction_data.data)) {
  return std::get<Index>(reduction_data.data);
}
/// \cond
template <size_t Index, class... Us>
auto get(const ReductionData<Us...>& reduction_data) noexcept
    -> decltype(std::get<Index>(reduction_data.data)) {
  return std::get<Index>(reduction_data.data);
}
template <size_t Index, class... Us>
auto get(ReductionData<Us...>&& reduction_data) noexcept
    -> decltype(std::get<Index>(reduction_data.data)) {
  return std::get<Index>(reduction_data.data);
}
template <size_t Index, class... Us>
auto get(const ReductionData<Us...>&& reduction_data) noexcept
    -> decltype(std::get<Index>(reduction_data.data)) {
  return std::get<Index>(reduction_data.data);
}
/// \endcond

namespace Parallel_detail {
template <class T, class = cpp17::void_t<>>
struct is_custom_reduction_type : std::false_type {};

template <class T>
struct is_custom_reduction_type<
    T,
    cpp17::void_t<
        Requires<cpp17::is_same_v<std::unique_ptr<char[]>,
                                  decltype(std::declval<T>().packed())>>,
        decltype(std::declval<T>().pup(std::declval<PUP::er&>())),
        Requires<cpp17::is_same_v<size_t, decltype(std::declval<T>().size())>>,
        Requires<std::is_constructible<T, CkReductionMsg* const>::value>>>
    : std::true_type {};

template <class T>
constexpr bool is_custom_reduction_type_v = is_custom_reduction_type<T>::value;
}  // namespace Parallel_detail

/*!
 * \ingroup ParallelGroup
 * \brief Perform a reduction from the current ParallelComponent to the
 * `TargetParallelComponent`, performing the `Action` upon receiving the
 * reduction
 *
 * The function must receive a `ConstGlobalCache` and `ArrayIndex` of
 * the sending ParallelComponent. A Charm++ reducer specifying what type of
 * reduction is to be done must also be passed (see
 * [here](http://charm.cs.illinois.edu/manuals/html/charm++/manual.html)),
 * as well as the data that is to be reduced.
 *
 * \example
 * Built-in Charm++ reductions are supported as:
 * \snippet Test_AlgorithmReduction.cpp contribute_to_reduction_example
 * and similarly custom reductions:
 * \snippet Test_AlgorithmReduction.cpp custom_contribute_to_reduction_example
 */
template <
    class SenderParallelComponent, class TargetParallelComponent, class Action,
    class Metavariables, class ArrayIndex, class ReductionType,
    Requires<not Parallel_detail::is_custom_reduction_type_v<ReductionType>> =
        nullptr>
void contribute_to_reduction(const ConstGlobalCache<Metavariables>& cache,
                             const ArrayIndex& array_index,
                             CkReduction::reducerType reducer,
                             const ReductionType& reduction_data) noexcept {
  CkCallback callback(
      Parallel::index_from_parallel_component<TargetParallelComponent>::
          template redn_wrapper_reduction_action<Action,
                                                 std::decay_t<ReductionType>>(
              nullptr),
      Parallel::get_parallel_component<TargetParallelComponent>(cache));
  Parallel::get_parallel_component<SenderParallelComponent>(cache)[array_index]
      .ckLocal()
      ->contribute(sizeof(reduction_data), &reduction_data, reducer, callback);
}

/// \cond
// Documented as part of other implementation
template <class SenderParallelComponent, class TargetParallelComponent,
          class Action, class Metavariables, class ArrayIndex,
          class ReductionType,
          Requires<Parallel_detail::is_custom_reduction_type_v<
              std::decay_t<ReductionType>>> = nullptr>
void contribute_to_reduction(const ConstGlobalCache<Metavariables>& cache,
                             const ArrayIndex& array_index,
                             CkReduction::reducerType reducer,
                             ReductionType&& reduction_data) noexcept {
  CkCallback callback(
      Parallel::index_from_parallel_component<TargetParallelComponent>::
          template redn_wrapper_reduction_action<Action,
                                                 std::decay_t<ReductionType>>(
              nullptr),
      Parallel::get_parallel_component<TargetParallelComponent>(cache));
  Parallel::get_parallel_component<SenderParallelComponent>(cache)[array_index]
      .ckLocal()
      ->contribute(static_cast<int>(reduction_data.size()),
                   reduction_data.packed().get(), reducer, callback);
}
/// \endcond

/*!
 * \ingroup ParallelGroup
 * \brief Convert a `ReductionData` to a `CkReductionMsg`. Used in custom
 * reducers.
 *
 * \example
 * See the return statement of:
 * \snippet Test_AlgorithmReduction.cpp custom_reduce_function
 */
template <class... Ts>
CkReductionMsg* new_reduction_msg(
    ReductionData<Ts...>& reduction_data) noexcept {
  return CkReductionMsg::buildNew(static_cast<int>(reduction_data.size()),
                                  reduction_data.packed().get());
}
}  // namespace Parallel
