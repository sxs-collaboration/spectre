// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions for manipulating tuples

#pragma once

#include <tuple>
#include <utility>

namespace tuple_impl_detail {
template <bool ReverseIteration, typename... Elements, typename N_aryOp,
          typename... Args, size_t... Is>
constexpr inline void tuple_fold_impl(
    const std::tuple<Elements...>& tupull, N_aryOp&& op,
    std::index_sequence<Is...> /*meta*/,
    Args&... args) noexcept(
        noexcept(static_cast<void>(std::initializer_list<char>{
    (static_cast<void>(
         op(std::get<(ReverseIteration ? sizeof...(Elements) - 1 - Is : Is)>(
                tupull),
            args...)),
     '0')...}))) {
  constexpr size_t tuple_size = sizeof...(Elements);
  static_cast<void>(std::initializer_list<char>{
      (static_cast<void>(
           op(std::get<(ReverseIteration ? tuple_size - 1 - Is : Is)>(tupull),
              args...)),
       '0')...});
}

template <bool ReverseIteration, typename... Elements, typename N_aryOp,
          typename... Args, size_t... Is>
constexpr inline void tuple_counted_fold_impl(
    const std::tuple<Elements...>& tupull, N_aryOp&& op,
    std::index_sequence<Is...> /*meta*/,
    Args&... args) noexcept(
        noexcept(static_cast<void>(std::initializer_list<char>{
    (static_cast<void>(
         op(std::get<(ReverseIteration ? sizeof...(Elements) - 1 - Is : Is)>(
                tupull),
            (ReverseIteration ? sizeof...(Elements) - 1 - Is : Is), args...)),
     '0')...}))) {
  constexpr size_t tuple_size = sizeof...(Elements);
  static_cast<void>(std::initializer_list<char>{
      (static_cast<void>(
           op(std::get<(ReverseIteration ? tuple_size - 1 - Is : Is)>(tupull),
              (ReverseIteration ? tuple_size - 1 - Is : Is), args...)),
       '0')...});
}

template <bool ReverseIteration, typename... Elements, typename N_aryOp,
          typename... Args, size_t... Is>
constexpr inline void tuple_transform_impl(
    const std::tuple<Elements...>& tupull, N_aryOp&& op,
    std::index_sequence<Is...> /*meta*/,
    Args&... args) noexcept(
        noexcept(static_cast<void>(std::initializer_list<char>{
    (static_cast<void>(op(
         std::get<(ReverseIteration ? sizeof...(Elements) - 1 - Is : Is)>(
             tupull),
         std::integral_constant<
             size_t, (ReverseIteration ? sizeof...(Elements) - 1 - Is : Is)>{},
         args...)),
     '0')...}))) {
  constexpr size_t tuple_size = sizeof...(Elements);
  static_cast<void>(std::initializer_list<char>{(
      static_cast<void>(op(
          std::get<(ReverseIteration ? tuple_size - 1 - Is : Is)>(tupull),
          std::integral_constant<size_t, (ReverseIteration ? tuple_size - 1 - Is
                                                           : Is)>{},
          args...)),
      '0')...});
}
}  // namespace tuple_impl_detail

// @{
/*!
 * \ingroup UtilitiesGroup
 * \brief Perform a fold over a std::tuple
 *
 * \details
 * Iterates over the elements in a std::tuple `tuple` from left to right
 * (left fold) calling `op(element, args...)` on each element in `tuple`. A
 * right fold can be done by explicitly setting the first template parameter
 * to true. Folds are easily implemented using `tuple_fold` by updating
 * one of the `args...` at each iteration. If you need the index of the current
 * element you can use the `tuple_counted_fold` variant. `tuple_counted_fold`
 * passes the current index as the second argument to the Callable `op`. That
 * is, `op(element, index, args...)`.
 *
 * \example
 * The sum of a std::tuple of Arithmetics can be computed in several ways.
 * First, you can use a lambda:
 * \snippet Utilities/Test_Tuple.cpp tuple_fold_lambda
 * You'll notice that `state` is taken by reference and mutated.
 *
 * You can do the same thing with a struct defined as
 * \snippet Utilities/Test_Tuple.cpp tuple_fold_struct_defn
 * and then using an instance of the struct
 * \snippet Utilities/Test_Tuple.cpp tuple_fold_struct
 *
 * \note You are not able to pass a function pointer to `tuple_fold` or
 * `tuple_counted_fold` because you cannot pass a pointer to a function
 * template, only a function.
 *
 * \see expand_pack tuple_transform tuple_fold tuple_counted_fold std::tuple
 */
template <bool ReverseIteration = false, typename... Elements, typename N_aryOp,
          typename... Args>
constexpr inline void tuple_fold(
    const std::tuple<Elements...>& tuple, N_aryOp&& op,
    Args&&... args) noexcept(noexcept(tuple_impl_detail::
                                          tuple_fold_impl<ReverseIteration>(
                                              tuple, std::forward<N_aryOp>(op),
                                              std::make_index_sequence<
                                                  sizeof...(Elements)>{},
                                              args...))) {
  tuple_impl_detail::tuple_fold_impl<ReverseIteration>(
      tuple, std::forward<N_aryOp>(op),
      std::make_index_sequence<sizeof...(Elements)>{}, args...);
}

template <bool ReverseIteration = false, typename... Elements, typename N_aryOp,
          typename... Args>
constexpr inline void tuple_counted_fold(
    const std::tuple<Elements...>& tuple, N_aryOp&& op,
    Args&&... args) noexcept(noexcept(tuple_impl_detail::
                                          tuple_counted_fold_impl<
                                              ReverseIteration>(
                                              tuple, std::forward<N_aryOp>(op),
                                              std::make_index_sequence<
                                                  sizeof...(Elements)>{},
                                              args...))) {
  tuple_impl_detail::tuple_counted_fold_impl<ReverseIteration>(
      tuple, std::forward<N_aryOp>(op),
      std::make_index_sequence<sizeof...(Elements)>{}, args...);
}
// @}

/*!
 * \ingroup UtilitiesGroup
 * \brief Perform a transform over a std::tuple
 *
 * \details
 * Iterates over the elements in a std::tuple `tuple` from left to right
 * calling `op.operator()(element, index, args...)` on each element
 * in `tuple`. A right-to-left transform can be done by explicitly setting
 * the first template parameter to true. The second argument of the invokable
 * will be a deduced `std::integral_constant<size_t, value>`, from which the
 * current index can be extracted by using `decltype(index)::%value`.
 * For a function object the `decltype(index)` can be replaced by the deduced
 * type of `index`. For example,
 * \snippet Utilities/Test_Tuple.cpp tuple_transform_negate
 *
 * Using `tuple_transform` with a generic lambda goes as follows,
 * \snippet Utilities/Test_Tuple.cpp tuple_transform
 *
 * \see expand_pack tuple_fold tuple_counted_fold std::tuple
 */
template <bool ReverseIteration = false, typename... Elements, typename N_aryOp,
          typename... Args>
constexpr inline void tuple_transform(
    const std::tuple<Elements...>& tuple, N_aryOp&& op,
    Args&&... args) noexcept(noexcept(tuple_impl_detail::
                                          tuple_transform_impl<
                                              ReverseIteration>(
                                              tuple, std::forward<N_aryOp>(op),
                                              std::make_index_sequence<
                                                  sizeof...(Elements)>{},
                                              args...))) {
  tuple_impl_detail::tuple_transform_impl<ReverseIteration>(
      tuple, std::forward<N_aryOp>(op),
      std::make_index_sequence<sizeof...(Elements)>{}, args...);
}
