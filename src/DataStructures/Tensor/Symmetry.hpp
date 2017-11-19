// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines metafunctions used to comute the Symmetry<...> for a Tensor

#pragma once

#include <type_traits>

#include "Utilities/TMPL.hpp"

namespace detail {
// @{
/// \ingroup TensorGroup
/// \brief Metafunction that computes the new sign map
///
/// \details
/// Metafunction that computes the new sign map depending on whether or not the
/// `index` is already in the sign map. If the current index is _not_ in the
/// sign map then we just return the `sign_map`. If the `index` is in the sign
/// map then we remove it and re-add it but mapping to -1 instead of +1.
/// This is how we handle anti-symmetries where it is the first index in the
/// user list that has the negative sign rather than the second.
///
/// \see compute_sign_map compute_sign
template <bool>
struct compute_sign_map_helper {
  /// \tparam SignMap the sign map to update
  /// \tparam Index the index whose sign to replace
  template <typename SignMap, std::int32_t Index>
  using type = SignMap;
};

template <>
struct compute_sign_map_helper<true> {
  /// \tparam SignMap the sign map to update
  /// \tparam Index the index whose sign to replace
  template <typename SignMap, std::int32_t Index>
  using type = tmpl::insert<
      tmpl::erase<SignMap,
                  tmpl::abs<tmpl::int32_t<static_cast<std::int32_t>(Index)>>>,
      tmpl::pair<tmpl::abs<tmpl::int32_t<Index>>, tmpl::int32_t<-1>>>;
};
// @}

// @{
/// \ingroup TensorGroup
/// Metafunction to compute the new `sign_map` which is a map between indices
/// and +/-1 depending on whether the index is symmetric (or has no symmetry) or
/// is anti-symmetric, respectively.
///
/// Which metafunction implementation is called is determined by whether
/// `index` is already in the `sign_map` or not.
///
/// If `index` is not in `sign_map` then we return a new `sign_map` with the
/// pair <index, sign<index>> added.
///
/// In the case that `index` is in the `sign_map` we use
/// `compute_sign_map_helper` with the last parameter being true if the value
/// already in the `sign_map` is 1 AND the current index is less than 0,
/// otherwise it is false.
///
/// the `bool` is
/// `tmpl::has_key<SignMap, tmpl::abs<tmpl::int32_t<Index>>>::value`
/// where `SignMap` and `Index` are the template parameters on the type alias
template <bool>
struct compute_sign_map;
/// \cond HIDDEN_SYMBOLS
template <>
struct compute_sign_map<true> {
  /// \tparam SignMap map between abs(Index) and +/- 1
  /// \tparam Index the index whose sign to possibly add to SignMap
  template <typename SignMap, std::int32_t Index>
  using type = typename compute_sign_map_helper<
      tmpl::greater<tmpl::at<SignMap, tmpl::abs<tmpl::int32_t<Index>>>,
                    tmpl::int32_t<0>>::value and
      (Index < 0)>::template type<SignMap, Index>;
};
template <>
struct compute_sign_map<false> {
  /// \tparam SignMap map between abs(Index) and +/- 1
  /// \tparam Index the index whose sign to possibly add to SignMap
  template <typename SignMap, std::int32_t Index>
  using type =
      tmpl::insert<SignMap, tmpl::pair<tmpl::abs<tmpl::int32_t<Index>>,
                                       tmpl::sign<tmpl::int32_t<Index>>>>;
};
/// \endcond
// @}

// @{
/// \ingroup TensorGroup
/// A metafunction that, given the current and past SignMap, returns either 1
/// if the absolute value of index is not in the `PastSignMap` or returns the
/// signed value in the `CurrentSignMap` if the absolute value of index is in
/// the `PastSignMap` (and therefore also in the `CurrentSignMap`). The
/// reason why the sign from the map is returned is because it is always the
/// first instance of an index that has the negative sign, and since we compute
/// from right to left this algorithm achieves that.
///
/// if `bool` is `true` then the `CurrentSignMap` has the value `abs<Index>` as
/// a key
template <bool>
struct compute_sign;
/// \cond HIDDEN_SYMBOLS
template <>
struct compute_sign<true> {
  /// \tparam CurrentSignMap current map between abs(Index) and +/- 1
  /// \tparam PastSignMap previous map between abs(Index) and +/- 1
  /// \tparam Index the index to find the sign of
  template <typename CurrentSignMap, typename PastSignMap, std::int32_t Index>
  using type = tmpl::at<CurrentSignMap, tmpl::abs<tmpl::int32_t<Index>>>;
};

template <>
struct compute_sign<false> {
  template <typename CurrentSignMap, typename PastSignMap, std::int32_t Index>
  using type = tmpl::int32_t<1>;
};
/// \endcond
// @}

/// \ingroup TensorGroup
/// Metafunction to compute the symmetry typelist. See the Symmetry type alias
/// for details of how to use.
template <std::int32_t... T>
struct symm;

/// \ingroup TensorGroup
/// Scalar tensor case
template <>
struct symm<> {
  using type = tmpl::integral_list<std::int32_t>;
};

/// \ingroup TensorGroup
/// Last index case to end recursion.
template <std::int32_t index>
struct symm<index> {
  static_assert(index > 0, "Anti-symmetries are currently not supported");
  using type = tmpl::integral_list<std::int32_t, 1>;
  using current_index_t = tmpl::int32_t<1>;
  using added_map =
      tmpl::map<tmpl::pair<tmpl::int32_t<index>, tmpl::int32_t<1>>>;
  using sign_map = tmpl::map<
      tmpl::pair<tmpl::int32_t<index>, tmpl::sign<tmpl::int32_t<index>>>>;
};

/// \ingroup TensorGroup
/// Metafunction that computes symmetry as a typelist of std::integral_constant.
/// The `added_map` is a typemap used to keep track of whether or not a
/// particular index has already been added to the list. If it or its negative
/// (to handle anti-symmetries) has not been added then we add to the map. This
/// conditional add is done using tmpl::conditional_t.
///
/// The `current_index_t` is the current symmetry index, for example value 2 in
/// `Symmetry<1, 2, 1>`. If the current index is already in the `next_added_map`
/// then we just use that value, otherwise we check if `next_added_map` has the
/// negative of the `index`, then use that. Finally, if `index` is not in the
/// `next_added_map` we have a new index given by `current_index_t` in the
/// `next_symm` plus one.
///
/// The `sign_map` is used to keep track of the positive and negative signs
/// needed for anti-symmetries. A helper metafunction, `compute_sign_map` is
/// used to compute the changes to the `sign_map`. The `sign_map` is a map
/// between the symmetry index that is in the typelist and either plus or minus
/// one, depending on whether the index is symmetric or anti-symmetric
/// respectively.
///
/// \tparam index the current index to add to the typelist
/// \tparam T the remaining indices to add to the typelist
template <std::int32_t index, std::int32_t... T>
struct symm<index, T...> {
  static_assert(index > 0, "Anti-symmetries are currently not supported");
  using next_symm = symm<T...>;
  using next_added_map = typename next_symm::added_map;
  using next_sign_map = typename next_symm::sign_map;

  using added_map = tmpl::conditional_t<
      (tmpl::has_key<next_added_map, tmpl::int32_t<index>>::value or
       tmpl::has_key<next_added_map, tmpl::int32_t<-index>>::value),
      next_added_map,
      tmpl::insert<
          next_added_map,
          tmpl::pair<tmpl::int32_t<index>,
                     tmpl::int32_t<next_symm::current_index_t::value + 1>>>>;

  using current_index_t = tmpl::conditional_t<
      tmpl::has_key<next_added_map, tmpl::int32_t<index>>::value,
      tmpl::at<next_added_map, tmpl::int32_t<index>>,
      tmpl::conditional_t<
          tmpl::has_key<next_added_map, tmpl::int32_t<-index>>::value,
          tmpl::at<next_added_map, tmpl::int32_t<-index>>,
          tmpl::int32_t<next_symm::current_index_t::value + 1>>>;

  using sign_map = typename compute_sign_map<
      tmpl::has_key<next_sign_map, tmpl::abs<tmpl::int32_t<index>>>::value>::
      template type<next_sign_map, index>;

  using type = tmpl::push_front<
      typename next_symm::type,
      tmpl::int32_t<
          compute_sign<
              tmpl::has_key<sign_map, tmpl::abs<tmpl::int32_t<index>>>::value>::
              template type<sign_map, next_sign_map, index>::type::value *
          current_index_t::value>>;
};
}  // namespace detail

/// \ingroup TensorGroup
/// \brief Computes the canonical symmetry from the integers `T`
///
/// \details
/// Compute the canonical symmetry typelist given a set of integers, T. The
/// resulting typelist is in descending order of the absolute value of the
/// integers. For example, the result of `Symmetry<1, 2, 3>` is
/// `integral_list<int32_t, 3, 2, 1>`. Anti-symmetries can be denoted with a
/// minus sign on either _or_ both indices. That is, `Symmetry<-1, 2, 1>` is
/// anti-symmetric in the first and last index and is the same as `Symmetry<-1,
/// 2, -1>`. Note: two minus signs are still anti-symmetric because it
/// simplifies the algorithm used to compute the canonical form of the symmetry.
///
/// \tparam T the integers denoting the symmetry of the Tensor
template <std::int32_t... T>
using Symmetry = typename detail::symm<T...>::type;
