// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines metafunctions used by Tensor

#pragma once

#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

/// \ingroup TensorGroup
/// Contains all metafunctions related to Tensor manipulations
namespace TensorMetafunctions {
namespace detail {
/*!
 * \ingroup TensorGroup
 * Compute the number of components and independent components for a given
 * Symmetry and TensorIndex list
 * \tparam Symmetry_t Symmetry of the Tensor
 * \tparam Indices_t typelist of the TensorIndex's of the Tensor
 */
template <typename Symmetry_t, typename Indices_t, typename = std::nullptr_t>
struct number_of_independent_components_impl;

/*!
 * \ingroup TensorGroup
 * Case where the Tensor is a scalar
 */
template <>
struct number_of_independent_components_impl<tmpl::list<>, tmpl::list<>> {
  using number_of_components = tmpl::size_t<1>;
  using number_of_independent_components = tmpl::size_t<1>;
};

/*!
 * \ingroup TensorGroup
 * End recursion, only one index left. The component number is 1 (first
 * component) and the number of components and independent components is the
 * dimensionality of the first index
 */

template <typename SymmetryValue, typename Index>
struct number_of_independent_components_impl<typelist<SymmetryValue>,
                                             typelist<Index>, std::nullptr_t> {
  using number_of_independent_components = tmpl::size_t<Index::dim>;
  using component_number = tmpl::size_t<1>;
  using number_of_components = tmpl::size_t<Index::dim>;
};

/// \cond HIDDEN_SYMBOLS
/*!
 * \ingroup TensorGroup
 *
 * General case for computing the number of (independent) components. The type
 * alias `next` computes the number of components with one fewer TensorIndex,
 * popping the front off the typelists. The `number_of_components` is given by
 * the product of the dimensionality of each TensorIndex. The component_number
 * is used to keep track of whether the current component is independent or
 * not. This is done by comparing the `component_number + 1` with the first
 * element in Symmetry, if they are equal then we have a new independent
 * component, otherwise the component is a symmetry. For the `component_number`
 * comparison to make sense recall that symmetry is sorted in _descending_
 * order.
 * Finally, to compute the number of independent components we perform the same
 * check with `component_number + 1` and the front of the Symmetry list. If
 * this is a new component we multiply by the dimensionality of the index,
 * otherwise we multiply by `dim + 1` or `dim - 1` depending on whether the
 * index is symmetric or anti-symmetric, respectively and  divide by two.
 */
template <typename FirstSymm, typename... SymmetryValues, typename FirstIndex,
          typename... Indices>
struct number_of_independent_components_impl<
    typelist<FirstSymm, SymmetryValues...>, typelist<FirstIndex, Indices...>,
    Requires<(sizeof...(SymmetryValues) > 0 and sizeof...(Indices) > 0)>> {
  static_assert(
      sizeof...(SymmetryValues) == sizeof...(Indices),
      "Number of components in Symmetry_t and Indices_t must be the same.");
  using next =
      number_of_independent_components_impl<typelist<SymmetryValues...>,
                                            typelist<Indices...>>;
  using Symm_t = typelist<SymmetryValues...>;
  using Indi_t = typelist<Indices...>;

  using number_of_components =
      tmpl::size_t<next::number_of_components::value * FirstIndex::dim>;

  using component_number =
      tmpl::conditional_t<next::component_number::value + 1 ==
                              tmpl::abs<FirstSymm>::value,
                          tmpl::size_t<next::component_number::value + 1>,
                          tmpl::size_t<next::component_number::value>>;
  using number_of_independent_components = tmpl::conditional_t<
      next::component_number::value + 1 == tmpl::abs<FirstSymm>::value,
      tmpl::size_t<next::number_of_independent_components::value *
                   FirstIndex::dim>,
      tmpl::size_t<(next::number_of_independent_components::value *
                    (FirstSymm::value > 0 ? FirstIndex::dim + 1
                                          : FirstIndex::dim - 1)) /
                   2>>;
};
/// \endcond
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Compute the number of independent components for a given ::Symmetry
 * and \ref SpacetimeIndex "TensorIndexType" list
 * \tparam Symmetry_t Symmetry of the Tensor
 * \tparam Indices_t typelist of the TensorIndex's of the Tensor
 */
template <typename Symmetry_t, typename Indices_t>
using independent_components =
    typename detail::number_of_independent_components_impl<
        Symmetry_t, Indices_t>::number_of_independent_components;

/*!
 * \ingroup TensorGroup
 * \brief Compute the number of components for a given ::Symmetry and \ref
 * SpacetimeIndex "TensorIndexType" list
 * \tparam Symmetry_t Symmetry of the Tensor
 * \tparam Indices_t typelist of the TensorIndex's of the Tensor
 */
template <typename Symmetry_t, typename Indices_t>
using number_of_components =
    typename detail::number_of_independent_components_impl<
        Symmetry_t, Indices_t>::number_of_components;

namespace detail {
template <bool>
struct increment_tensor_index_impl;

template <>
struct increment_tensor_index_impl<false> {
  template <typename IndexList, typename TensorIndex, size_t I>
  using type = tmpl::replace_at<
      TensorIndex, tmpl::size_t<I>,
      tmpl::size_t<tmpl::at<TensorIndex, tmpl::size_t<I>>::value + 1>>;
};

template <>
struct increment_tensor_index_impl<true> {
  template <typename IndexList, typename TensorIndex, size_t I>
  using type =
      typename ::TensorMetafunctions::detail::increment_tensor_index_impl<(
          tmpl::at<TensorIndex, tmpl::size_t<I + 1>>::value + 1 >=
          tmpl::at<IndexList, tmpl::size_t<I + 1>>::dim)>::
          template type<
              IndexList,
              tmpl::replace_at<TensorIndex, tmpl::size_t<I>, tmpl::size_t<0>>,
              I + 1>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Increment a tensor index
 *
 * \details
 * Given a tensor index `TensorIndex` and a list of \ref SpacetimeIndex
 * "TensorIndex"'s, `IndexList`, increment the tensor index to the next
 * component. Incrementing is done by incremeting the first (left-most) index
 * first, once it reaches the dimensionality of that index, the next index is
 * increment. That is, for a tensor \f$T_{abc}\f$ where all indices are
 * 3-dimensional, incrementing the index \f$(1, 3, 0)\f$ results in the index
 * \f$(2, 3, 0)\f$, and incrementing the index \f$(3, 3, 0)\f$ results
 * \f$(0, 0, 1)\f$.
 *
 * For a ::typelist of \ref SpacetimeIndex "TensorIndexType"'s
 * `IndexList`
 * and a tensor index (::typelist of std::integral_constant) `TensorIndex`,
 * \code
 * using result = IncrementTensorIndex<IndexList, TensorIndex>;
 * \endcode
 * \metareturns
 * ::typelist<std::integral_constant...>
 *
 * \semantics
 * Let `IndexList` by a ::typelist
 * \code
 * typelist<SpacetimeTensorIndex<3, UpLo::Lo, Frame::Grid>,
 *          SpacetimeTensorIndex<3, UpLo::Lo, Frame::Grid>,
 *          SpacetimeTensorIndex<3, UpLo::Lo, Frame::Grid>>;
 * \endcode
 * and `TensorIndex` be
 * `tmpl::integral_list<int, 3, 3, 0>`, then
 * \code
 * using result = tmpl::integral_list<int, 0, 0, 1>;
 * \endcode
 */
template <typename IndexList, typename TensorIndex>
using increment_tensor_index = typename detail::increment_tensor_index_impl<(
    tmpl::at<TensorIndex, tmpl::size_t<0>>::value + 1 >=
    tmpl::at<IndexList, tmpl::size_t<0>>::dim)>::template type<IndexList,
                                                               TensorIndex, 0>;

// typename detail::increment_tensor_index_impl<IndexList, TensorIndex,
// 0>::type;

namespace detail {
template <bool>
struct tensor_index_to_swap_impl;

// Rank == Offset case
template <>
struct tensor_index_to_swap_impl<true> {
  template <typename Symm, typename TensorIndex, typename Rank, typename I,
            typename Offset>
  using type = I;
};

// Rank != Offset case
template <>
struct tensor_index_to_swap_impl<false> {
  template <typename Symm, typename TensorIndex, typename Rank, typename I,
            typename Offset>
  using type = tmpl::conditional_t<
      (tmpl::and_<
          tmpl::less<tmpl::at<TensorIndex, I>, tmpl::at<TensorIndex, Offset>>,
          tmpl::equal_to<tmpl::at<Symm, I>, tmpl::at<Symm, Offset>>>::value),
      Offset,
      typename tensor_index_to_swap_impl<Rank::value == Offset::value + 1>::
          template type<Symm, TensorIndex, Rank, I,
                        tmpl::size_t<Offset::value + 1>>>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 */
template <typename Symm, typename TensorIndex, size_t Rank, size_t I,
          size_t Offset>
using tensor_index_to_swap = typename detail::tensor_index_to_swap_impl<
    Rank == Offset + 1>::template type<Symm, TensorIndex, tmpl::size_t<Rank>,
                                       tmpl::size_t<I>, tmpl::size_t<Offset>>;

namespace detail {
template <bool>
struct canonicalize_tensor_index_impl;

template <>
struct canonicalize_tensor_index_impl<false> {
  template <typename Symm, typename IndexList, typename TensorIndex,
            size_t Rank, size_t I>
  using type = TensorIndex;
};

template <>
struct canonicalize_tensor_index_impl<true> {
  template <typename Symm, typename IndexList, typename TensorIndex,
            size_t Rank, size_t I>
  using type =
      typename canonicalize_tensor_index_impl<(Rank > I + 1)>::template type<
          Symm, IndexList,
          tmpl::swap_at<TensorIndex, tmpl::size_t<I>,
                        ::TensorMetafunctions::tensor_index_to_swap<
                            Symm, TensorIndex, Rank, I, I>>,
          Rank, I + 1>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 */
template <typename Symm, typename IndexList, typename TensorIndex>
using canonicalize_tensor_index =
    typename detail::canonicalize_tensor_index_impl<(
        tmpl::size<IndexList>::value >
        0)>::template type<Symm, IndexList, TensorIndex,
                           tmpl::size<IndexList>::value, 0>;

namespace detail {
template <bool IndicesRemaining>
struct compute_collapsed_index_impl;

template <>
struct compute_collapsed_index_impl<false> {
  template <typename TensorIndex, typename IndexList>
  using type = tmpl::front<TensorIndex>;
};

template <>
struct compute_collapsed_index_impl<true> {
  template <typename TensorIndex, typename IndexList>
  using type =
      tmpl::plus<tmpl::front<TensorIndex>,
                 tmpl::times<tmpl::front<IndexList>,
                             typename compute_collapsed_index_impl<(
                                 tmpl::size<TensorIndex>::value > 2)>::
                                 template type<tmpl::pop_front<TensorIndex>,
                                               tmpl::pop_front<IndexList>>>>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 *
 * \requires `is_a<typelist, IndexList>::%value` is true, `tt::is_a<typelist,
 * TensorIndex>::%value` is true, and `tmpl::all<TensorIndex,
 * is_a<std::integral_constant, tmpl::_1>>::%value` is true
 */
template <typename TensorIndex, typename IndexList>
using compute_collapsed_index = typename detail::compute_collapsed_index_impl<(
    tmpl::size<TensorIndex>::value > 1)>::template type<TensorIndex, IndexList>;

namespace detail {
template <bool>
struct update_collapsed_to_storage_using_canonical_tensor_index_impl;

template <>
struct update_collapsed_to_storage_using_canonical_tensor_index_impl<false> {
  template <typename IndexList, typename CanonicalTensorIndex,
            typename TensorIndex, typename CollapsedToStorageList, typename I,
            typename Count>
  using type = tmpl::replace_at<
      CollapsedToStorageList, I,
      tmpl::at<CollapsedToStorageList,
               compute_collapsed_index<CanonicalTensorIndex, IndexList>>>;
};

template <>
struct update_collapsed_to_storage_using_canonical_tensor_index_impl<true> {
  template <typename IndexList, typename CanonicalTensorIndex,
            typename TensorIndex, typename CollapsedToStorageList, typename I,
            typename Count>
  using type = tmpl::replace_at<CollapsedToStorageList, I, Count>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \note Cannot use `std::conditional_t` because `typelist<>` will fail
 */
template <typename IndexList, typename CanonicalTensorIndex,
          typename TensorIndex, typename CollapsedToStorageList, typename I,
          typename Count>
using update_collapsed_to_storage_using_canonical_tensor_index =
    typename detail::
        update_collapsed_to_storage_using_canonical_tensor_index_impl<
            std::is_same<CanonicalTensorIndex, TensorIndex>::value>::
            template type<IndexList, CanonicalTensorIndex, TensorIndex,
                          CollapsedToStorageList, I, Count>;

/*!
 * \ingroup TensorGroup
 * \brief given a `TensorIndex` checks if it is the canonical form, if so
 * returns `tmpl::plus<Count, tmpl::size_t<1>>` otherwise returns `Count`
 */
template <typename CanonicalizeTensorIndex, typename TI, typename Count>
using increase_count =
    tmpl::conditional_t<std::is_same<CanonicalizeTensorIndex, TI>::value,
                        tmpl::size_t<Count::value + 1>, Count>;

namespace detail {
template <bool>
struct compute_collapsed_to_storage_impl;

template <>
struct compute_collapsed_to_storage_impl<false> {
  template <typename CollapsedToStorageList, typename IndexList, typename Symm,
            typename TensorIndex, typename I, typename Count, typename NumComps>
  using type = update_collapsed_to_storage_using_canonical_tensor_index<
      IndexList, canonicalize_tensor_index<Symm, IndexList, TensorIndex>,
      TensorIndex, CollapsedToStorageList, I, Count>;
};
template <>
struct compute_collapsed_to_storage_impl<true> {
  template <typename CollapsedToStorageList, typename IndexList, typename Symm,
            typename TensorIndex, typename I, typename Count, typename NumComps>
  using type = typename compute_collapsed_to_storage_impl<(
      I::value + 1 < NumComps::value - 1)>::
      template type<
          update_collapsed_to_storage_using_canonical_tensor_index<
              IndexList,
              canonicalize_tensor_index<Symm, IndexList, TensorIndex>,
              TensorIndex, CollapsedToStorageList, I, Count>,
          IndexList, Symm,
          ::TensorMetafunctions::increment_tensor_index<IndexList, TensorIndex>,
          tmpl::size_t<I::value + 1>,
          ::TensorMetafunctions::increase_count<
              canonicalize_tensor_index<Symm, IndexList, TensorIndex>,
              TensorIndex, Count>,
          NumComps>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Compute the collapse index to storage index ::typelist
 */
template <typename IndexList, typename Symm, typename NumComps>
using compute_collapsed_to_storage = typename detail::
    compute_collapsed_to_storage_impl<(0 < NumComps::value - 1)>::template type<
        tmpl::filled_list<tmpl::size_t<0>, NumComps::value>, IndexList, Symm,
        tmpl::filled_list<tmpl::size_t<0>, tmpl::size<IndexList>::value>,
        tmpl::size_t<0>, tmpl::size_t<0>, NumComps>;

namespace detail {
template <unsigned>
struct compute_storage_to_tensor_impl;

template <>
struct compute_storage_to_tensor_impl<0> {
  template <typename Symm, typename IndexList, typename CollapsedToStorageList,
            typename TensorIndex, typename StorageToTensorList>
  using type =
      tmpl::replace_at<StorageToTensorList, tmpl::front<CollapsedToStorageList>,
                       TensorIndex>;
};

// Case where
// std::is_same_v<CollapsedToStorageList, tmpl::list<tmpl::size_t<0>>> == true
template <>
struct compute_storage_to_tensor_impl<1> {
  template <typename Symm, typename IndexList, typename CollapsedToStorageList,
            typename TensorIndex, typename StorageToTensorList>
  using type = tmpl::list<tmpl::integral_list<std::size_t, 0>>;
};

// Case where:
// tmpl::size<CollapsedToStorageList>::value > 1 == true
template <>
struct compute_storage_to_tensor_impl<2> {
  template <typename Symm, typename IndexList, typename CollapsedToStorageList,
            typename TensorIndex, typename StorageToTensorList>
  using type = typename compute_storage_to_tensor_impl<
      (tmpl::size<CollapsedToStorageList>::value > 2)
          ? 2
          : ((tmpl::size<CollapsedToStorageList>::value == 2 and
              std::is_same<tmpl::back<CollapsedToStorageList>,
                           tmpl::list<tmpl::size_t<0>>>::value)
                 ? 1
                 : 0)>::
      template type<
          Symm, IndexList, tmpl::pop_front<CollapsedToStorageList>,
          ::TensorMetafunctions::increment_tensor_index<IndexList, TensorIndex>,
          tmpl::replace_at<
              StorageToTensorList, tmpl::front<CollapsedToStorageList>,
              canonicalize_tensor_index<Symm, IndexList, TensorIndex>>>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Compute a ::typelist holding the tensor index for each storage index
 */
template <typename Symm, typename IndexList, typename CollapsedToStorageList,
          typename NumIndComps>
using compute_storage_to_tensor =
    typename detail::compute_storage_to_tensor_impl<
        (tmpl::size<CollapsedToStorageList>::value > 1)
            ? 2
            : (tmpl::size<CollapsedToStorageList>::value == 2 and
               std::is_same<tmpl::back<CollapsedToStorageList>,
                            tmpl::list<tmpl::size_t<0>>>::value)
                  ? 1
                  : 0>::
        template type<
            Symm, IndexList, CollapsedToStorageList,
            tmpl::filled_list<tmpl::size_t<0>, tmpl::size<IndexList>::value>,
            tmpl::filled_list<tmpl::size_t<0>, NumIndComps::value>>;

namespace detail {
template <bool>
struct compute_multiplicity_impl;

template <>
struct compute_multiplicity_impl<false> {
  template <typename CollapsedToStorageList, typename MultiplicityList>
  using type = tmpl::replace_at<
      MultiplicityList, tmpl::front<CollapsedToStorageList>,
      tmpl::size_t<tmpl::at<MultiplicityList,
                            tmpl::front<CollapsedToStorageList>>::value +
                   1>>;
};

template <>
struct compute_multiplicity_impl<true> {
  template <typename CollapsedToStorageList, typename MultiplicityList>
  using type = typename compute_multiplicity_impl<(
      tmpl::size<CollapsedToStorageList>::value > 2)>::
      template type<
          tmpl::pop_front<CollapsedToStorageList>,
          tmpl::replace_at<
              MultiplicityList, tmpl::front<CollapsedToStorageList>,
              tmpl::size_t<
                  tmpl::at<MultiplicityList,
                           tmpl::front<CollapsedToStorageList>>::value +
                  1>>>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Compute ::typelist of the multiplicity of each storage index of a
 * Tensor
 *
 * \details
 * The multiplicity of a storage index is the number of tensor components it
 * represents. For example, if a tensor is symmetric then the multiplicity
 * for the storage index's representing symmetric terms have a multiplicity
 * of 2. Specifically, consider a rank-2 symmetric tensor \f$T_{ab}\f$, let
 * the storage of the \f$T_{02}\f$ component be `N`, for example. Then, since
 * \f$T_{20} = T_{02}\f$
 * \code
 * using result = tmpl::at<ComputeMultiplicity<ColToStor, NumIndComps>,
 *                          tmpl::size_t<5>> = tmpl::size_t<2>;
 * \endcode
 */
template <typename CollapsedToStorageList, typename NumIndComps>
using compute_multiplicity = typename detail::compute_multiplicity_impl<(
    tmpl::size<CollapsedToStorageList>::value >
    1)>::template type<CollapsedToStorageList,
                       tmpl::filled_list<tmpl::size_t<0>, NumIndComps::value>>;

namespace detail {
template <unsigned>
struct check_index_symmetry_impl;

// empty typelist, done recursion
template <>
struct check_index_symmetry_impl<0> {
  template <typename Symm, typename IndexList, typename IndexSymm>
  using type = std::true_type;
};

// found incorrect symmetric index
template <>
struct check_index_symmetry_impl<1> {
  template <typename Symm, typename IndexList, typename IndexSymm>
  using type = std::false_type;
};

// recurse the list
template <>
struct check_index_symmetry_impl<2> {
  template <typename Symm, typename IndexList, typename IndexSymm>
  using type = typename check_index_symmetry_impl<
      tmpl::has_key<IndexSymm, tmpl::front<Symm>>::value and
              not std::is_same<tmpl::front<IndexList>,
                               tmpl::at<IndexSymm, tmpl::front<Symm>>>::value
          ? 1
          : tmpl::size<Symm>::value == 1 ? 0 : 2>::
      template type<
          tmpl::pop_front<Symm>, tmpl::pop_front<IndexList>,
          tmpl::insert<IndexSymm,
                       tmpl::pair<tmpl::front<Symm>, tmpl::front<IndexList>>>>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Check that each of symmetric indices is in the same frame and have the
 * same dimensionality.
 */
template <typename Symm, typename IndexList>
using check_index_symmetry = typename detail::check_index_symmetry_impl<
    tmpl::size<Symm>::value == 0 ? 0 : 2>::template type<Symm, IndexList,
                                                         tmpl::map<>>;
template <typename Symm, typename IndexList>
constexpr bool check_index_symmetry_v =
    check_index_symmetry<Symm, IndexList>::value;

/*!
 * \ingroup TensorGroup
 * \brief Add a spatial index to the from of a Tensor
 *
 * \tparam Tensor_t the tensor type to prepend
 * \tparam VolumeDim the volume dimension of the tensor index to prepend
 * \tparam Fr the ::Frame of the tensor index to prepend
 */
template <typename Tensor_t, std::size_t VolumeDim, UpLo Ul,
          typename Fr = Frame::Grid>
using prepend_spatial_index = Tensor<
    typename Tensor_t::type,
    tmpl::push_front<
        typename Tensor_t::symmetry,
        tmpl::int32_t<
            1 + tmpl::fold<typename Tensor_t::symmetry, tmpl::int32_t<0>,
                           tmpl::max<tmpl::_state, tmpl::_element>>::value>>,
    tmpl::push_front<typename Tensor_t::index_list,
                     SpatialIndex<VolumeDim, Ul, Fr>>>;

/*!
 * \ingroup TensorGroup
 * \brief Add a spacetime index to the from of a Tensor
 *
 * \tparam Tensor_t the tensor type to prepend
 * \tparam VolumeDim the volume dimension of the tensor index to prepend
 * \tparam Fr the ::Frame of the tensor index to prepend
 */
template <typename Tensor_t, std::size_t VolumeDim, UpLo Ul,
          typename Fr = Frame::Grid>
using prepend_spacetime_index = Tensor<
    typename Tensor_t::type,
    tmpl::push_front<
        typename Tensor_t::symmetry,
        tmpl::int32_t<
            1 + tmpl::fold<typename Tensor_t::symmetry, tmpl::int32_t<0>,
                           tmpl::max<tmpl::_state, tmpl::_element>>::value>>,
    tmpl::push_front<typename Tensor_t::index_list,
                     SpacetimeIndex<VolumeDim, Ul, Fr>>>;
}  // namespace TensorMetafunctions
