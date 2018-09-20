// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines metafunctions used by Tensor

#pragma once

#include "DataStructures/Tensor/IndexType.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;
/// \endcond

/// \ingroup TensorGroup
/// Contains all metafunctions related to Tensor manipulations
namespace TensorMetafunctions {
namespace detail {
template <unsigned>
struct check_index_symmetry_impl;
// empty typelist or only had a vector to start with
template <>
struct check_index_symmetry_impl<0> {
  template <typename...>
  using f = std::true_type;
};

// found incorrect symmetric index
template <>
struct check_index_symmetry_impl<1> {
  template <typename...>
  using f = std::false_type;
};

// recurse the list
template <>
struct check_index_symmetry_impl<2> {
  template <typename Symm, typename IndexSymm, typename Index0,
            typename... IndexPack>
  using f = typename check_index_symmetry_impl<
      tmpl::has_key<IndexSymm, tmpl::front<Symm>>::value and
              not std::is_same<Index0,
                               tmpl::at<IndexSymm, tmpl::front<Symm>>>::value
          ? 1
          : tmpl::size<Symm>::value == 1 ? 0 : 2>::
      template f<tmpl::pop_front<Symm>,
                 tmpl::insert<IndexSymm, tmpl::pair<tmpl::front<Symm>, Index0>>,
                 IndexPack...>;
};
}  // namespace detail

/*!
 * \ingroup TensorGroup
 * \brief Check that each of symmetric indices is in the same frame and have the
 * same dimensionality.
 */
template <typename Symm, typename... IndexPack>
using check_index_symmetry = typename detail::check_index_symmetry_impl<
    tmpl::size<Symm>::value == 0 or tmpl::size<Symm>::value == 1 ? 0 : 2>::
    template f<Symm, tmpl::map<>, IndexPack...>;
template <typename Symm, typename... IndexPack>
constexpr bool check_index_symmetry_v =
    check_index_symmetry<Symm, IndexPack...>::value;

/*!
 * \ingroup TensorGroup
 * \brief Add a spatial index to the front of a Tensor
 *
 * \tparam Tensor the tensor type to which the new index is prepended
 * \tparam VolumeDim the volume dimension of the tensor index to prepend
 * \tparam Fr the ::Frame of the tensor index to prepend
 */
template <typename Tensor, std::size_t VolumeDim, UpLo Ul,
          typename Fr = Frame::Grid>
using prepend_spatial_index = ::Tensor<
    typename Tensor::type,
    tmpl::push_front<
        typename Tensor::symmetry,
        tmpl::int32_t<
            1 + tmpl::fold<typename Tensor::symmetry, tmpl::int32_t<0>,
                           tmpl::max<tmpl::_state, tmpl::_element>>::value>>,
    tmpl::push_front<typename Tensor::index_list,
                     SpatialIndex<VolumeDim, Ul, Fr>>>;

/*!
 * \ingroup TensorGroup
 * \brief Add a spacetime index to the front of a Tensor
 *
 * \tparam Tensor the tensor type to which the new index is prepended
 * \tparam VolumeDim the volume dimension of the tensor index to prepend
 * \tparam Fr the ::Frame of the tensor index to prepend
 */
template <typename Tensor, std::size_t VolumeDim, UpLo Ul,
          typename Fr = Frame::Grid>
using prepend_spacetime_index = ::Tensor<
    typename Tensor::type,
    tmpl::push_front<
        typename Tensor::symmetry,
        tmpl::int32_t<
            1 + tmpl::fold<typename Tensor::symmetry, tmpl::int32_t<0>,
                           tmpl::max<tmpl::_state, tmpl::_element>>::value>>,
    tmpl::push_front<typename Tensor::index_list,
                     SpacetimeIndex<VolumeDim, Ul, Fr>>>;

/// \ingroup TensorGroup
/// \brief remove the first index of a tensor
/// \tparam Tensor the tensor type whose first index is removed
template <typename Tensor>
using remove_first_index =
    ::Tensor<typename Tensor::type, tmpl::pop_front<typename Tensor::symmetry>,
             tmpl::pop_front<typename Tensor::index_list>>;

/// \ingroup TensorGroup
/// \brief Swap the data type of a tensor for a new type
/// \tparam NewType the new data type
/// \tparam Tensor the tensor from which to keep symmetry and index information
template <typename NewType, typename Tensor>
using swap_type =
    ::Tensor<NewType, typename Tensor::symmetry, typename Tensor::index_list>;
}  // namespace TensorMetafunctions
