// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define prefixes for DataBox tags

#pragma once

#include <cstddef>
#include <type_traits>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <class>
class Variables;
/// \endcond

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a time derivative
///
/// \snippet Test_DataBoxPrefixes.cpp dt_name
template <typename Tag>
struct dt : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/*!
 * \ingroup DataBoxTagsGroup
 * \brief Prefix indicating spatial derivatives
 *
 * Prefix indicating the spatial derivatives of a Tensor.
 *
 * \tparam Tag The tag to wrap
 * \tparam Dim The volume dim as a type (e.g. `tmpl::size_t<Dim>`)
 * \tparam Frame The frame of the derivative index
 *
 * \see Tags::DerivCompute
 */
template <typename Tag, typename Dim, typename Frame>
struct deriv;

/// \cond
template <typename Tag, typename Dim, typename Frame>
  requires(tt::is_a_v<Tensor, typename Tag::type>)
struct deriv<Tag, Dim, Frame> : db::PrefixTag, db::SimpleTag {
  using type =
      TensorMetafunctions::prepend_spatial_index<typename Tag::type, Dim::value,
                                                 UpLo::Lo, Frame>;
  using tag = Tag;
};
/// \endcond

/*!
 * \ingroup DataBoxTagsGroup
 * \brief Prefix indicating symmetric second spatial derivatives
 *
 * Prefix indicating the symmetric second spatial derivatives of a Tensor.
 *
 * \tparam Tag The tag to wrap
 * \tparam Dim The volume dim as a type (e.g. `tmpl::size_t<Dim>`)
 * \tparam Frame The frame of the derivative index
 *
 * \see Tags::DerivCompute
 */
template <typename Tag, typename Dim, typename Frame>
struct second_deriv;

/// \cond
template <typename Tag, typename Dim, typename Frame>
  requires(tt::is_a_v<Tensor, typename Tag::type>)
struct second_deriv<Tag, Dim, Frame> : db::PrefixTag, db::SimpleTag {
  using type = TensorMetafunctions::prepend_two_symmetric_spatial_indices<
      typename Tag::type, Dim::value, UpLo::Lo, Frame>;
  using tag = Tag;
};
/// \endcond

/*!
 * \ingroup DataBoxTagsGroup
 * \brief Prefix indicating spacetime derivatives
 *
 * Prefix indicating the spacetime derivatives of a Tensor or that a Variables
 * contains spatial derivatives of Tensors.
 *
 * \tparam Tag The tag to wrap
 * \tparam Dim The volume dim as a type (e.g. `tmpl::size_t<Dim>`)
 * \tparam Frame The frame of the derivative index
 */
template <typename Tag, typename Dim, typename Frame>
struct spacetime_deriv;

/// \cond
template <typename Tag, typename Dim, typename Frame>
  requires(tt::is_a_v<Tensor, typename Tag::type>)
struct spacetime_deriv<Tag, Dim, Frame> : db::PrefixTag, db::SimpleTag {
  using type =
      TensorMetafunctions::prepend_spacetime_index<typename Tag::type,
                                                   Dim::value, UpLo::Lo, Frame>;
  using tag = Tag;
};
/// \endcond

/*!
 * \ingroup DataBoxTagsGroup
 * \brief Prefix indicating the spatial covariant derivative of a tensor.
 *
 * Prefix indicating the first covariant derivative of a tensor with respect
 * to the spatial metric.
 *
 * \tparam Tag The tag to wrap
 * \tparam Dim The volume dim as a type (e.g. `tmpl::size_t<Dim>`)
 * \tparam Frame The frame of the derivative index
 *
 * \snippet Test_DataBoxPrefixes.cpp covariant_deriv_name
 */
template <typename Tag, typename Dim, typename Frame>
struct covariant_deriv;

/// \cond
template <typename Tag, typename Dim, typename Frame>
  requires(tt::is_a_v<Tensor, typename Tag::type>)
struct covariant_deriv<Tag, Dim, Frame> : db::PrefixTag, db::SimpleTag {
  using type =
      TensorMetafunctions::prepend_spatial_index<typename Tag::type, Dim::value,
                                                 UpLo::Lo, Frame>;
  using tag = Tag;
};
/// \endcond

/*!
 * \ingroup DataBoxTagsGroup
 * \brief Prefix indicating the second spatial covariant derivative of a tensor.
 *
 * Prefix indicating the second covariant derivative of a tensor with respect
 * to the spatial metric.
 *
 * \tparam Tag The tag to wrap
 * \tparam Dim The volume dim as a type (e.g. `tmpl::size_t<Dim>`)
 * \tparam Frame The frame of the derivative index
 *
 * \note If ``Tag::type`` is ``Scalar``, then ``second_covariant_derivative``
 * will hold a symmetric tensor.
 *
 * \snippet Test_DataBoxPrefixes.cpp second_covariant_deriv_name
 */
template <typename Tag, typename Dim, typename Frame>
struct second_covariant_deriv;

/// \cond
template <typename Tag, typename Dim, typename Frame>
  requires(tt::is_a_v<Tensor, typename Tag::type> and
           not std::is_same_v<Scalar<typename Tag::type::type>,
                              typename Tag::type>)
struct second_covariant_deriv<Tag, Dim, Frame> : db::PrefixTag, db::SimpleTag {
  using type = TensorMetafunctions::prepend_spatial_index<
      TensorMetafunctions::prepend_spatial_index<typename Tag::type, Dim::value,
                                                 UpLo::Lo, Frame>,
      Dim::value, UpLo::Lo, Frame>;
  using tag = Tag;
};
/// \endcond

/// \cond
template <typename Tag, typename Dim, typename Frame>
  requires(std::is_same_v<Scalar<typename Tag::type::type>, typename Tag::type>)
struct second_covariant_deriv<Tag, Dim, Frame> : db::PrefixTag, db::SimpleTag {
  using type = TensorMetafunctions::prepend_two_symmetric_spatial_indices<
      typename Tag::type, Dim::value, UpLo::Lo, Frame>;
  using tag = Tag;
};
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a flux
///
/// \snippet Test_DataBoxPrefixes.cpp flux_name
template <typename Tag, typename VolumeDim, typename Fr>
struct Flux;

/// \cond
template <typename Tag, typename VolumeDim, typename Fr>
  requires(tt::is_a_v<Tensor, typename Tag::type>)
struct Flux<Tag, VolumeDim, Fr> : db::PrefixTag, db::SimpleTag {
  using type = TensorMetafunctions::prepend_spatial_index<
      typename Tag::type, VolumeDim::value, UpLo::Up, Fr>;
  using tag = Tag;
};

template <typename Tag, typename VolumeDim, typename Fr>
  requires(tt::is_a_v<::Variables, typename Tag::type>)
struct Flux<Tag, VolumeDim, Fr> : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a source term
///
/// \snippet Test_DataBoxPrefixes.cpp source_name
template <typename Tag>
struct Source : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a source term that is independent of dynamic
/// variables
template <typename Tag>
struct FixedSource : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the initial value of a quantity
///
/// \snippet Test_DataBoxPrefixes.cpp initial_name
template <typename Tag>
struct Initial : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a boundary unit normal vector dotted into
/// the flux
///
/// \snippet Test_DataBoxPrefixes.cpp normal_dot_flux_name
template <typename Tag>
struct NormalDotFlux : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a boundary unit normal vector dotted into
/// the numerical flux
///
/// \snippet Test_DataBoxPrefixes.cpp normal_dot_numerical_flux_name
template <typename Tag>
struct NormalDotNumericalFlux : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the value a quantity took in the previous iteration
/// of the algorithm.
template <typename Tag>
struct Previous : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the value a quantity will take on the
/// next iteration of the algorithm.
///
/// \snippet Test_DataBoxPrefixes.cpp next_name
template <typename Tag>
struct Next : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

}  // namespace Tags
