// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines functions and tags for taking a divergence.

#pragma once

#include <cstddef>
#include <cstdint>
#include <tuple>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
template <size_t Dim>
class Index;
template <typename DataType, typename Symm, typename IndexList>
class Tensor;
template <typename TagsList>
class Variables;

namespace Tags {
template <size_t Dim>
struct Extents;
template <class TagList>
struct Variables;

namespace Tags_detail {
template <typename T, typename S, typename = std::nullptr_t>
struct div_impl;
}  // namespace Tags_detail
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the divergence
///
/// There are two variants of this tag that change how the derivatives are
/// computed depending on which is chosen. The simplest is the non-compute
/// item versions for a Tensor tag. This takes two template parameters:
/// 1. The tag to wrap
/// 2. The frame in which the divergence was taken
///
/// The second variant of the divergence tag is a compute item that computes
/// the divergence in a Frame that is not the logical frame. This compute
/// item is used by specifying the the list of flux tags to be differentiated,
/// and the Tag for the inverse Jacobian between the logical frame and the frame
/// in which the divergence is taken.
template <typename T, typename S>
struct div : Tags_detail::div_impl<T, S> {};
}  // namespace Tags

/// \ingroup NumericalAlgorithmsGroup
/// \brief Compute the (Euclidean) divergence of fluxes
///
/// \return a `Variables` with the same structure as `F` and each tag in
/// `FluxTags` wrapped with a `Tags::div`.
template <typename FluxTags, size_t Dim, typename DerivativeFrame>
Variables<db::wrap_tags_in<Tags::div, FluxTags, DerivativeFrame>> divergence(
    const Variables<FluxTags>& F, const Index<Dim>& extents,
    const InverseJacobian<DataVector, Dim, Frame::Logical, DerivativeFrame>&
        inverse_jacobian) noexcept;

namespace Tags {
namespace Tags_detail {
template <typename Tag, typename Frame>
struct div_impl<Tag, Frame, Requires<tt::is_a_v<Tensor, db::item_type<Tag>>>>
    : db::DataBoxPrefix {
  static_assert(
      cpp17::is_same_v<Frame,
                       std::tuple_element_t<
                           0, decltype(db::item_type<Tag>::index_frames())>>,
      "the first index is not in the specified Frame");
  using type = TensorMetafunctions::remove_first_index<db::item_type<Tag>>;
  using tag = Tag;
  static constexpr db::DataBoxString label = "div";
};

template <typename... FluxTags, typename InverseJacobianTag>
struct div_impl<tmpl::list<FluxTags...>, InverseJacobianTag,
                Requires<tt::is_a_v<Tensor, db::item_type<InverseJacobianTag>>>>
    : db::ComputeItemTag {
 private:
  using derivative_frame_index =
      tmpl::back<typename db::item_type<InverseJacobianTag>::index_list>;
  using flux_tags = tmpl::list<FluxTags...>;

 public:
  static constexpr db::DataBoxString label = "div";
  static constexpr auto function =
      divergence<flux_tags, derivative_frame_index::dim,
                 typename derivative_frame_index::Frame>;
  using argument_tags = tmpl::list<Tags::Variables<flux_tags>,
                                   Tags::Extents<derivative_frame_index::dim>,
                                   InverseJacobianTag>;
};
}  // namespace Tags_detail
}  // namespace Tags
