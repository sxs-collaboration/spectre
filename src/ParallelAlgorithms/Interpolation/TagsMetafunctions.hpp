// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines metafunctions for manipulating Tags that refer to Tensors

#pragma once

#include "DataStructures/Tensor/IndexType.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/TMPL.hpp"

namespace TensorMetafunctions {
/// \ingroup TensorGroup
/// \brief Replaces Tag with an equivalent Tag but in frame NewFrame
template <typename Tag, typename NewFrame>
struct replace_frame_in_tag {
  // Base definition, for tags with no template parameters,
  // like some scalars.
  using type = Tag;
};
// Specialization for tensors in GeneralizedHarmonic::Tags
template <template <size_t, typename> typename Tag, size_t Dim, typename Frame,
          typename NewFrame>
struct replace_frame_in_tag<Tag<Dim, Frame>, NewFrame> {
  using type = Tag<Dim, NewFrame>;
};
// Specialization for tensors in gr::Tags
template <template <size_t, typename, typename> typename Tag, size_t Dim,
          typename Frame, typename DataType, typename NewFrame>
struct replace_frame_in_tag<Tag<Dim, Frame, DataType>, NewFrame> {
  using type = Tag<Dim, NewFrame, DataType>;
};
// Specialization for scalars in gr::Tags (which have a DataType).
template <template <typename> typename Tag, typename DataType,
          typename NewFrame>
struct replace_frame_in_tag<Tag<DataType>, NewFrame> {
  using type = Tag<DataType>;
};
// Specialization for Tags::deriv<Tag> with Tag in GeneralizedHarmonic::Tags
template <template <size_t, typename> typename Tag, size_t Dim, typename Frame,
          typename NewFrame>
struct replace_frame_in_tag<
    ::Tags::deriv<Tag<Dim, Frame>, tmpl::size_t<Dim>, Frame>, NewFrame> {
  using type = ::Tags::deriv<Tag<Dim, NewFrame>, tmpl::size_t<Dim>, NewFrame>;
};

/// \ingroup TensorGroup
/// \brief Replaces Tag with an equivalent Tag but in frame NewFrame
template <typename Tag, typename NewFrame>
using replace_frame_in_tag_t =
    typename replace_frame_in_tag<Tag, NewFrame>::type;

/// \ingroup TensorGroup
/// \brief Replaces every Tag in Taglist with an equivalent Tag
/// but in frame NewFrame
template <typename TagList, typename NewFrame>
using replace_frame_in_taglist =
    tmpl::transform<TagList,
                    tmpl::bind<replace_frame_in_tag_t, tmpl::_1, NewFrame>>;
}  // namespace TensorMetafunctions
