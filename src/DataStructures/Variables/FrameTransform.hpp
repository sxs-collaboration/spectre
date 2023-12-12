// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/EagerMath/FrameTransform.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/Gsl.hpp"

namespace transform {

/// Tags to represent the result of frame-transforming Variables
namespace Tags {
/// The `Tag` with the first index transformed to a different frame
template <typename Tag, typename FirstIndexFrame>
struct TransformedFirstIndex : db::SimpleTag {
  using type = TensorMetafunctions::prepend_spatial_index<
      TensorMetafunctions::remove_first_index<typename Tag::type>,
      tmpl::front<typename Tag::type::index_list>::dim, UpLo::Up,
      FirstIndexFrame>;
};
}  // namespace Tags

/// @{
/// Transforms the first index of all tensors in the Variables to a different
/// frame
///
/// See single-Tensor overload for details.
template <typename... ResultTags, typename... InputTags, size_t Dim,
          typename SourceFrame, typename TargetFrame>
void first_index_to_different_frame(
    const gsl::not_null<Variables<tmpl::list<ResultTags...>>*> result,
    const Variables<tmpl::list<InputTags...>>& input,
    const InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>&
        inv_jacobian) {
  EXPAND_PACK_LEFT_TO_RIGHT(
      first_index_to_different_frame(make_not_null(&get<ResultTags>(*result)),
                                     get<InputTags>(input), inv_jacobian));
}

template <typename... InputTags, size_t Dim, typename SourceFrame,
          typename TargetFrame,
          typename ResultVars = Variables<tmpl::list<
              Tags::TransformedFirstIndex<InputTags, SourceFrame>...>>>
ResultVars first_index_to_different_frame(
    const Variables<tmpl::list<InputTags...>>& input,
    const InverseJacobian<DataVector, Dim, SourceFrame, TargetFrame>&
        inv_jacobian) {
  ResultVars result{input.number_of_grid_points()};
  first_index_to_different_frame(make_not_null(&result), input, inv_jacobian);
  return result;
}
/// @}

}  // namespace transform
