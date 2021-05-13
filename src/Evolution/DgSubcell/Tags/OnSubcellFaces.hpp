// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/VariablesTag.hpp"

/// \cond
template <typename X, typename Symm, typename IndexList>
class Tensor;
template <typename TagsList>
class Variables;
/// \endcond

namespace evolution::dg::subcell::Tags {
/// Mark a tag as the being on the subcell faces
template <typename Tag, size_t Dim>
struct OnSubcellFaces : db::PrefixTag, db::SimpleTag {
  using type = std::array<typename Tag::type, Dim>;
  using tag = Tag;
};
}  // namespace evolution::dg::subcell::Tags
