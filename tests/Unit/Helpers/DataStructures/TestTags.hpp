// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

namespace VariablesTestTags_detail {
/// [simple_variables_tag]
template <typename VectorType>
struct tensor : db::SimpleTag {
  using type = tnsr::I<VectorType, 3, Frame::Grid>;
};
/// [simple_variables_tag]
template <typename VectorType>
struct scalar : db::SimpleTag {
  using type = Scalar<VectorType>;
};
template <typename VectorType>
struct scalar2 : db::SimpleTag {
  using type = Scalar<VectorType>;
};

/// [prefix_variables_tag]
template <class Tag>
struct Prefix0 : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
};
/// [prefix_variables_tag]

template <class Tag>
struct Prefix1 : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
};

template <class Tag>
struct Prefix2 : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
};

template <class Tag>
struct Prefix3 : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
};
}  // namespace VariablesTestTags_detail
