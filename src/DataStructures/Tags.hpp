// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"

/// \cond
class DataVector;
class ModalVector;
/// \endcond

namespace Tags {
/// Given the `Tag` holding a `Tensor<DataVector, ...>`, swap the
/// `DataVector` with a `double`.
template <typename Tag>
struct Mean : db::SimpleTag, db::PrefixTag {
  using tag = Tag;
  static_assert(cpp17::is_same_v<typename Tag::type::type, DataVector>,
                "The Mean tag should only be used on tags that hold "
                "Tensors of DataVectors");
  using type = TensorMetafunctions::swap_type<double, typename Tag::type>;
  static std::string name() noexcept { return "Mean(" + Tag::name() + ")"; }
};

/// Given the `NodalTag` holding a `Tensor<DataVector, ...>`, swap the
/// `DataVector` with a `ModalVector`.
template <typename NodalTag>
struct Modal : db::SimpleTag, db::PrefixTag {
  using tag = NodalTag;
  static_assert(cpp17::is_same_v<typename NodalTag::type::type, DataVector>,
                "The Modal tag should only be used on tags that hold "
                "Tensors of DataVectors");
  using type =
      TensorMetafunctions::swap_type<ModalVector, typename NodalTag::type>;
  static std::string name() noexcept {
    return "Modal(" + NodalTag::name() + ")";
  }
};
}  // namespace Tags
