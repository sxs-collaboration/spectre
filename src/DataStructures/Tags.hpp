// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"

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
  static_assert(std::is_same_v<typename Tag::type::type, DataVector>,
                "The Mean tag should only be used on tags that hold "
                "Tensors of DataVectors");
  using type = TensorMetafunctions::swap_type<double, typename Tag::type>;
};

/// Given the `NodalTag` holding a `Tensor<DataVector, ...>`, swap the
/// `DataVector` with a `ModalVector`.
template <typename NodalTag>
struct Modal : db::SimpleTag, db::PrefixTag {
  using tag = NodalTag;
  static_assert(std::is_same_v<typename NodalTag::type::type, DataVector>,
                "The Modal tag should only be used on tags that hold "
                "Tensors of DataVectors");
  using type =
      TensorMetafunctions::swap_type<ModalVector, typename NodalTag::type>;
};

/// Given a Tag with a `type` of `Tensor<VectorType, ...>`, acts as a new
/// version of the tag with `type` of `Tensor<SpinWeighted<VectorType,
/// SpinConstant::value>, ...>`, which is the preferred tensor type associated
/// with spin-weighted quantities. Here, `SpinConstant` must be a
/// `std::integral_constant` or similar type wrapper for a compile-time
/// constant, in order to work properly with DataBox utilities.
/// \note There are restrictions of which vector types can be wrapped
/// by SpinWeighted in a Tensor, so some `Tag`s that have valid `type`s may
/// give rise to compilation errors when used as `Tags::SpinWeighted<Tag,
/// SpinConstant>`. If you find such trouble, consult the whitelist of possible
/// Tensor storage types in `Tensor.hpp`.
template <typename Tag, typename SpinConstant>
struct SpinWeighted : db::PrefixTag, db::SimpleTag {
  static_assert(not is_any_spin_weighted_v<typename Tag::type::type>,
                "The SpinWeighted tag should only be used to create a "
                "spin-weighted version of a non-spin-weighted tag. The "
                "provided tag already has a spin-weighted type");
  using type = TensorMetafunctions::swap_type<
      ::SpinWeighted<typename Tag::type::type, SpinConstant::value>,
      typename Tag::type>;
  using tag = Tag;
  static std::string name() noexcept {
    return "SpinWeighted(" + db::tag_name<Tag>() + ", " +
           std::to_string(SpinConstant::value) + ")";
  }
};

namespace detail {
template <typename LhsType, typename RhsType>
using product_t =
    Scalar<::SpinWeighted<ComplexDataVector,
                          LhsType::type::spin + RhsType::type::spin>>;
}  // namespace detail

/// A prefix tag representing the product of two other tags. Note that if
/// non-spin-weighted types are needed in this tag, the type alias
/// `detail::product_t` should be generalized to give a reasonable type for
/// those cases, or template specializations should be constructed for this tag.
template <typename LhsTag, typename RhsTag>
struct Multiplies : db::PrefixTag, db::SimpleTag {
  using type = detail::product_t<db::item_type<LhsTag>, db::item_type<RhsTag>>;
  using tag = LhsTag;
  static std::string name() noexcept {
    return "Multiplies(" + db::tag_name<LhsTag>() + ", " +
           db::tag_name<RhsTag>() + ")";
  }
};

}  // namespace Tags
