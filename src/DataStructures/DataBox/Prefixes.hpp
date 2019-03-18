// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define prefixes for DataBox tags

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits.hpp"  // IWYU pragma: keep

/// \cond
template <class>
class Variables;  // IWYU pragma: keep
/// \endcond

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a time derivative
template <typename Tag>
struct dt : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "dt(" + Tag::name() + ")"; }
  using type = db::item_type<Tag>;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a flux
template <typename Tag, typename VolumeDim, typename Fr,
          typename = std::nullptr_t>
struct Flux;

/// \cond
template <typename Tag, typename VolumeDim, typename Fr>
struct Flux<Tag, VolumeDim, Fr,
            Requires<tt::is_a_v<Tensor, db::item_type<Tag>>>> : db::PrefixTag,
                                                                db::SimpleTag {
  using type = TensorMetafunctions::prepend_spatial_index<
      db::item_type<Tag>, VolumeDim::value, UpLo::Up, Fr>;
  using tag = Tag;
  static std::string name() noexcept { return "Flux"; }
};

template <typename Tag, typename VolumeDim, typename Fr>
struct Flux<Tag, VolumeDim, Fr,
            Requires<tt::is_a_v<::Variables, db::item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "Flux"; }
};
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a source term
///
/// \snippet Test_DataBoxPrefixes.cpp source_name
template <typename Tag>
struct Source : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "Source(" + Tag::name() + ")"; }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the initial value of a quantity
template <typename Tag>
struct Initial : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "Initial(" + Tag::name() + ")"; }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a boundary unit normal vector dotted into
/// the flux
template <typename Tag>
struct NormalDotFlux : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "NormalDotFlux"; }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a boundary unit normal vector dotted into
/// the numerical flux
template <typename Tag>
struct NormalDotNumericalFlux : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "NormalDotNumericalFlux"; }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the value a quantity will take on the
/// next iteration of the algorithm.
///
/// \snippet Test_DataBoxPrefixes.cpp next_name
template <typename Tag>
struct Next : db::PrefixTag, db::SimpleTag {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept { return "Next(" + Tag::name() + ")"; }
};
}  // namespace Tags
