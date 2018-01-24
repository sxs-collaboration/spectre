// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define prefixes for DataBox tags

#pragma once

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a time derivative
template <typename Tag>
struct dt : db::DataBoxPrefix {
  static constexpr db::DataBoxString label = "dt";
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
            Requires<tt::is_a_v<Tensor, db::item_type<Tag>>>>
    : db::DataBoxPrefix {
  using type = TensorMetafunctions::prepend_spatial_index<
      db::item_type<Tag>, VolumeDim::value, UpLo::Up, Fr>;
  using tag = Tag;
  static constexpr db::DataBoxString label = "Flux";
};

template <typename Tag, typename VolumeDim, typename Fr>
struct Flux<Tag, VolumeDim, Fr,
            Requires<tt::is_a_v<::Variables, db::item_type<Tag>>>>
    : db::DataBoxPrefix {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static constexpr db::DataBoxString label = "Flux";
};
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating an unnormalized boundary normal vector
/// dotted into the flux
template <typename Tag>
struct NormalDotFlux : db::DataBoxPrefix {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static constexpr db::DataBoxString label = "NormalDotFlux";
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating an unnormalized boundary normal vector
/// dotted into the numerical flux
template <typename Tag>
struct NormalDotNumericalFlux : db::DataBoxPrefix {
  using type = db::item_type<Tag>;
  using tag = Tag;
  static constexpr db::DataBoxString label = "NormalDotNumericalFlux";
};
}  // namespace Tags
