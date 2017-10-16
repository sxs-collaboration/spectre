// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define prefixes for DataBox tags

#pragma once

#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace Tags {
/// \ingroup DataBoxTags
/// \brief prefix for a `VolumeDim` spatial derivatives with respect to the `Fr`
/// coordinate frame.
template <typename Tag, typename VolumeDim, typename Fr,
          typename = std::nullptr_t>
struct d;

template <typename Tag, typename VolumeDim, typename Fr>
struct d<Tag, VolumeDim, Fr, Requires<tt::is_a_v<Tensor, typename Tag::type>>>
    : db::DataBoxPrefix {
  using type = TensorMetafunctions::prepend_spatial_index<
      typename Tag::type, VolumeDim::value, UpLo::Lo, Fr>;
  using tag = Tag;
  static constexpr db::DataBoxString_t label = "d";
};

template <typename Tag, typename VolumeDim, typename Fr>
struct d<Tag, VolumeDim, Fr,
         Requires<tt::is_a_v<Variables, typename Tag::type>>>
    : db::DataBoxPrefix {
  using type = typename Tag::type;
  using tag = Tag;
  static constexpr db::DataBoxString_t label = "d";
};
}  // namespace Tags
