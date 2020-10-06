// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define prefixes for DataBox tags

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TypeTraits/IsA.hpp"

/// \cond
template <class>
class Variables;  // IWYU pragma: keep
/// \endcond

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a time derivative
///
/// \snippet Test_DataBoxPrefixes.cpp dt_name
template <typename Tag>
struct dt : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a flux
///
/// \snippet Test_DataBoxPrefixes.cpp flux_name
template <typename Tag, typename VolumeDim, typename Fr,
          typename = std::nullptr_t>
struct Flux;

/// \cond
template <typename Tag, typename VolumeDim, typename Fr>
struct Flux<Tag, VolumeDim, Fr,
            Requires<tt::is_a_v<Tensor, typename Tag::type>>> : db::PrefixTag,
                                                                db::SimpleTag {
  using type = TensorMetafunctions::prepend_spatial_index<
      typename Tag::type, VolumeDim::value, UpLo::Up, Fr>;
  using tag = Tag;
};

template <typename Tag, typename VolumeDim, typename Fr>
struct Flux<Tag, VolumeDim, Fr,
            Requires<tt::is_a_v<::Variables, typename Tag::type>>>
    : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a source term
///
/// \snippet Test_DataBoxPrefixes.cpp source_name
template <typename Tag>
struct Source : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a source term that is independent of dynamic
/// variables
template <typename Tag>
struct FixedSource : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the initial value of a quantity
///
/// \snippet Test_DataBoxPrefixes.cpp initial_name
template <typename Tag>
struct Initial : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a boundary unit normal vector dotted into
/// the flux
///
/// \snippet Test_DataBoxPrefixes.cpp normal_dot_flux_name
template <typename Tag>
struct NormalDotFlux : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a boundary unit normal vector dotted into
/// the numerical flux
///
/// \snippet Test_DataBoxPrefixes.cpp normal_dot_numerical_flux_name
template <typename Tag>
struct NormalDotNumericalFlux : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the value a quantity will take on the
/// next iteration of the algorithm.
///
/// \snippet Test_DataBoxPrefixes.cpp next_name
template <typename Tag>
struct Next : db::PrefixTag, db::SimpleTag {
  using type = typename Tag::type;
  using tag = Tag;
};

}  // namespace Tags
