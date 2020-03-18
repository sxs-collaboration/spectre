// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Define prefixes for DataBox tags

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
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
  static std::string name() noexcept {
    return "dt(" + db::tag_name<Tag>() + ")";
  }
  using type = db::const_item_type<Tag>;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the analytic solution value for a quantity
///
/// \snippet Test_DataBoxPrefixes.cpp analytic_name
template <typename Tag>
struct Analytic : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
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
            Requires<tt::is_a_v<Tensor, db::const_item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  using type = TensorMetafunctions::prepend_spatial_index<
      db::const_item_type<Tag>, VolumeDim::value, UpLo::Up, Fr>;
  using tag = Tag;
  static std::string name() noexcept {
    return "Flux(" + db::tag_name<Tag>() + ")";
  }
};

template <typename Tag, typename VolumeDim, typename Fr>
struct Flux<Tag, VolumeDim, Fr,
            Requires<tt::is_a_v<::Variables, db::const_item_type<Tag>>>>
    : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "Flux(" + db::tag_name<Tag>() + ")";
  }
};
/// \endcond

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a source term
///
/// \snippet Test_DataBoxPrefixes.cpp source_name
template <typename Tag>
struct Source : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "Source(" + db::tag_name<Tag>() + ")";
  }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a source term that is independent of dynamic
/// variables
template <typename Tag>
struct FixedSource : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the initial value of a quantity
///
/// \snippet Test_DataBoxPrefixes.cpp initial_name
template <typename Tag>
struct Initial : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "Initial(" + db::tag_name<Tag>() + ")";
  }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a boundary unit normal vector dotted into
/// the flux
///
/// \snippet Test_DataBoxPrefixes.cpp normal_dot_flux_name
template <typename Tag>
struct NormalDotFlux : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "NormalDotFlux(" + db::tag_name<Tag>() + ")";
  }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating a boundary unit normal vector dotted into
/// the numerical flux
///
/// \snippet Test_DataBoxPrefixes.cpp normal_dot_numerical_flux_name
template <typename Tag>
struct NormalDotNumericalFlux : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "NormalDotNumericalFlux(" + db::tag_name<Tag>() + ")";
  }
};

/// \ingroup DataBoxTagsGroup
/// \brief Prefix indicating the value a quantity will take on the
/// next iteration of the algorithm.
///
/// \snippet Test_DataBoxPrefixes.cpp next_name
template <typename Tag>
struct Next : db::PrefixTag, db::SimpleTag {
  using type = db::const_item_type<Tag>;
  using tag = Tag;
  static std::string name() noexcept {
    return "Next(" + db::tag_name<Tag>() + ")";
  }
};

/// \ingroup DataBoxTagsGroup
/// \brief Compute item for `Tags::Next<Tag>`
///
/// Requires that a template specialization for the requested `Tag` is defined.
template <typename Tag>
struct NextCompute;

}  // namespace Tags
