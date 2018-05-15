// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/functional/hash.hpp>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Options/Options.hpp"

namespace Tags {
template <typename Tag, size_t VolumeDim>
struct Mortars : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "Mortar"; }
  using tag = Tag;
  using Key = std::pair<::Direction<VolumeDim>, ::ElementId<VolumeDim>>;
  using type = std::unordered_map<Key, db::item_type<Tag>, boost::hash<Key>>;
};
}  // namespace Tags

namespace CacheTags {
/*!
 * \ingroup CacheTagsGroup
 * \brief The global cache tag that retrieves the parameters for the numerical
 * flux from the input file
 */
template <typename NumericalFluxType>
struct NumericalFluxParams {
  static constexpr OptionString help = "The options for the numerical flux";
  using type = NumericalFluxType;
};
}  // namespace CacheTags
