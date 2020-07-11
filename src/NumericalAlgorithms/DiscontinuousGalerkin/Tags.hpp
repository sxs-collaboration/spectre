// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/Structure/Direction.hpp"  // IWYU pragma: keep
#include "Domain/Structure/ElementId.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Options/Options.hpp"

/// Functionality related to discontinuous Galerkin schemes
namespace dg {}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Simple boundary communication data
template <typename TemporalId, typename LocalData, typename RemoteData>
struct SimpleMortarData : db::SimpleTag {
  using type = dg::SimpleMortarData<TemporalId, LocalData, RemoteData>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// Data on mortars, indexed by (Direction, ElementId) pairs
template <typename Tag, size_t VolumeDim>
struct Mortars : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept {
    return "Mortars(" + db::tag_name<Tag>() + ")";
  }
  using tag = Tag;
  using Key = std::pair<::Direction<VolumeDim>, ::ElementId<VolumeDim>>;
  using type = std::unordered_map<Key, db::item_type<Tag>, boost::hash<Key>>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// Size of a mortar, relative to the element face.  That is, the part
/// of the face that it covers.
template <size_t Dim>
struct MortarSize : db::SimpleTag {
  using type = std::array<Spectral::MortarSize, Dim>;
};
}  // namespace Tags

namespace OptionTags {
/*!
 * \ingroup OptionGroupsGroup
 * \brief Holds the `OptionTags::NumericalFlux` option in the input file
 */
struct NumericalFluxGroup {
  static std::string name() noexcept { return "NumericalFlux"; }
  static constexpr OptionString help = "The numerical flux scheme";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The option tag that retrieves the parameters for the numerical
 * flux from the input file
 */
template <typename NumericalFluxType>
struct NumericalFlux {
  static std::string name() noexcept {
    return option_name<NumericalFluxType>();
  }
  static constexpr OptionString help = "Options for the numerical flux";
  using type = NumericalFluxType;
  using group = NumericalFluxGroup;
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \brief The global cache tag for the numerical flux
 */
template <typename NumericalFluxType>
struct NumericalFlux : db::SimpleTag {
  using type = NumericalFluxType;
  using option_tags =
      tmpl::list<::OptionTags::NumericalFlux<NumericalFluxType>>;

  static constexpr bool pass_metavariables = false;
  static NumericalFluxType create_from_options(
      const NumericalFluxType& numerical_flux) noexcept {
    return numerical_flux;
  }
};
}  // namespace Tags
