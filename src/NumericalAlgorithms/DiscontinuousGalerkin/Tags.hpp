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
#include "Domain/Direction.hpp"  // IWYU pragma: keep
#include "Domain/ElementId.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleBoundaryData.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Options/Options.hpp"

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// \brief Simple boundary communication data
template <typename TemporalId, typename LocalData, typename RemoteData>
struct SimpleBoundaryData : db::SimpleTag {
  static std::string name() noexcept { return "SimpleBoundaryData"; }
  using type = dg::SimpleBoundaryData<TemporalId, LocalData, RemoteData>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// Data on mortars, indexed by (Direction, ElementId) pairs
template <typename Tag, size_t VolumeDim>
struct Mortars : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "Mortars"; }
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
  static std::string name() noexcept { return "MortarSize"; }
  using type = std::array<Spectral::MortarSize, Dim>;
};

/// \ingroup DataBoxTagsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// The object that computes numerical fluxes for a DG scheme
template <typename NumericalFluxType>
struct NormalDotNumericalFluxComputer : db::SimpleTag {
  static std::string name() noexcept {
    return "NormalDotNumericalFluxComputer(" +
           pretty_type::short_name<NumericalFluxType>() + ")";
  }
  using type = NumericalFluxType;
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
 * \brief The global cache tag that retrieves the parameters for the numerical
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
  using container_tag = Tags::NormalDotNumericalFluxComputer<NumericalFluxType>;
};
}  // namespace OptionTags
