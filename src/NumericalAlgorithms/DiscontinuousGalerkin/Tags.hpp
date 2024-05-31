// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "Domain/Structure/DirectionalIdMap.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Options/String.hpp"
#include "Utilities/PrettyType.hpp"

/// Functionality related to discontinuous Galerkin schemes
namespace dg {}

namespace Tags {
/// \ingroup DataBoxTagsGroup
/// \ingroup DiscontinuousGalerkinGroup
/// Data on mortars, indexed by a DirectionalId
template <typename Tag, size_t VolumeDim>
struct Mortars : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = DirectionalIdMap<VolumeDim, typename Tag::type>;
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
  static std::string name() { return "NumericalFlux"; }
  static constexpr Options::String help = "The numerical flux scheme";
};

/*!
 * \ingroup OptionTagsGroup
 * \brief The option tag that retrieves the parameters for the numerical
 * flux from the input file
 */
template <typename NumericalFluxType>
struct NumericalFlux {
  static std::string name() { return pretty_type::name<NumericalFluxType>(); }
  static constexpr Options::String help = "Options for the numerical flux";
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
      const NumericalFluxType& numerical_flux) {
    return numerical_flux;
  }
};
}  // namespace Tags
