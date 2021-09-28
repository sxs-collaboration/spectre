// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace evolution::dg {
namespace OptionTags {
/// The quadrature points to use.
struct Quadrature {
  using type = Spectral::Quadrature;
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
  static constexpr Options::String help =
      "The point distribution/quadrature rule used.";
};
}  // namespace OptionTags

namespace Tags {
/// The quadrature points to use initially.
///
/// While they could be changed during the evolution, it is unclear there is any
/// reason to do so or that changing them during an evolution would even be
/// stable.
struct Quadrature : db::SimpleTag {
  using type = Spectral::Quadrature;

  using option_tags = tmpl::list<OptionTags::Quadrature>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& quadrature) { return quadrature; }
};
}  // namespace Tags
}  // namespace evolution::dg
