// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

namespace dg {
namespace OptionTags {
struct Formulation {
  using type = dg::Formulation;
  using group = DiscontinuousGalerkinGroup;
  static constexpr Options::String help =
      "Discontinuous Galerkin formulation to use, e.g. StrongInertial for the "
      "strong form.";
};
}  // namespace OptionTags

namespace Tags {
/*!
 * \brief The DG formulation to use.
 */
struct Formulation : db::SimpleTag {
  using type = dg::Formulation;

  using option_tags = tmpl::list<OptionTags::Formulation>;
  static constexpr bool pass_metavariables = false;
  static dg::Formulation create_from_options(
      const dg::Formulation& formulation) noexcept;
};
}  // namespace Tags
}  // namespace dg
