// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace Burgers::subcell {
/*!
 * \brief Class holding options using by the Burgers-specific parts of the
 * troubled-cell indicator.
 */
struct TciOptions {
  using options = tmpl::list<>;
  static constexpr Options::String help = {
      "Burgers-specific options for the troubled-cell indicator."};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/) noexcept {}
};

namespace OptionTags {
struct TciOptions {
  using type = subcell::TciOptions;
  static constexpr Options::String help =
      "Burgers-specific options for the TCI.";
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
};
}  // namespace OptionTags

namespace Tags {
struct TciOptions : db::SimpleTag {
  using type = subcell::TciOptions;

  using option_tags = tmpl::list<OptionTags::TciOptions>;
  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& tci_options) noexcept {
    return tci_options;
  }
};
}  // namespace Tags
}  // namespace Burgers::subcell
