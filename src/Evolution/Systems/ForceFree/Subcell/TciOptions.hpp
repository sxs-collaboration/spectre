// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Auto.hpp"
#include "Options/String.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace ForceFree::subcell {

struct TciOptions {
 private:
  struct DoNotCheckTildeQ {};

 public:
  /*!
   * \brief The cutoff of the absolute value of the generalized charge density
   * \f$\tilde{Q}\f$ in an element to apply the Persson TCI.
   *
   * If maximum absolute value of \f$\tilde{Q}\f$ is below this option value,
   * the Persson TCI is not triggered for it.
   */
  struct TildeQCutoff {
    using type = Options::Auto<double, DoNotCheckTildeQ>;
    static constexpr Options::String help = {
        "If maximum absolute value of TildeQ in an element is below this value "
        "we do not apply the Persson TCI to TildeQ. To disable the check, set "
        "this option to 'DoNotCheckTildeQ'."};
  };

  using options = tmpl::list<TildeQCutoff>;

  static constexpr Options::String help = {
      "Options for the troubled-cell indicator"};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& /*p*/);

  std::optional<double> tilde_q_cutoff{
      std::numeric_limits<double>::signaling_NaN()};
};

namespace OptionTags {
struct TciOptions {
  using type = subcell::TciOptions;
  static constexpr Options::String help = "TCI options for ForceFree system";
  using group = ::dg::OptionTags::DiscontinuousGalerkinGroup;
};
}  // namespace OptionTags

namespace Tags {
struct TciOptions : db::SimpleTag {
  using type = subcell::TciOptions;
  using option_tags = tmpl::list<typename OptionTags::TciOptions>;

  static constexpr bool pass_metavariables = false;
  static type create_from_options(const type& tci_options) {
    return tci_options;
  }
};
}  // namespace Tags
}  // namespace ForceFree::subcell
