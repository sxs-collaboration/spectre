// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Options.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace grmhd::ValenciaDivClean::subcell {
/*!
 * \brief Class holding options using by the GRMHD-specific parts of the
 * troubled-cell indicator.
 */
struct TciOptions {
  /// \brief Minimum value of rest-mass density times Lorentz factor before we
  /// switch to subcell. Used to identify places where the density has suddenly
  /// become negative
  struct MinimumValueOfD {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr Options::String help = {
        "Minimum value of rest-mass density times Lorentz factor before we "
        "switch to subcell."};
  };
  /// \brief The density cutoff where if the maximum value of the density in the
  /// DG element is below this value we skip primitive recovery and treat the
  /// cell as atmosphere.
  struct AtmosphereDensity {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr Options::String help = {
        "The density cutoff where if the maximum value of the density in the "
        "DG element is below this value we skip primitive recovery and treat "
        "the cell as atmosphere."};
  };
  /// \brief Safety factor \f$\epsilon_B\f$.
  ///
  /// See the documentation for TciOnDgGrid for details on what this parameter
  /// controls.
  struct SafetyFactorForB {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr Options::String help = {
        "Safety factor for magnetic field bound."};
  };

  using options =
      tmpl::list<MinimumValueOfD, AtmosphereDensity, SafetyFactorForB>;
  static constexpr Options::String help = {
      "Options for the troubled-cell indicator."};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  double minimum_rest_mass_density_times_lorentz_factor{
      std::numeric_limits<double>::signaling_NaN()};
  double atmosphere_density{std::numeric_limits<double>::signaling_NaN()};
  double safety_factor_for_magnetic_field{
      std::numeric_limits<double>::signaling_NaN()};
};

namespace OptionTags {
struct TciOptions {
  using type = subcell::TciOptions;
  static constexpr Options::String help = "GRMHD-specific options for the TCI.";
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
}  // namespace grmhd::ValenciaDivClean::subcell
