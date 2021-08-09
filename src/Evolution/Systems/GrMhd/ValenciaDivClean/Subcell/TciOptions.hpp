// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>
#include <optional>

#include "DataStructures/DataBox/Tag.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/OptionsGroup.hpp"
#include "Options/Auto.hpp"
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
 private:
  struct DoNotCheckMagneticField {};

 public:
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
  /// \brief Minimum value of \f$\tilde{\tau}\f$ before we switch to subcell.
  /// Used to identify places where the energy has suddenly become negative
  struct MinimumValueOfTildeTau {
    using type = double;
    static type lower_bound() noexcept { return 0.0; }
    static constexpr Options::String help = {
        "Minimum value of tilde tau before we switch to subcell."};
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
  /// \brief The cutoff where if the maximum of the magnetic field in an element
  /// is below this value we do not apply the Persson TCI to the magnetic field.
  struct MagneticFieldCutoff {
    using type = Options::Auto<double, DoNotCheckMagneticField>;
    static constexpr Options::String help = {
        "The cutoff where if the maximum of the magnetic field in an element "
        "is below this value we do not apply the Persson TCI to the magnetic "
        "field. This is to avoid switching to subcell in regions where there's "
        "no magnetic field.\n"
        "To disable the magnetic field check, set to "
        "'DoNotCheckMagneticField'."};
  };

  using options =
      tmpl::list<MinimumValueOfD, MinimumValueOfTildeTau, AtmosphereDensity,
                 SafetyFactorForB, MagneticFieldCutoff>;
  static constexpr Options::String help = {
      "Options for the troubled-cell indicator."};

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) noexcept;

  double minimum_rest_mass_density_times_lorentz_factor{
      std::numeric_limits<double>::signaling_NaN()};
  double minimum_tilde_tau{std::numeric_limits<double>::signaling_NaN()};
  double atmosphere_density{std::numeric_limits<double>::signaling_NaN()};
  double safety_factor_for_magnetic_field{
      std::numeric_limits<double>::signaling_NaN()};
  // The signaling_NaN default is chosen so that users hit an error/FPE if the
  // cutoff is not specified, rather than silently defaulting to ignoring the
  // magnetic field.
  std::optional<double> magnetic_field_cutoff{
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
