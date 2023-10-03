// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <limits>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;

namespace PUP {
class er;
}  // namespace PUP
/// \endcond

namespace VariableFixing {

/*!
 * \ingroup VariableFixingGroup
 * \brief Adjust the electron fraction (Ye) based on rest mass density (rho).
 *
 * Based on \cite Liebendorfer2005 spherically symmetric Boltzmann calculations,
 * during the collapse phase just before a core-collapse supernova, the electron
 * fraction naturally follows the rest mass density.  Intuitively, the higher
 * the density, the more electrons are captured onto protons, leading to
 * neutrino production.  As these neutrinos diffuse out of the center of the
 * collapsing star, the total number of leptons near center of the CCSN
 * drops---the process of deleptonization.
 *
 * https://iopscience.iop.org/article/10.1086/466517
 */
class ParameterizedDeleptonization {
 public:
  /// \brief Use an analytic expression vs rest mass density profile
  ///
  /// The option to choose between analytic and tabulated will be added in the
  /// future.  If analytic is chosen (currently the only option), use the below
  /// parameters.  If tabulated is chosen, a Ye (rho) profile will need to be
  /// provided.
  //   struct DeleptonizationFormat {
  //     using type = double;
  //     static type lower_bound() { return 0.0; }
  //     static constexpr Options::String help = {"Choose an 'analytic' or
  //     'tabulated' expression to express electron fraction as a function of
  //     rest mass density."};
  //   };

  /// \brief Enable parameterized deleptonizations
  struct Enable {
    using type = bool;
    static constexpr Options::String help = {
        "Whether or not to activate parameterized deleptonization for "
        "supernovae."};
  };

  /// \brief Density near the center of the supernova at bounce, above which
  /// the central Ye is assumed
  ///
  /// In practice: rho(high) ~ 2x10^13 g / cm^3
  struct HighDensityScale {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "High end of density scale for parameterized deleptonization."};
  };
  /// \brief Density near the Silicon-Oxygen interface, below which the lower
  /// Ye is assumed
  ///
  /// In practice: rho(low) ~ 2x10^7 g / cm^3
  struct LowDensityScale {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "Low end of density scale for parameterized deleptonization."};
  };
  /// \brief Electron fraction of material when the rest mass density is above
  /// HighDensityScale
  ///
  /// In practice: Y_e (high density) ~ 0.28.
  struct ElectronFractionAtHighDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 0.5; }
    static constexpr Options::String help = {
        "For densities above HighDensityScale, the electron fraction will "
        "take this value."};
  };
  /// \brief Electron fraction of material when the rest mass density is below
  /// LowDensityScale
  ///
  /// In practice: Y_e (low density) ~ 0.5
  struct ElectronFractionAtLowDensity {
    using type = double;
    static type lower_bound() { return 0.0; }
    static type upper_bound() { return 0.5; }
    static constexpr Options::String help = {
        "For densities below LowDensityScale, the electron fraction will "
        "take this value."};
  };

  /// \brief Electron fraction correction term.  The larger this value, the
  /// higher the Ye of matter at densities between LowDensityScale and
  /// HighDensityScale.
  ///
  /// In practice: Y_e (correction) ~ 0.035
  struct ElectronFractionCorrectionScale {
    using type = double;
    static type lower_bound() { return 0.0; }
    static constexpr Options::String help = {
        "For densities between low and high limits, a higher value of "
        "ElectronFractionCorrectionScale will increase the value of Ye."};
  };

  using options =
      tmpl::list<Enable, HighDensityScale, LowDensityScale,
                 ElectronFractionAtHighDensity, ElectronFractionAtLowDensity,
                 ElectronFractionCorrectionScale>;
  static constexpr Options::String help = {
      "Set electron fraction based on rest mass density.  "
      "(Low/High)DensityScale sets the limits of the density, beyond which "
      "the ElectronFractionAt(Low/High)Density is assumed.  At intermediate "
      "densities, the higher the ElectronFractionCorrectionScale, the higher "
      "the Ye."};

  ParameterizedDeleptonization(bool enable, double high_density_scale,
                               double low_density_scale,
                               double electron_fraction_at_high_density,
                               double electron_fraction_at_low_density,
                               double electron_fraction_correction_scale,
                               const Options::Context& context = {});

  ParameterizedDeleptonization() = default;
  ParameterizedDeleptonization(const ParameterizedDeleptonization& /*rhs*/) =
      default;
  ParameterizedDeleptonization& operator=(
      const ParameterizedDeleptonization& /*rhs*/) = default;
  ParameterizedDeleptonization(ParameterizedDeleptonization&& /*rhs*/) =
      default;
  ParameterizedDeleptonization& operator=(
      ParameterizedDeleptonization&& /*rhs*/) = default;
  ~ParameterizedDeleptonization() = default;

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  // Quantities being mutated
  using return_tags =
      tmpl::list<hydro::Tags::SpecificInternalEnergy<DataVector>,
                 hydro::Tags::ElectronFraction<DataVector>,
                 hydro::Tags::Pressure<DataVector>,
                 hydro::Tags::SpecificEnthalpy<DataVector>,
                 hydro::Tags::Temperature<DataVector>>;

  // Things you want from DataBox that won't be change and are passed in as
  // const-refs
  using argument_tags = tmpl::list<hydro::Tags::RestMassDensity<DataVector>,
                                   hydro::Tags::EquationOfStateBase>;

  // for use in `db::mutate_apply`
  template <size_t ThermodynamicDim>
  void operator()(
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
      gsl::not_null<Scalar<DataVector>*> temperature,
      const Scalar<DataVector>& rest_mass_density,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state) const;

 private:
  template <size_t ThermodynamicDim>
  void correct_electron_fraction(
      gsl::not_null<Scalar<DataVector>*> specific_internal_energy,
      gsl::not_null<Scalar<DataVector>*> electron_fraction,
      gsl::not_null<Scalar<DataVector>*> pressure,
      gsl::not_null<Scalar<DataVector>*> specific_enthalpy,
      gsl::not_null<Scalar<DataVector>*> temperature,
      const Scalar<DataVector>& rest_mass_density,
      const EquationsOfState::EquationOfState<true, ThermodynamicDim>&
          equation_of_state,
      size_t grid_index) const;

  // NOLINTNEXTLINE(readability-redundant-declaration)
  friend bool operator==(const ParameterizedDeleptonization& lhs,
                         const ParameterizedDeleptonization& rhs);

  bool enable_{false};
  double high_density_scale_{std::numeric_limits<double>::signaling_NaN()};
  double low_density_scale_{std::numeric_limits<double>::signaling_NaN()};
  double electron_fraction_at_high_density_{
      std::numeric_limits<double>::signaling_NaN()};
  double electron_fraction_at_low_density_{
      std::numeric_limits<double>::signaling_NaN()};
  double electron_fraction_half_sum_{
      std::numeric_limits<double>::signaling_NaN()};
  double electron_fraction_half_difference_{
      std::numeric_limits<double>::signaling_NaN()};
  double electron_fraction_correction_scale_{
      std::numeric_limits<double>::signaling_NaN()};
};
bool operator!=(const ParameterizedDeleptonization& lhs,
                const ParameterizedDeleptonization& rhs);

}  // namespace VariableFixing
