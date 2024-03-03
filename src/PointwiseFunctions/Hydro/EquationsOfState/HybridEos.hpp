// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/arithmetic/dec.hpp>
#include <boost/preprocessor/arithmetic/inc.hpp>
#include <boost/preprocessor/control/expr_iif.hpp>
#include <boost/preprocessor/list/adt.hpp>
#include <boost/preprocessor/repetition/for.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>
#include <boost/preprocessor/tuple/to_list.hpp>
#include <cstddef>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 *
 * \brief Hybrid equation of state combining a barotropic EOS for cold
 * (zero-temperature) part with a simple thermal part
 *
 * The hybrid equation of state:
 *
 * \f[
 * p = p_{cold}(\rho) + \rho (\Gamma_{th}-1) (\epsilon - \epsilon_{cold}(\rho))
 * \f]
 *
 * where \f$p\f$ is the pressure, \f$\rho\f$ is the rest mass density,
 * \f$\epsilon\f$ is the specific internal energy, \f$p_{cold}\f$ and
 * \f$\epsilon_{cold}\f$ are the pressure and specific internal energy evaluated
 * using the cold EOS, and \f$\Gamma_{th}\f$ is the adiabatic index for the
 * thermal part.
 *
 * The temperature \f$T\f$ is defined as
 *
 * \f[
 * T = (\Gamma_{th} - 1) (\epsilon - \epsilon_{cold})
 * \f]
 */
template <typename ColdEquationOfState>
class HybridEos
    : public EquationOfState<ColdEquationOfState::is_relativistic, 2> {
 public:
  static constexpr size_t thermodynamic_dim = 2;
  static constexpr bool is_relativistic = ColdEquationOfState::is_relativistic;

  struct ColdEos {
    using type = ColdEquationOfState;
    static constexpr Options::String help = {"Cold equation of state"};
    static std::string name() {
      return pretty_type::short_name<ColdEquationOfState>();
    }
  };

  struct ThermalAdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {"Adiabatic index Gamma_th"};
  };

  static constexpr Options::String help = {
      "A hybrid equation of state combining a cold EOS with a simple thermal "
      "part.  The pressure is related to the rest mass density by "
      " p = p_cold(rho) + rho * (Gamma_th - 1) * (epsilon - "
      "epsilon_cold(rho)), where p is the pressure, rho is the rest mass "
      "density, epsilon is the specific internal energy, p_cold and "
      "epsilon_cold are the pressure and specific internal energy evaluated "
      "using the cold EOS and Gamma_th is the adiabatic index for the thermal "
      "part."};

  using options = tmpl::list<ColdEos, ThermalAdiabaticIndex>;

  HybridEos() = default;
  HybridEos(const HybridEos&) = default;
  HybridEos& operator=(const HybridEos&) = default;
  HybridEos(HybridEos&&) = default;
  HybridEos& operator=(HybridEos&&) = default;
  ~HybridEos() override = default;

  HybridEos(ColdEquationOfState cold_eos, double thermal_adiabatic_index);

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(HybridEos, 2)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<is_relativistic, 2>), HybridEos);

  std::unique_ptr<EquationOfState<is_relativistic, 2>> get_clone()
      const override;

  std::unique_ptr<EquationOfState<is_relativistic, 3>> promote_to_3d_eos()
      const override;

  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const override { return false; }

  bool operator==(const HybridEos<ColdEquationOfState>& rhs) const;

  bool operator!=(const HybridEos<ColdEquationOfState>& rhs) const;

  bool is_equal(const EquationOfState<is_relativistic, 2>& rhs) const override;

  static std::string name() {
    return "HybridEos(" + pretty_type::name<ColdEquationOfState>() + ")";
  }

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override {
    return cold_eos_.rest_mass_density_lower_bound();
  }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return cold_eos_.rest_mass_density_upper_bound();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_lower_bound(
      const double rest_mass_density) const override {
    return cold_eos_.specific_internal_energy_lower_bound(rest_mass_density);
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      const double /*rest_mass_density*/) const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return cold_eos_.specific_enthalpy_lower_bound();
  }

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override { return cold_eos_.baryon_mass(); }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(2)

  ColdEquationOfState cold_eos_;
  double thermal_adiabatic_index_ =
      std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <typename ColdEquationOfState>
PUP::able::PUP_ID EquationsOfState::HybridEos<ColdEquationOfState>::my_PUP_ID =
    0;
/// \endcond
}  // namespace EquationsOfState
