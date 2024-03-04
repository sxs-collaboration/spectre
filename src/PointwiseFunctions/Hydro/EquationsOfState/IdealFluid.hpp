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
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Equation of state for an ideal fluid
 *
 * An ideal fluid equation of state:
 *
 * \f[
 * p = \rho \epsilon(\gamma-1)
 * \f]
 *
 * where \f$\rho\f$ is the rest mass density, \f$\epsilon\f$ is the specific
 * internal energy, and \f$\gamma\f$ is the adiabatic index.
 *
 * The temperature \f$T\f$ is defined as
 *
 * \f[
 * T = (\gamma - 1) \epsilon
 * \f]
 */
template <bool IsRelativistic>
class IdealFluid : public EquationOfState<IsRelativistic, 2> {
 public:
  static constexpr size_t thermodynamic_dim = 2;
  static constexpr bool is_relativistic = IsRelativistic;

  struct AdiabaticIndex {
    using type = double;
    static constexpr Options::String help = {"Adiabatic index gamma"};
  };

  static constexpr Options::String help = {
      "An ideal fluid equation of state.\n"
      "The pressure is related to the rest mass density by p = rho * epsilon * "
      "(gamma - 1), where p is the pressure, rho is the rest mass density, "
      "epsilon is the specific internal energy, and gamma is the adiabatic "
      "index.\n"
      "The temperature T is defined as T=epsilon."};

  using options = tmpl::list<AdiabaticIndex>;

  IdealFluid() = default;
  IdealFluid(const IdealFluid&) = default;
  IdealFluid& operator=(const IdealFluid&) = default;
  IdealFluid(IdealFluid&&) = default;
  IdealFluid& operator=(IdealFluid&&) = default;
  ~IdealFluid() override = default;

  explicit IdealFluid(double adiabatic_index);

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(IdealFluid, 2)

  std::unique_ptr<EquationOfState<IsRelativistic, 2>> get_clone()
      const override;

  std::unique_ptr<EquationOfState<IsRelativistic, 3>> promote_to_3d_eos()
      const override;

  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const override { return false; }

  bool operator==(const IdealFluid<IsRelativistic>& rhs) const;

  bool operator!=(const IdealFluid<IsRelativistic>& rhs) const;

  bool is_equal(const EquationOfState<IsRelativistic, 2>& rhs) const override;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 2>), IdealFluid);

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override { return 0.0; }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_lower_bound(
      const double /* rest_mass_density */) const override {
    return 0.0;
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      double rest_mass_density) const override;

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return IsRelativistic ? 1.0 : 0.0;
  }

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override {
    return hydro::units::geometric::default_baryon_mass;
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(2)

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::IdealFluid<IsRelativistic>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
