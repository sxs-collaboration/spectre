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
#include <memory>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief A 3D equation of state representing a fluid in compositional
 * equalibrium.
 *
 *
 * The equation of state takes the form
 *
 * \f[
 * p = p (T, rho, Y_e) = p(T, rho, Y_e= Y_{e, \beta})
 * \f]
 *
 * where \f$\rho\f$ is the rest mass density and  \f$T\f$ is the
 * temperaturee;   \f$Y_e\f$  the electron fraction,  is not
 * used, and therefore this evaluating this EoS at any arbtirary
 * electron fraction is equivalent to evaluating it  in beta equalibrium
 *
 */
template <typename EquilEos>
class Equilibrium3D : public EquationOfState<EquilEos::is_relativistic, 3> {
 public:
  static constexpr size_t thermodynamic_dim = 3;
  static constexpr bool is_relativistic = EquilEos::is_relativistic;
  static std::string name() {
    return "Equilibrium3D(" + pretty_type::name<EquilEos>() + ")";
  }
  static constexpr Options::String help = {
      "An 3D EoS which is independent of electron fraction. "
      "Contains an underlying 2D EoS which is dependent only "
      "on rest mass density and temperature/internal energy."};

  struct UnderlyingEos {
    using type = EquilEos;
    static std::string name() { return pretty_type::short_name<EquilEos>(); }
    static constexpr Options::String help{
        "The underlying Eos which is being represented as a "
        "3D Eos.  Must be a 2D EoS"};
  };

  using options = tmpl::list<UnderlyingEos>;

  Equilibrium3D() = default;
  Equilibrium3D(const Equilibrium3D&) = default;
  Equilibrium3D& operator=(const Equilibrium3D&) = default;
  Equilibrium3D(Equilibrium3D&&) = default;
  Equilibrium3D& operator=(Equilibrium3D&&) = default;
  ~Equilibrium3D() override = default;

  explicit Equilibrium3D(const EquilEos& underlying_eos)
      : underlying_eos_(underlying_eos){};

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(Equilibrium3D, 3)

  std::unique_ptr<EquationOfState<EquilEos::is_relativistic, 3>> get_clone()
      const override;

  bool is_equal(
      const EquationOfState<EquilEos::is_relativistic, 3>& rhs) const override;

  /// \brief Returns `true` if the EOS is barotropic
  bool is_barotropic() const override { return false; }

  bool operator==(const Equilibrium3D<EquilEos>& rhs) const;

  bool operator!=(const Equilibrium3D<EquilEos>& rhs) const;
  /// @{
  /*!
   * Computes the electron fraction in beta-equilibrium \f$Y_e^{\rm eq}\f$ from
   * the rest mass density \f$\rho\f$ and the temperature \f$T\f$.
   */
  Scalar<double> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<double>& rest_mass_density,
      const Scalar<double>& temperature) const {
    return underlying_eos_
        .equilibrium_electron_fraction_from_density_temperature(
            rest_mass_density, temperature);
  }

  Scalar<DataVector> equilibrium_electron_fraction_from_density_temperature(
      const Scalar<DataVector>& rest_mass_density,
      const Scalar<DataVector>& temperature) const {
    return underlying_eos_
        .equilibrium_electron_fraction_from_density_temperature(
            rest_mass_density, temperature);
  }
  /// @}
  //

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<EquilEos::is_relativistic, 3>), Equilibrium3D);

  /// The lower bound of the electron fraction that is valid for this EOS
  double electron_fraction_lower_bound() const override { return 0.0; }

  /// The upper bound of the electron fraction that is valid for this EOS
  double electron_fraction_upper_bound() const override { return 1.0; }

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override {
    return underlying_eos_.rest_mass_density_lower_bound();
  }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override {
    return underlying_eos_.rest_mass_density_upper_bound();
  }

  /// The lower bound of the temperature that is valid for this EOS
  double temperature_lower_bound() const override {
    return underlying_eos_.temperature_lower_bound();
  }

  /// The upper bound of the temperature that is valid for this EOS
  double temperature_upper_bound() const override {
    return underlying_eos_.temperature_upper_bound();
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$ and electron fraction \f$Y_e\f$
  double specific_internal_energy_lower_bound(
      const double rest_mass_density,
      const double /*electron_fraction*/) const override {
    return underlying_eos_.specific_internal_energy_lower_bound(
        rest_mass_density);
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      const double rest_mass_density,
      const double /*electron_fraction*/) const override {
    return underlying_eos_.specific_internal_energy_upper_bound(
        rest_mass_density);
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return underlying_eos_.specific_enthalpy_lower_bound();
  }

  /// The baryon mass for this EoS
  double baryon_mass() const override { return underlying_eos_.baryon_mass(); }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(3)
  EquilEos underlying_eos_;
};
/// \cond
template <typename EquilEos>
PUP::able::PUP_ID EquationsOfState::Equilibrium3D<EquilEos>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
