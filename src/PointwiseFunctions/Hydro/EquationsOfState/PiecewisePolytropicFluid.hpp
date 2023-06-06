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
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Equation of state for a piecewise polytropic fluid
 *
 * A piecewise polytropic equation of state \f$p=K_i\rho^{\Gamma_i}\f$ where
 *  \f$K_i\f$ is the polytropic constant and \f$\Gamma_i\f$ is the polytropic
 * exponent. Here the subscript \f$i\f$ indicates two pairs of constants and
 *  exponents which characterize `the stiffness' of the matter at low and high
 *  densities.  For a given density, the polytropic exponent is related to the
 *  polytropic index \f$N_p\f$ by \f$N_p=1/(\Gamma-1)\f$.  For posterity,
 *  this two piece polytrope has been used in toy models of CCSNe (e.g.,
 *  \cite Dimmelmeier2001 ) and could be extended to a general "M" number of
 * parts for simplified equations of state for neutron stars (e.g.,
 *  \cite OBoyle2020 ). For a reference to a general piecewise polytrope, see
 * Section 2.4.7 of \cite RezzollaBook.
 */
template <bool IsRelativistic>
class PiecewisePolytropicFluid : public EquationOfState<IsRelativistic, 1> {
 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = IsRelativistic;

  /// The density demarcating the high and low density descriptions of the
  /// fluid.
  struct PiecewisePolytropicTransitionDensity {
    using type = double;
    static constexpr Options::String help = {
        "Density below (above) which, the matter is described by a low (high) "
        "density polytropic fluid."};
    static double lower_bound() { return 0.0; }
  };

  /// The constant \f$K\f$ scaling the low density material
  /// \f$p=K\rho^{\Gamma}\f$.
  ///
  /// Note, by enforcing pressure continuity at the transition density
  /// \f$\bar{\rho}\f$, the high density constant \f$K_{high}\f$ is given
  /// as \f$K_{high} = K_{low} (\bar{\rho})^{\Gamma_{low} - \Gamma_{high}}\f$.
  struct PolytropicConstantLow {
    using type = double;
    static constexpr Options::String help = {
        "Polytropic constant K for lower"
        " density material"};
    static double lower_bound() { return 0.0; }
  };

  /// The exponent \f$\Gamma\f$, scaling the low density material
  /// \f$p=K\rho^{\Gamma}\f$.
  struct PolytropicExponentLow {
    using type = double;
    static constexpr Options::String help = {
        "Polytropic exponent for lower"
        " density material."};
    static double lower_bound() { return 1.0; }
  };

  /// The exponent \f$\Gamma\f$, scaling the high density material
  /// \f$p=K\rho^{\Gamma}\f$.
  struct PolytropicExponentHigh {
    using type = double;
    static constexpr Options::String help = {
        "Polytropic exponent for higher"
        " density material."};
    static double lower_bound() { return 1.0; }
  };

  static constexpr Options::String help = {
      "A piecewise polytropic fluid equation of state.\n"
      "The pressure is related to the rest mass density by p = K_i rho ^ "
      "Gamma_i, "
      "where p is the pressure, rho is the rest mass density, K_i is the "
      "polytropic constant either describing the low or high density material, "
      "and Gamma_i is the polytropic exponent for the low or high density "
      "material. The polytropic index N_i is defined as Gamma_i = 1 + 1 / N_i."
      "  The subscript `i' refers to different pairs of Gamma and K that can"
      " describe either low or high density material."};

  using options =
      tmpl::list<PiecewisePolytropicTransitionDensity, PolytropicConstantLow,
                 PolytropicExponentLow, PolytropicExponentHigh>;

  PiecewisePolytropicFluid() = default;
  PiecewisePolytropicFluid(const PiecewisePolytropicFluid&) = default;
  PiecewisePolytropicFluid& operator=(const PiecewisePolytropicFluid&) =
      default;
  PiecewisePolytropicFluid(PiecewisePolytropicFluid&&) = default;
  PiecewisePolytropicFluid& operator=(PiecewisePolytropicFluid&&) = default;
  ~PiecewisePolytropicFluid() override = default;

  PiecewisePolytropicFluid(double transition_density,
                           double polytropic_constant_lo,
                           double polytropic_exponent_lo,
                           double polytropic_exponent_hi);

  std::unique_ptr<EquationOfState<IsRelativistic, 1>> get_clone()
      const override;

  bool is_equal(const EquationOfState<IsRelativistic, 1>& rhs) const override;

  bool operator==(const PiecewisePolytropicFluid<IsRelativistic>& rhs) const;

  bool operator!=(const PiecewisePolytropicFluid<IsRelativistic>& rhs) const;

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(PiecewisePolytropicFluid, 1)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 1>), PiecewisePolytropicFluid);

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override { return 0.0; }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override;

  /// The lower bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_lower_bound(
      const double /* rest_mass_density */) const override {
    return 0.0;
  }

  /// The upper bound of the specific internal energy that is valid for this EOS
  /// at the given rest mass density \f$\rho\f$
  double specific_internal_energy_upper_bound(
      const double /* rest_mass_density */) const override {
    return std::numeric_limits<double>::max();
  }

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return IsRelativistic ? 1.0 : 0.0;
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(1)

  double transition_density_ = std::numeric_limits<double>::signaling_NaN();
  double transition_pressure_ = std::numeric_limits<double>::signaling_NaN();
  double transition_spec_eint_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_lo_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_lo_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_constant_hi_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_hi_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID
    EquationsOfState::PiecewisePolytropicFluid<IsRelativistic>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
