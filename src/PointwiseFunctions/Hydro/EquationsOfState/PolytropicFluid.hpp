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
 * \brief Equation of state for a polytropic fluid
 *
 * A polytropic equation of state \f$p=K\rho^{\Gamma}\f$ where \f$K\f$ is the
 * polytropic constant and \f$\Gamma\f$ is the polytropic exponent. The
 * polytropic exponent is related to the polytropic index \f$N_p\f$ by
 * \f$N_p=1/(\Gamma-1)\f$.
 *
 * We also have
 *
 * \f{align}{
 * \epsilon&=\frac{K\rho^{\Gamma-1}}{\Gamma-1}\\
 * h&=1+\epsilon+\frac{p}{\rho}=1+\frac{K\Gamma}{\Gamma-1}\rho^{\Gamma-1} \\
 * T&=0 \\
 * c_s^2&=\frac{\Gamma p}{\rho h}
 * =\frac{\Gamma(\Gamma-1)p}{\rho(\Gamma-1)+\Gamma p}
 * =\left(\frac{1}{\Gamma K\rho^{\Gamma-1}}+\frac{1}{\Gamma-1}\right)^{-1}
 * \f}
 */
template <bool IsRelativistic>
class PolytropicFluid : public EquationOfState<IsRelativistic, 1> {
 public:
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = IsRelativistic;

  struct PolytropicConstant {
    using type = double;
    static constexpr Options::String help = {"Polytropic constant K"};
    static double lower_bound() { return 0.0; }
  };

  struct PolytropicExponent {
    using type = double;
    static constexpr Options::String help = {"Polytropic exponent Gamma"};
    static double lower_bound() { return 1.0; }
  };

  static constexpr Options::String help = {
      "A polytropic fluid equation of state.\n"
      "The pressure is related to the rest mass density by p = K rho ^ Gamma, "
      "where p is the pressure, rho is the rest mass density, K is the "
      "polytropic constant, and Gamma is the polytropic exponent. The "
      "polytropic index N is defined as Gamma = 1 + 1 / N."};

  using options = tmpl::list<PolytropicConstant, PolytropicExponent>;

  PolytropicFluid() = default;
  PolytropicFluid(const PolytropicFluid&) = default;
  PolytropicFluid& operator=(const PolytropicFluid&) = default;
  PolytropicFluid(PolytropicFluid&&) = default;
  PolytropicFluid& operator=(PolytropicFluid&&) = default;
  ~PolytropicFluid() override = default;

  PolytropicFluid(double polytropic_constant, double polytropic_exponent);

  std::unique_ptr<EquationOfState<IsRelativistic, 1>> get_clone()
      const override;

  std::unique_ptr<EquationOfState<IsRelativistic, 3>> promote_to_3d_eos()
      const override;

  std::unique_ptr<EquationOfState<IsRelativistic, 2>> promote_to_2d_eos()
      const override;

  bool is_equal(const EquationOfState<IsRelativistic, 1>& rhs) const override;

  bool operator==(const PolytropicFluid<IsRelativistic>& rhs) const;

  bool operator!=(const PolytropicFluid<IsRelativistic>& rhs) const;

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(PolytropicFluid, 1)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 1>), PolytropicFluid);

  /// The lower bound of the rest mass density that is valid for this EOS
  double rest_mass_density_lower_bound() const override { return 0.0; }

  /// The upper bound of the rest mass density that is valid for this EOS
  double rest_mass_density_upper_bound() const override;

  /// The lower bound of the specific enthalpy that is valid for this EOS
  double specific_enthalpy_lower_bound() const override {
    return IsRelativistic ? 1.0 : 0.0;
  }

  /// The lower bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_lower_bound() const override { return 0.0; }

  /// The upper bound of the specific internal energy that is valid for this EOS
  double specific_internal_energy_upper_bound() const override;

  /// The vacuum baryon mass for this EoS
  double baryon_mass() const override {
    return hydro::units::geometric::default_baryon_mass;
  }

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(1)

  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::PolytropicFluid<IsRelativistic>::my_PUP_ID =
    0;
/// \endcond
}  // namespace EquationsOfState
