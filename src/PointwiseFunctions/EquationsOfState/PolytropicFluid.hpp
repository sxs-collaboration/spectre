// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/repetition/repeat.hpp>
#include <limits>
#include <pup.h>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Equation of state for a polytropic fluid
 *
 * A polytropic equation of state \f$p=K\rho^{\Gamma}\f$ where \f$K\f$ is the
 * polytropic constant and \f$\Gamma\f$ is the polytropic exponent. The
 * polytropic exponent is related to the polytropic index \f$N_p\f$ by
 * \f$N_p=1/(\Gamma-1)\f$.
 */
template <bool IsRelativistic>
class PolytropicFluid : public EquationOfState<IsRelativistic, 1> {
 public:
  struct PolytropicConstant {
    using type = double;
    static constexpr OptionString help = {"Polytropic constant K"};
  };

  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {"Polytropic exponent Gamma"};
  };

  static constexpr OptionString help = {
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

  PolytropicFluid(double polytropic_constant,
                  double polytropic_exponent) noexcept;

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(PolytropicFluid, 1);

  Scalar<double> rest_mass_density_from_enthalpy(
      const Scalar<double>& specific_enthalpy) const noexcept override;
  Scalar<DataVector> rest_mass_density_from_enthalpy(
      const Scalar<DataVector>& specific_enthalpy) const noexcept override;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 1>), PolytropicFluid);

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(1);

  template <class DataType>
  Scalar<DataType> rest_mass_density_from_enthalpy_impl(
      const Scalar<DataType>& specific_enthalpy) const noexcept;

  double polytropic_constant_ = std::numeric_limits<double>::signaling_NaN();
  double polytropic_exponent_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::PolytropicFluid<IsRelativistic>::my_PUP_ID =
    0;
/// \endcond
}  // namespace EquationsOfState
