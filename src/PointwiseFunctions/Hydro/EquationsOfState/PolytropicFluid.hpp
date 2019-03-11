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
#include "Options/Options.hpp"
#include "Parallel/CharmPupable.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"  // IWYU pragma: keep
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
  static constexpr size_t thermodynamic_dim = 1;
  static constexpr bool is_relativistic = IsRelativistic;

  struct PolytropicConstant {
    using type = double;
    static constexpr OptionString help = {"Polytropic constant K"};
    static double lower_bound() noexcept { return 0.0; }
  };

  struct PolytropicExponent {
    using type = double;
    static constexpr OptionString help = {"Polytropic exponent Gamma"};
    static double lower_bound() noexcept { return 1.0; }
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

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(PolytropicFluid, 1)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 1>), PolytropicFluid);

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
