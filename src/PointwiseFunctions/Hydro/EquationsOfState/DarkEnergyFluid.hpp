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

namespace EquationsOfState {
/*!
 * \ingroup EquationsOfStateGroup
 * \brief Equation of state for a dark energy fluid
 *
 * A dark energy fluid equation of state:
 * \f[
 * p = w(z) \rho ( 1.0 + \epsilon)
 * \f]
 * where \f$\rho\f$ is the rest mass density, \f$\epsilon\f$ is the specific
 * internal energy, and \f$w(z) > 0\f$ is a parameter depending on the redshift
 * \f$z\f$.
 */
template <bool IsRelativistic>
class DarkEnergyFluid : public EquationOfState<IsRelativistic, 2> {
 public:
  static constexpr size_t thermodynamic_dim = 2;
  static constexpr bool is_relativistic = IsRelativistic;
  static_assert(IsRelativistic,
                "Dark energy fluid equation of state only makes sense in a "
                "relativistic setting.");

  struct ParameterW {
    using type = double;
    static constexpr OptionString help = {"Parameter w(z)"};
  };

  static constexpr OptionString help = {
      "A dark energy fluid equation of state.\n"
      "The pressure is related to the rest mass density by "
      "p = w(z) * rho * (1 + epsilon), where p is the pressure, rho is the "
      "rest mass density, epsilon is the specific internal energy, and w(z) is "
      "a parameter."};

  using options = tmpl::list<ParameterW>;

  DarkEnergyFluid() = default;
  DarkEnergyFluid(const DarkEnergyFluid&) = default;
  DarkEnergyFluid& operator=(const DarkEnergyFluid&) = default;
  DarkEnergyFluid(DarkEnergyFluid&&) = default;
  DarkEnergyFluid& operator=(DarkEnergyFluid&&) = default;
  ~DarkEnergyFluid() override = default;

  explicit DarkEnergyFluid(double parameter_w) noexcept;

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(DarkEnergyFluid, 2)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 2>), DarkEnergyFluid);

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(2)

  double parameter_w_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::DarkEnergyFluid<IsRelativistic>::my_PUP_ID =
    0;
/// \endcond
}  // namespace EquationsOfState
