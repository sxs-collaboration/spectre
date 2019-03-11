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
 * \brief Equation of state for an ideal fluid
 *
 * An ideal fluid equation of state:
 * \f[
 * p = \rho \epsilon(\gamma-1)
 * \f]
 * where \f$\rho\f$ is the rest mass density, \f$\epsilon\f$ is the specific
 * internal energy, and \f$\gamma\f$ is the adiabatic index.
 */
template <bool IsRelativistic>
class IdealFluid : public EquationOfState<IsRelativistic, 2> {
 public:
  static constexpr size_t thermodynamic_dim = 2;
  static constexpr bool is_relativistic = IsRelativistic;

  struct AdiabaticIndex {
    using type = double;
    static constexpr OptionString help = {"Adiabatic index gamma"};
  };

  static constexpr OptionString help = {
      "An ideal fluid equation of state.\n"
      "The pressure is related to the rest mass density by p = rho * epsilon * "
      "(gamma - 1), where p is the pressure, rho is the rest mass density, "
      "epsilon is the specific internal energy, and gamma is the adiabatic "
      "index."};

  using options = tmpl::list<AdiabaticIndex>;

  IdealFluid() = default;
  IdealFluid(const IdealFluid&) = default;
  IdealFluid& operator=(const IdealFluid&) = default;
  IdealFluid(IdealFluid&&) = default;
  IdealFluid& operator=(IdealFluid&&) = default;
  ~IdealFluid() override = default;

  explicit IdealFluid(double adiabatic_index) noexcept;

  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(IdealFluid, 2)

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EquationOfState<IsRelativistic, 2>), IdealFluid);

 private:
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(2)

  double adiabatic_index_ = std::numeric_limits<double>::signaling_NaN();
};

/// \cond
template <bool IsRelativistic>
PUP::able::PUP_ID EquationsOfState::IdealFluid<IsRelativistic>::my_PUP_ID = 0;
/// \endcond
}  // namespace EquationsOfState
