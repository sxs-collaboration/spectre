// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <boost/preprocessor/punctuation/comma_if.hpp>
#include <boost/preprocessor/repetition/repeat.hpp>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

/// \cond
class DataVector;
namespace EquationsOfState {
template <bool IsRelativistic>
class DarkEnergyFluid;
template <bool IsRelativistic>
class IdealFluid;
template <bool IsRelativistic>
class PolytropicFluid;
}  // namespace EquationsOfState
/// \endcond

/// Contains all equations of state, including base class
namespace EquationsOfState {

namespace detail {
template <bool IsRelativistic, size_t ThermodynamicDim>
struct DerivedClasses;

template <bool IsRelativistic>
struct DerivedClasses<IsRelativistic, 1> {
  using type = tmpl::list<PolytropicFluid<IsRelativistic>>;
};

template <>
struct DerivedClasses<true, 2> {
  using type = tmpl::list<DarkEnergyFluid<true>, IdealFluid<true>>;
};

template <>
struct DerivedClasses<false, 2> {
  using type = tmpl::list<IdealFluid<false>>;
};
}  // namespace detail

/// \cond
template <bool IsRelativistic, size_t ThermodynamicDim,
          typename = std::make_index_sequence<ThermodynamicDim>>
class EquationOfState;
/// \endcond

/*!
 * \ingroup EquationsOfStateGroup
 * \brief Base class for equations of state depending on whether or not the
 * system is relativistic, and the number of variables used to determine the
 * pressure.
 *
 * The template parameter `IsRelativistic` is `true` for relativistic equations
 * of state and `false` for non-relativistic equations of state.
 *
 * For equations of state that have `ThermodynamicDim == 1`, there is also a
 * function that, given the enthalpy computes the rest mass density.
 */
template <bool IsRelativistic, size_t ThermodynamicDim, size_t... Is>
class EquationOfState<IsRelativistic, ThermodynamicDim,
                      std::index_sequence<Is...>> : public PUP::able {
 public:
  static_assert(
      sizeof...(Is) == ThermodynamicDim,
      "You must not pass a std::index_sequence directly to "
      "EquationOfState, you only need to specify whether or not it is "
      "relativistic and the number of arguments the functions should take.");

  static constexpr bool is_relativistic = IsRelativistic;
  static constexpr size_t thermodynamic_dim = ThermodynamicDim;
  using creatable_classes =
      typename detail::DerivedClasses<IsRelativistic, ThermodynamicDim>::type;

  EquationOfState() = default;
  EquationOfState(const EquationOfState&) = default;
  EquationOfState& operator=(const EquationOfState&) = default;
  EquationOfState(EquationOfState&&) = default;
  EquationOfState& operator=(EquationOfState&&) = default;
  ~EquationOfState() override = default;

  WRAPPED_PUPable_abstract(EquationOfState);  // NOLINT

  // @{
  /*!
   * Computes the pressure \f$p\f$ from:
   * - `ThermodynamicDim = 1`: the  rest mass density \f$\rho\f$
   * - `ThermodynamicDim = 2`: the  rest mass density \f$\rho\f$ and the
   * specific internal energy \f$\epsilon\f$
   * - `ThermodynamicDim = 3`: the  rest mass density \f$\rho\f$, the
   * specific internal energy \f$\epsilon\f$, and the electron fraction
   * \f$Y_e\f$
   */
  virtual Scalar<double> pressure_from_density(
      const Scalar<tt::identity_t<double, Is>>&...) const noexcept = 0;
  virtual Scalar<DataVector> pressure_from_density(
      const Scalar<tt::identity_t<DataVector, Is>>&...) const noexcept = 0;
  // @}

  // @{
  /*!
   * Computes the specific enthalpy \f$h\f$ from:
   * - `ThermodynamicDim = 1`: the  rest mass density \f$\rho\f$
   * - `ThermodynamicDim = 2`: the  rest mass density \f$\rho\f$ and the
   * specific internal energy \f$\epsilon\f$
   * - `ThermodynamicDim = 3`: the  rest mass density \f$\rho\f$, the
   * specific internal energy \f$\epsilon\f$, and the electron fraction
   * \f$Y_e\f$
   */
  virtual Scalar<double> specific_enthalpy_from_density(
      const Scalar<tt::identity_t<double, Is>>&...) const noexcept = 0;
  virtual Scalar<DataVector> specific_enthalpy_from_density(
      const Scalar<tt::identity_t<DataVector, Is>>&...) const noexcept = 0;
  // @}

  // @{
  /*!
   * Computes the specific internal energy \f$\epsilon\f$ from:
   * - `ThermodynamicDim = 1`: the  rest mass density \f$\rho\f$
   * - `ThermodynamicDim = 2`: the  rest mass density \f$\rho\f$ and the
   * specific internal energy \f$\epsilon\f$
   * - `ThermodynamicDim = 3`: the  rest mass density \f$\rho\f$, the
   * specific internal energy \f$\epsilon\f$, and the electron fraction
   * \f$Y_e\f$
   */
  virtual Scalar<double> specific_internal_energy_from_density(
      const Scalar<tt::identity_t<double, Is>>&...) const noexcept = 0;

  virtual Scalar<DataVector> specific_internal_energy_from_density(
      const Scalar<tt::identity_t<DataVector, Is>>&...) const noexcept = 0;
  // @}

  // @{
  /*!
   * Computes \f$\chi=\partial p / \partial \rho\f$ where \f$p\f$ is the
   * pressure and \f$\rho\f$ is the rest mass density, from:
   * - `ThermodynamicDim = 1`: the  rest mass density \f$\rho\f$
   * - `ThermodynamicDim = 2`: the  rest mass density \f$\rho\f$ and the
   * specific internal energy \f$\epsilon\f$
   * - `ThermodynamicDim = 3`: the  rest mass density \f$\rho\f$, the
   * specific internal energy \f$\epsilon\f$, and the electron fraction
   * \f$Y_e\f$
   */
  virtual Scalar<double> chi_from_density(
      const Scalar<tt::identity_t<double, Is>>&...) const noexcept = 0;

  virtual Scalar<DataVector> chi_from_density(
      const Scalar<tt::identity_t<DataVector, Is>>&...) const noexcept = 0;
  // @}

  // @{
  /*!
   * Computes \f$\kappa p/\rho^2=(p/\rho^2)\partial p / \partial \epsilon\f$
   * where \f$p\f$ is the pressure, \f$\rho\f$ is the rest mass density, and
   * \f$\epsilon\f$ is the specific internal energy from:
   * - `ThermodynamicDim = 1`: the  rest mass density \f$\rho\f$
   * - `ThermodynamicDim = 2`: the  rest mass density \f$\rho\f$ and the
   * specific internal energy \f$\epsilon\f$
   * - `ThermodynamicDim = 3`: the  rest mass density \f$\rho\f$, the
   * specific internal energy \f$\epsilon\f$, and the electron fraction
   * \f$Y_e\f$
   *
   * The reason for not returning just
   * \f$\kappa=\partial p / \partial \epsilon\f$ is to avoid division by zero
   * for small values of \f$\rho\f$ when assembling the speed of sound with
   * some equations of state.
   */
  virtual Scalar<double> kappa_times_p_over_rho_squared_from_density(
      const Scalar<tt::identity_t<double, Is>>&...) const noexcept = 0;

  virtual Scalar<DataVector> kappa_times_p_over_rho_squared_from_density(
      const Scalar<tt::identity_t<DataVector, Is>>&...) const noexcept = 0;
  // @}
};

/// \cond
template <bool IsRelativistic, size_t Is>
class EquationOfState<IsRelativistic, 1, std::index_sequence<Is>>
    : public PUP::able {
 public:
  static constexpr bool is_relativistic = IsRelativistic;
  static constexpr size_t thermodynamic_dim = 1;
  using creatable_classes =
      typename detail::DerivedClasses<IsRelativistic, 1>::type;

  EquationOfState() = default;
  EquationOfState(const EquationOfState&) = default;
  EquationOfState& operator=(const EquationOfState&) = default;
  EquationOfState(EquationOfState&&) = default;
  EquationOfState& operator=(EquationOfState&&) = default;
  ~EquationOfState() override = default;

  WRAPPED_PUPable_abstract(EquationOfState);  // NOLINT

  virtual Scalar<double> pressure_from_density(
      const Scalar<double>& /*rest_mass_density*/) const noexcept = 0;
  virtual Scalar<DataVector> pressure_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const noexcept = 0;

  virtual Scalar<double> rest_mass_density_from_enthalpy(
      const Scalar<double>& /*specific_enthalpy*/) const noexcept = 0;
  virtual Scalar<DataVector> rest_mass_density_from_enthalpy(
      const Scalar<DataVector>& /*specific_enthalpy*/) const noexcept = 0;

  virtual Scalar<double> specific_enthalpy_from_density(
      const Scalar<double>& /*rest_mass_density*/) const noexcept = 0;
  virtual Scalar<DataVector> specific_enthalpy_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const noexcept = 0;

  virtual Scalar<double> specific_internal_energy_from_density(
      const Scalar<double>& /*rest_mass_density*/) const noexcept = 0;
  virtual Scalar<DataVector> specific_internal_energy_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const noexcept = 0;

  virtual Scalar<double> chi_from_density(
      const Scalar<double>& /*rest_mass_density*/) const noexcept = 0;
  virtual Scalar<DataVector> chi_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const noexcept = 0;

  virtual Scalar<double> kappa_times_p_over_rho_squared_from_density(
      const Scalar<double>& /*rest_mass_density*/) const noexcept = 0;
  virtual Scalar<DataVector> kappa_times_p_over_rho_squared_from_density(
      const Scalar<DataVector>& /*rest_mass_density*/) const noexcept = 0;
};
/// \endcond
}  // namespace EquationsOfState

/// \cond
#define EQUATION_OF_STATE_ARGUMENTS_EXPAND(z, n, type) \
  BOOST_PP_COMMA_IF(n) const Scalar<type>&

#define EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS_HELPER(DIM, FUNCTION_NAME) \
  Scalar<double> FUNCTION_NAME(                                              \
      BOOST_PP_REPEAT(DIM, EQUATION_OF_STATE_ARGUMENTS_EXPAND, double))      \
      const noexcept override;                                               \
  Scalar<DataVector> FUNCTION_NAME(                                          \
      BOOST_PP_REPEAT(DIM, EQUATION_OF_STATE_ARGUMENTS_EXPAND, DataVector))  \
      const noexcept override
/// \endcond

/*!
 * \ingroup EquationsOfStateGroup
 * \brief Macro used to generate forward declarations of member functions in
 * derived classes
 */
#define EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS(DERIVED, DIM)            \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS_HELPER(DIM,                    \
                                                   pressure_from_density); \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS_HELPER(                        \
      DIM, specific_enthalpy_from_density);                                \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS_HELPER(                        \
      DIM, specific_internal_energy_from_density);                         \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS_HELPER(DIM, chi_from_density); \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBERS_HELPER(                        \
      DIM, kappa_times_p_over_rho_squared_from_density);                   \
                                                                           \
  /* clang-tidy: do not use non-const references */                        \
  void pup(PUP::er& p) noexcept override; /* NOLINT */                     \
                                                                           \
  explicit DERIVED(CkMigrateMessage* /*unused*/) noexcept

/// \cond
#define EQUATION_OF_STATE_FORWARD_ARGUMENTS(z, n, unused) \
  BOOST_PP_COMMA_IF(n) arg##n

#define EQUATION_OF_STATE_ARGUMENTS_EXPAND_NAMED(z, n, type) \
  BOOST_PP_COMMA_IF(n) const Scalar<type>& arg##n

#define EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER(                        \
    TEMPLATE, DERIVED, DATA_TYPE, DIM, FUNCTION_NAME)                       \
  TEMPLATE                                                                  \
  Scalar<DATA_TYPE> DERIVED::FUNCTION_NAME(BOOST_PP_REPEAT(                 \
      DIM, EQUATION_OF_STATE_ARGUMENTS_EXPAND_NAMED, DATA_TYPE))            \
      const noexcept {                                                      \
    return FUNCTION_NAME##_impl(                                            \
        BOOST_PP_REPEAT(DIM, EQUATION_OF_STATE_FORWARD_ARGUMENTS, UNUSED)); \
  }
/// \endcond

#define EQUATION_OF_STATE_MEMBER_DEFINITIONS(TEMPLATE, DERIVED, DATA_TYPE,  \
                                             DIM)                           \
  EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER(TEMPLATE, DERIVED, DATA_TYPE, \
                                              DIM, pressure_from_density)   \
  EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER(                              \
      TEMPLATE, DERIVED, DATA_TYPE, DIM, specific_enthalpy_from_density)    \
  EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER(                              \
      TEMPLATE, DERIVED, DATA_TYPE, DIM,                                    \
      specific_internal_energy_from_density)                                \
  EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER(TEMPLATE, DERIVED, DATA_TYPE, \
                                              DIM, chi_from_density)        \
  EQUATION_OF_STATE_MEMBER_DEFINITIONS_HELPER(                              \
      TEMPLATE, DERIVED, DATA_TYPE, DIM,                                    \
      kappa_times_p_over_rho_squared_from_density)

/// \cond
#define EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS_HELPER(DIM,           \
                                                              FUNCTION_NAME) \
  template <class DataType>                                                  \
  Scalar<DataType> FUNCTION_NAME(BOOST_PP_REPEAT(                            \
      DIM, EQUATION_OF_STATE_ARGUMENTS_EXPAND, DataType)) const noexcept
/// \endcond

#define EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS(DIM) \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS_HELPER(    \
      DIM, pressure_from_density_impl);                     \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS_HELPER(    \
      DIM, specific_enthalpy_from_density_impl);            \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS_HELPER(    \
      DIM, specific_internal_energy_from_density_impl);     \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS_HELPER(    \
      DIM, chi_from_density_impl);                          \
  EQUATION_OF_STATE_FORWARD_DECLARE_MEMBER_IMPLS_HELPER(    \
      DIM, kappa_times_p_over_rho_squared_from_density_impl)

#include "PointwiseFunctions/EquationsOfState/DarkEnergyFluid.hpp"
#include "PointwiseFunctions/EquationsOfState/IdealFluid.hpp"
#include "PointwiseFunctions/EquationsOfState/PolytropicFluid.hpp"
