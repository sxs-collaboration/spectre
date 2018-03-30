// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "tests/Unit/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <string>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Parallel/PupStlCpp11.hpp"
#include "PointwiseFunctions/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Overloader.hpp"
#include "tests/Unit/Pypp/CheckWithRandomValues.hpp"
#include "tests/Unit/Pypp/SetupLocalPythonEnvironment.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace TestHelpers {
namespace EquationsOfState {
namespace detail {
template <size_t ThermodynamicDim,
          typename = std::make_index_sequence<ThermodynamicDim>>
struct CreateMemberFunctionPointer;
template <size_t ThermodynamicDim, size_t... Is>
struct CreateMemberFunctionPointer<ThermodynamicDim,
                                   std::index_sequence<Is...>> {
  template <class DataType, class EoS>
  using f = Scalar<DataType> (EoS::*)(
      const Scalar<std::remove_pointer_t<decltype(
          (void)Is, std::add_pointer_t<DataType>{nullptr})>>&...) const;
};

template <class T, typename EoS>
using Function = Scalar<T> (EoS::*)(const Scalar<T>&, const Scalar<T>&) const;

template <bool IsRelativistic, size_t ThermodynamicDim, class... MemberArgs,
          class T, size_t... Is>
void check_impl(const std::unique_ptr<::EquationsOfState::EquationOfState<
                    IsRelativistic, ThermodynamicDim>>& in_eos,
                const std::string& python_function_prefix,
                const T& used_for_size,
                const std::index_sequence<Is...>& /*0 to ThermodynamicDim - 2*/,
                const MemberArgs&... member_args) noexcept {
  // `Is` does not go full range because we want to be able to go from enthalpy
  // to rest mass density, which means we need to be able to adjust the first
  // bounds independently

  // Bounds for: density, specific internal energy
  const std::array<std::pair<double, double>, 2> random_value_bounds{
      {{1.0e-4, 4.0}, {0.0, 1.0e4}}};
  using EoS =
      ::EquationsOfState::EquationOfState<IsRelativistic, ThermodynamicDim>;
  using Function =
      typename CreateMemberFunctionPointer<ThermodynamicDim>::template f<T,
                                                                         EoS>;
  INFO("Testing "s + (IsRelativistic ? "relativistic"s : "Newtonian"s) +
       " equation of state"s)
  const auto helper = [&](const std::unique_ptr<EoS>& eos) noexcept {
    // need func variable to work around GCC bug
    Function func{&EoS::pressure_from_density};
    INFO("Testing pressure_from_density...")
    pypp::check_with_random_values<sizeof...(Is) + 1>(
        func, *eos, "TestFunctions",
        python_function_prefix + "_pressure_from_density",
        {{random_value_bounds[0], random_value_bounds[Is + 1]...}},
        std::make_tuple(
            make_with_value<Scalar<T>>(used_for_size, member_args)...),
        used_for_size);
    make_overloader(
        [&](const std::integral_constant<size_t, 1>& /*thermodynamic_dim*/,
            auto eos_for_type) {
          INFO("Done\nTesting rest_mass_density_from_enthalpy...")
          pypp::check_with_random_values<sizeof...(Is) + 1>(
              func = &std::remove_pointer_t<decltype(
                         eos_for_type)>::rest_mass_density_from_enthalpy,
              *eos, "TestFunctions",
              IsRelativistic
                  ? std::string(python_function_prefix +
                                "_rel_rest_mass_density_from_enthalpy")
                  : std::string(python_function_prefix +
                                "_newt_rest_mass_density_from_enthalpy"),
              {{std::make_pair(random_value_bounds[0].first * 1.0e4,
                               random_value_bounds[0].second * 1.0e4),
                random_value_bounds[Is + 1]...}},
              std::make_tuple(
                  make_with_value<Scalar<T>>(used_for_size, member_args)...),
              used_for_size);
        },
        [](const auto& /*meta*/, const auto& /*meta*/) {})(
        std::integral_constant<size_t, ThermodynamicDim>{},
        std::add_pointer_t<EoS>{nullptr});
    INFO("Done\nTesting specific_enthalpy_from_density...")
    pypp::check_with_random_values<sizeof...(Is) + 1>(
        func = &EoS::specific_enthalpy_from_density, *eos, "TestFunctions",
        IsRelativistic ? std::string(python_function_prefix +
                                     "_rel_specific_enthalpy_from_density")
                       : std::string(python_function_prefix +
                                     "_newt_specific_enthalpy_from_density"),
        {{random_value_bounds[0], random_value_bounds[Is + 1]...}},
        std::make_tuple(
            make_with_value<Scalar<T>>(used_for_size, member_args)...),
        used_for_size);
    INFO("Done\nTesting specific_internal_energy_from_density...")
    pypp::check_with_random_values<sizeof...(Is) + 1>(
        func = &EoS::specific_internal_energy_from_density, *eos,
        "TestFunctions",
        python_function_prefix + "_specific_internal_energy_from_density",
        {{random_value_bounds[0], random_value_bounds[Is + 1]...}},
        std::make_tuple(
            make_with_value<Scalar<T>>(used_for_size, member_args)...),
        used_for_size);
    INFO("Done\nTesting chi_from_density...")
    pypp::check_with_random_values<sizeof...(Is) + 1>(
        func = &EoS::chi_from_density, *eos, "TestFunctions",
        python_function_prefix + "_chi_from_density",
        {{random_value_bounds[0], random_value_bounds[Is + 1]...}},
        std::make_tuple(
            make_with_value<Scalar<T>>(used_for_size, member_args)...),
        used_for_size);
    INFO("Done\nTesting kappa_times_p_over_rho_squared_from_density...")
    pypp::check_with_random_values<sizeof...(Is) + 1>(
        func = &EoS::kappa_times_p_over_rho_squared_from_density, *eos,
        "TestFunctions",
        python_function_prefix + "_kappa_times_p_over_rho_squared_from_density",
        {{random_value_bounds[0], random_value_bounds[Is + 1]...}},
        std::make_tuple(
            make_with_value<Scalar<T>>(used_for_size, member_args)...),
        used_for_size);
    INFO("Done\n\n")
  };
  helper(in_eos);
  helper(serialize_and_deserialize(in_eos));
}
}  // namespace detail

// @{
/*!
 * \ingroup TestingFrameworkGroup
 * \brief Test an equation of state by comparing to python functions
 *
 * The python functions must be added to
 * tests/Unit/PointwiseFunctions/EquationsOfState/TestFunctions.py. The prefix
 * for each class of equation of state is arbitrary, but should generally be
 * something like "polytropic" for polytropic fluids. The necessary python
 * functions are:
 * - `PREFIX_pressure_from_density`
 * - `PREFIX_rel_rest_mass_density_from_enthalpy`
 * - `PREFIX_newt_rest_mass_density_from_enthalpy`
 * - `PREFIX_rel_specific_enthalpy_from_density`
 * - `PREFIX_newt_specific_enthalpy_from_density`
 * - `PREFIX_specific_internal_energy_from_density`
 * - `PREFIX_chi_from_density`
 * - `PREFIX_kappa_times_p_over_rho_squared_from_density`
 *
 * The `python_function_prefix` argument passed to `check` must be `PREFIX`. If
 * an EoS class has member variables (these must be `double`s currently) that
 * are used to compute the quantities, such as the polytropic constant and
 * polytropic exponent for a fluid, then they must be passed in as the last
 * arguments to the `check` function`. Each python function must take these same
 * arguments as the trailing arguments.
 */
template <class EosType, class T, class... MemberArgs>
void check(std::unique_ptr<EosType> in_eos,
           const std::string& python_function_prefix, const T& used_for_size,
           const MemberArgs&... member_args) noexcept {
  detail::check_impl(std::unique_ptr<::EquationsOfState::EquationOfState<
                         EosType::is_relativistic, EosType::thermodynamic_dim>>(
                         std::move(in_eos)),
                     python_function_prefix, used_for_size,
                     std::make_index_sequence<EosType::thermodynamic_dim - 1>{},
                     member_args...);
}

template <class EosType, class T, class... MemberArgs>
void check(EosType in_eos, const std::string& python_function_prefix,
           const T& used_for_size, const MemberArgs&... member_args) noexcept {
  detail::check_impl(std::unique_ptr<::EquationsOfState::EquationOfState<
                         EosType::is_relativistic, EosType::thermodynamic_dim>>(
                         std::make_unique<EosType>(std::move(in_eos))),
                     python_function_prefix, used_for_size,
                     std::make_index_sequence<EosType::thermodynamic_dim - 1>{},
                     member_args...);
}
// @}
}  // namespace EquationsOfState
}  // namespace TestHelpers
