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
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
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

template <bool IsRelativistic, class... MemberArgs, class T>
void check_impl(
    const std::unique_ptr<
        ::EquationsOfState::EquationOfState<IsRelativistic, 1>>& in_eos,
    const std::string& python_function_prefix, const T& used_for_size,
    const MemberArgs&... member_args) noexcept {
  // Bounds for: density
  const std::array<std::pair<double, double>, 1> random_value_bounds{
      {{1.0e-4, 4.0}}};
  using EoS = ::EquationsOfState::EquationOfState<IsRelativistic, 1>;
  using Function = typename CreateMemberFunctionPointer<1>::template f<T, EoS>;
  INFO("Testing "s + (IsRelativistic ? "relativistic"s : "Newtonian"s) +
       " equation of state"s)
  const auto helper = [&](const std::unique_ptr<EoS>& eos) noexcept {
    // need func variable to work around GCC bug
    Function func{&EoS::pressure_from_density};
    INFO("Testing pressure_from_density...")
    pypp::check_with_random_values<1>(
        func, *eos, "TestFunctions",
        python_function_prefix + "_pressure_from_density", random_value_bounds,
        std::make_tuple(member_args...), used_for_size);
    INFO("Done\nTesting rest_mass_density_from_enthalpy...")
    pypp::check_with_random_values<1>(
        func = &EoS::rest_mass_density_from_enthalpy, *eos, "TestFunctions",
        IsRelativistic ? std::string(python_function_prefix +
                                     "_rel_rest_mass_density_from_enthalpy")
                       : std::string(python_function_prefix +
                                     "_newt_rest_mass_density_from_enthalpy"),
        {{{1, 1.0e4}}}, std::make_tuple(member_args...), used_for_size);
    INFO("Done\nTesting specific_enthalpy_from_density...")
    pypp::check_with_random_values<1>(
        func = &EoS::specific_enthalpy_from_density, *eos, "TestFunctions",
        IsRelativistic ? std::string(python_function_prefix +
                                     "_rel_specific_enthalpy_from_density")
                       : std::string(python_function_prefix +
                                     "_newt_specific_enthalpy_from_density"),
        random_value_bounds, std::make_tuple(member_args...), used_for_size);
    INFO("Done\nTesting specific_internal_energy_from_density...")
    pypp::check_with_random_values<1>(
        func = &EoS::specific_internal_energy_from_density, *eos,
        "TestFunctions",
        python_function_prefix + "_specific_internal_energy_from_density",
        random_value_bounds, std::make_tuple(member_args...), used_for_size);
    INFO("Done\nTesting chi_from_density...")
    pypp::check_with_random_values<1>(
        func = &EoS::chi_from_density, *eos, "TestFunctions",
        python_function_prefix + "_chi_from_density", random_value_bounds,
        std::make_tuple(member_args...), used_for_size);
    INFO("Done\nTesting kappa_times_p_over_rho_squared_from_density...")
    pypp::check_with_random_values<1>(
        func = &EoS::kappa_times_p_over_rho_squared_from_density, *eos,
        "TestFunctions",
        python_function_prefix + "_kappa_times_p_over_rho_squared_from_density",
        random_value_bounds, std::make_tuple(member_args...), used_for_size);
    INFO(
        "Done\nTesting that rest_mass_density_from_enthalpy and "
        "specific_enthalpy_from_density are inverses of each other...")
    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> distribution(1.0, 1.0e+04);
    const auto specific_enthalpy = make_with_random_values<Scalar<T>>(
        make_not_null(&generator), make_not_null(&distribution), used_for_size);
    CHECK_ITERABLE_APPROX(
        specific_enthalpy,
        eos->specific_enthalpy_from_density(
            eos->rest_mass_density_from_enthalpy(specific_enthalpy)));
    INFO("Done\n\n")
  };
  helper(in_eos);
  helper(serialize_and_deserialize(in_eos));
}

template <bool IsRelativistic, class... MemberArgs, class T>
void check_impl(
    const std::unique_ptr<
        ::EquationsOfState::EquationOfState<IsRelativistic, 2>>& in_eos,
    const std::string& python_function_prefix, const T& used_for_size,
    const MemberArgs&... member_args) noexcept {
  // Bounds for: density, specific internal energy
  const std::array<std::pair<double, double>, 2> random_value_bounds{
      {{1.0e-4, 4.0}, {0.0, 1.0e4}}};
  using EoS = ::EquationsOfState::EquationOfState<IsRelativistic, 2>;
  using Function = typename CreateMemberFunctionPointer<2>::template f<T, EoS>;
  INFO("Testing "s + (IsRelativistic ? "relativistic"s : "Newtonian"s) +
       " equation of state"s)
  const auto helper = [&](const std::unique_ptr<EoS>& eos) noexcept {
    // need func variable to work around GCC bug
    Function func{&EoS::pressure_from_density_and_energy};
    INFO("Testing pressure_from_density_and_energy...")
    pypp::check_with_random_values<2>(
        func, *eos, "TestFunctions",
        python_function_prefix + "_pressure_from_density_and_energy",
        random_value_bounds, std::make_tuple(member_args...), used_for_size);
    INFO("Done\nTesting pressure_from_density_and_enthalpy...")
    pypp::check_with_random_values<2>(
        func = &EoS::pressure_from_density_and_enthalpy, *eos, "TestFunctions",
        IsRelativistic
            ? std::string(python_function_prefix +
                          "_rel_pressure_from_density_and_enthalpy")
            : std::string(python_function_prefix +
                          "_newt_pressure_from_density_and_enthalpy"),
        {{{1.0e-4, 4.0}, {1.0, 1.0e4}}}, std::make_tuple(member_args...),
        used_for_size);
    INFO("Done\nTesting specific_enthalpy_from_density_and_energy...")
    pypp::check_with_random_values<2>(
        func = &EoS::specific_enthalpy_from_density_and_energy, *eos,
        "TestFunctions",
        IsRelativistic
            ? std::string(python_function_prefix +
                          "_rel_specific_enthalpy_from_density_and_energy")
            : std::string(python_function_prefix +
                          "_newt_specific_enthalpy_from_density_and_energy"),
        random_value_bounds, std::make_tuple(member_args...), used_for_size);
    INFO("Done\nTesting specific_internal_energy_from_density_and_pressure...")
    pypp::check_with_random_values<2>(
        func = &EoS::specific_internal_energy_from_density_and_pressure, *eos,
        "TestFunctions",
        python_function_prefix +
            "_specific_internal_energy_from_density_and_pressure",
        random_value_bounds, std::make_tuple(member_args...), used_for_size);
    INFO("Done\nTesting chi_from_density_and_energy...")
    pypp::check_with_random_values<2>(
        func = &EoS::chi_from_density_and_energy, *eos, "TestFunctions",
        python_function_prefix + "_chi_from_density_and_energy",
        random_value_bounds, std::make_tuple(member_args...), used_for_size);
    INFO(
        "Done\nTesting "
        "kappa_times_p_over_rho_squared_from_density_and_energy...")
    pypp::check_with_random_values<2>(
        func = &EoS::kappa_times_p_over_rho_squared_from_density_and_energy,
        *eos, "TestFunctions",
        python_function_prefix +
            "_kappa_times_p_over_rho_squared_from_density_and_energy",
        random_value_bounds, std::make_tuple(member_args...), used_for_size);
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
 * tests/Unit/PointwiseFunctions/Hydro/EquationsOfState/TestFunctions.py. The
 * prefix for each class of equation of state is arbitrary, but should generally
 * be something like "polytropic" for polytropic fluids.
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
                     python_function_prefix, used_for_size, member_args...);
}

template <class EosType, class T, class... MemberArgs>
void check(EosType in_eos, const std::string& python_function_prefix,
           const T& used_for_size, const MemberArgs&... member_args) noexcept {
  detail::check_impl(std::unique_ptr<::EquationsOfState::EquationOfState<
                         EosType::is_relativistic, EosType::thermodynamic_dim>>(
                         std::make_unique<EosType>(std::move(in_eos))),
                     python_function_prefix, used_for_size, member_args...);
}
// @}
}  // namespace EquationsOfState
}  // namespace TestHelpers
