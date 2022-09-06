// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <pup.h>
#include <string>
#include <tuple>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "Framework/TestHelpers.hpp"
#include "PointwiseFunctions/Hydro/EquationsOfState/EquationOfState.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Overloader.hpp"

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
    const std::string& python_file_name,
    const std::string& python_function_prefix, const T& used_for_size,
    const MemberArgs&... member_args) {
  // Bounds for: density
  const std::array<std::pair<double, double>, 1> random_value_bounds{
      {{1.0e-4, 4.0}}};
  MAKE_GENERATOR(generator, std::random_device{}());
  std::uniform_real_distribution<> distribution(random_value_bounds[0].first,
                                                random_value_bounds[0].second);
  const auto rest_mass_density = make_with_random_values<Scalar<T>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const auto specific_internal_energy = make_with_random_values<Scalar<T>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  using EoS = ::EquationsOfState::EquationOfState<IsRelativistic, 1>;
  using Function = typename CreateMemberFunctionPointer<1>::template f<T, EoS>;
  INFO("Testing "s + (IsRelativistic ? "relativistic"s : "Newtonian"s) +
       " equation of state"s)
  const auto member_args_tuple = std::make_tuple(member_args...);
  const auto helper = [&](const std::unique_ptr<EoS>& eos) {
    // need func variable to work around GCC bug
    Function func{&EoS::pressure_from_density};
    INFO("Testing pressure_from_density...")
    pypp::check_with_random_values<1>(
        func, *eos, python_file_name,
        python_function_prefix + "_pressure_from_density", random_value_bounds,
        member_args_tuple, used_for_size);
    INFO("Done\nTesting rest_mass_density_from_enthalpy...")
    pypp::check_with_random_values<1>(
        func = &EoS::rest_mass_density_from_enthalpy, *eos, python_file_name,
        IsRelativistic ? std::string(python_function_prefix +
                                     "_rel_rest_mass_density_from_enthalpy")
                       : std::string(python_function_prefix +
                                     "_newt_rest_mass_density_from_enthalpy"),
        {{{1, 1.0e4}}}, member_args_tuple, used_for_size);
    INFO("Done\nTesting specific_internal_energy_from_density...")
    pypp::check_with_random_values<1>(
        func = &EoS::specific_internal_energy_from_density, *eos,
        python_file_name,
        python_function_prefix + "_specific_internal_energy_from_density",
        random_value_bounds, member_args_tuple, used_for_size);
    INFO("Done\nTesting temperature_from_density...")
    CHECK(make_with_value<Scalar<T>>(used_for_size, 0.0) ==
          eos->temperature_from_density(rest_mass_density));
    INFO("Done\nTesting temperature_from_specific_internal_energy...")
    CHECK(make_with_value<Scalar<T>>(used_for_size, 0.0) ==
          eos->temperature_from_specific_internal_energy(
              specific_internal_energy));
    INFO("Done\nTesting chi_from_density...")
    pypp::check_with_random_values<1>(
        func = &EoS::chi_from_density, *eos, python_file_name,
        python_function_prefix + "_chi_from_density", random_value_bounds,
        member_args_tuple, used_for_size);
    INFO("Done\nTesting kappa_times_p_over_rho_squared_from_density...")
    pypp::check_with_random_values<1>(
        func = &EoS::kappa_times_p_over_rho_squared_from_density, *eos,
        python_file_name,
        python_function_prefix + "_kappa_times_p_over_rho_squared_from_density",
        random_value_bounds, member_args_tuple, used_for_size);
    INFO("Done\n\n")
  };
  helper(in_eos);
  helper(serialize_and_deserialize(in_eos));
}

template <bool IsRelativistic, class... MemberArgs, class T>
void check_impl(
    const std::unique_ptr<
        ::EquationsOfState::EquationOfState<IsRelativistic, 2>>& in_eos,
    const std::string& python_file_name,
    const std::string& python_function_prefix, const T& used_for_size,
    const MemberArgs&... member_args) {
  // Bounds for: density, specific internal energy
  const std::array<std::pair<double, double>, 2> random_value_bounds{
      {{1.0e-4, 4.0}, {0.0, 1.0e4}}};
  MAKE_GENERATOR(generator, std::random_device{}());
  std::uniform_real_distribution<> distribution(random_value_bounds[0].first,
                                                random_value_bounds[0].second);
  const auto rest_mass_density = make_with_random_values<Scalar<T>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  const auto specific_internal_energy = make_with_random_values<Scalar<T>>(
      make_not_null(&generator), make_not_null(&distribution), used_for_size);
  using EoS = ::EquationsOfState::EquationOfState<IsRelativistic, 2>;
  using Function = typename CreateMemberFunctionPointer<2>::template f<T, EoS>;
  INFO("Testing "s + (IsRelativistic ? "relativistic"s : "Newtonian"s) +
       " equation of state"s)
  const auto member_args_tuple = std::make_tuple(member_args...);
  const auto helper = [&](const std::unique_ptr<EoS>& eos) {
    // need func variable to work around GCC bug
    Function func{&EoS::pressure_from_density_and_energy};
    INFO("Testing pressure_from_density_and_energy...")
    pypp::check_with_random_values<2>(
        func, *eos, python_file_name,
        python_function_prefix + "_pressure_from_density_and_energy",
        random_value_bounds, member_args_tuple, used_for_size);
    INFO("Done\nTesting pressure_from_density_and_enthalpy...")
    pypp::check_with_random_values<2>(
        func = &EoS::pressure_from_density_and_enthalpy, *eos, python_file_name,
        IsRelativistic
            ? std::string(python_function_prefix +
                          "_rel_pressure_from_density_and_enthalpy")
            : std::string(python_function_prefix +
                          "_newt_pressure_from_density_and_enthalpy"),
        {{{1.0e-4, 4.0}, {1.0, 1.0e4}}}, member_args_tuple, used_for_size);
    INFO("Done\nTesting specific_internal_energy_from_density_and_pressure...")
    pypp::check_with_random_values<2>(
        func = &EoS::specific_internal_energy_from_density_and_pressure, *eos,
        python_file_name,
        python_function_prefix +
            "_specific_internal_energy_from_density_and_pressure",
        random_value_bounds, member_args_tuple, used_for_size);
    INFO("Done\nTesting temperature_from_density_and_specific_int_energy...")
    pypp::check_with_random_values<2>(
        func = &EoS::temperature_from_density_and_energy, *eos,
        python_file_name,
        python_function_prefix + "_temperature_from_density_and_energy",
        random_value_bounds, member_args_tuple, used_for_size);
    INFO("Done\nTesting specific_int_energy_from_density_and_temperature...");
    Approx custom_approx = Approx::custom().epsilon(1.e-9);
    CHECK_ITERABLE_CUSTOM_APPROX(
        specific_internal_energy,
        eos->specific_internal_energy_from_density_and_temperature(
            rest_mass_density,
            eos->temperature_from_density_and_energy(rest_mass_density,
                                                     specific_internal_energy)),
        custom_approx);
    INFO("Done\nTesting chi_from_density_and_energy...")
    pypp::check_with_random_values<2>(
        func = &EoS::chi_from_density_and_energy, *eos, python_file_name,
        python_function_prefix + "_chi_from_density_and_energy",
        random_value_bounds, member_args_tuple, used_for_size);
    INFO(
        "Done\nTesting "
        "kappa_times_p_over_rho_squared_from_density_and_energy...")
    pypp::check_with_random_values<2>(
        func = &EoS::kappa_times_p_over_rho_squared_from_density_and_energy,
        *eos, python_file_name,
        python_function_prefix +
            "_kappa_times_p_over_rho_squared_from_density_and_energy",
        random_value_bounds, member_args_tuple, used_for_size);
    INFO("Done\n\n")
  };
  helper(in_eos);
  helper(serialize_and_deserialize(in_eos));
}
}  // namespace detail

/// @{
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
void check(std::unique_ptr<EosType> in_eos, const std::string& python_file_name,
           const std::string& python_function_prefix, const T& used_for_size,
           const MemberArgs&... member_args) {
  detail::check_impl(std::unique_ptr<::EquationsOfState::EquationOfState<
                         EosType::is_relativistic, EosType::thermodynamic_dim>>(
                         std::move(in_eos)),
                     python_file_name, python_function_prefix, used_for_size,
                     member_args...);
}

template <class EosType, class T, class... MemberArgs>
void check(EosType in_eos, const std::string& python_file_name,
           const std::string& python_function_prefix, const T& used_for_size,
           const MemberArgs&... member_args) {
  detail::check_impl(std::unique_ptr<::EquationsOfState::EquationOfState<
                         EosType::is_relativistic, EosType::thermodynamic_dim>>(
                         std::make_unique<EosType>(std::move(in_eos))),
                     python_file_name, python_function_prefix, used_for_size,
                     member_args...);
}
/// @}

/// Test that cloning is correct, and that the equality operator is implemented
/// correctly
template <bool IsRelativistic, size_t ThermodynamicDim>
void test_get_clone(
    const ::EquationsOfState::EquationOfState<IsRelativistic, ThermodynamicDim>&
        in_eos) {
  auto cloned_eos = in_eos.get_clone();
  CHECK(*cloned_eos == in_eos);
}
}  // namespace EquationsOfState
}  // namespace TestHelpers
