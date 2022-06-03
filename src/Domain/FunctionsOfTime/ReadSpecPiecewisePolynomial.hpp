// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl

namespace domain::FunctionsOfTime {
template <size_t max_deriv>
class PiecewisePolynomial;
class FunctionOfTime;

/// \brief Import SpEC `FunctionOfTime` data from an H5 file.
///
/// Columns in the file to be read must have the following form:
///   - 0 = time
///   - 1 = time of last update
///   - 2 = number of components
///   - 3 = maximum derivative order
///   - 4 = version
///   - 5 = function
///   - 6 = d/dt (function)
///   - 7 = d^2/dt^2 (function)
///   - 8 = d^3/dt^3 (function)
///
/// If the function has more than one component, columns 5-8 give
/// the first component and its derivatives, columns 9-12 give the second
/// component and its derivatives, etc.
///
/// Currently, only support order 2 and 3 piecewise polynomials and order 3
/// quaternion functions of time. This could be generalized later, but the SpEC
/// functions of time that we will read in with this action will always be
/// 3rd-order piecewise polynomials.
///
template <template <size_t> class FoTType, size_t MaxDeriv>
void read_spec_piecewise_polynomial(
    gsl::not_null<std::unordered_map<std::string, FoTType<MaxDeriv>>*>
        spec_functions_of_time,
    const std::string& file_name,
    const std::map<std::string, std::string>& dataset_name_map,
    const bool quaternion_rotation = false);

/// \brief Replace the functions of time from the `domain_creator` with the ones
/// read in from `function_of_time_file`.
///
/// \note Currently, only support order 2 or 3 piecewise polynomials. This could
/// be generalized later, but the SpEC functions of time that we will read in
/// with this action will always be 2nd-order or 3rd-order piecewise polynomials
void override_functions_of_time(
    const gsl::not_null<std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>*>
        functions_of_time,
    const std::string& function_of_time_file,
    const std::map<std::string, std::string>& function_of_time_name_map);
}  // namespace domain::FunctionsOfTime
