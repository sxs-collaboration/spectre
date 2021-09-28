// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <map>
#include <string>
#include <unordered_map>

namespace gsl {
template <class T>
class not_null;
}  // namespace gsl

namespace domain::FunctionsOfTime {
template <size_t max_deriv>
class PiecewisePolynomial;

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
/// Currently, only support order 3 piecewise polynomials.
/// This could be generalized later, but the SpEC functions of time
/// that we will read in with this action will always be 3rd-order
/// piecewise polynomials.
///
template <size_t MaxDeriv>
void read_spec_piecewise_polynomial(
    gsl::not_null<std::unordered_map<
        std::string, domain::FunctionsOfTime::PiecewisePolynomial<MaxDeriv>>*>
        spec_functions_of_time,
    const std::string& file_name,
    const std::map<std::string, std::string>& dataset_name_map);
}  // namespace domain::FunctionsOfTime
