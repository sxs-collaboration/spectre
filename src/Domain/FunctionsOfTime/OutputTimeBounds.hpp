// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"

namespace domain::FunctionsOfTime {
/*!
 * \brief Put time bounds for all functions of time in a nicely formatted string
 *
 * All time bounds are printed in the following format:
 *
 * \code
 * FunctionsOfTime time bounds:
 *  Name1: (0.0000000000000000e+00,1.0000000000000000e+00)
 *  Name2: (3.0000000000000000e+00,4.0000000000000000e+00)
 *  ...
 * \endcode
 *
 */
std::string ouput_time_bounds(
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time);
}  // namespace domain::FunctionsOfTime
