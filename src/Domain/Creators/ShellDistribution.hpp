// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>
#include <variant>
#include <vector>

#include "Domain/CoordinateMaps/Distribution.hpp"
#include "Options/Context.hpp"
#include "Utilities/Gsl.hpp"

namespace domain::creators {
/*!
 * \brief Given info about the radial distribution, set the number of shells and
 * the distribution for each shell.
 *
 * \details This is common code that all domain creators can use.
 */
void set_shell_distribution(
    gsl::not_null<size_t*> number_of_shells,
    gsl::not_null<std::vector<domain::CoordinateMaps::Distribution>*>
        radial_distribution,
    const std::vector<double>& radial_partitioning,
    const std::variant<domain::CoordinateMaps::Distribution,
                       std::vector<domain::CoordinateMaps::Distribution>>&
        input_radial_distribution,
    double innermost_shell_radius, double outermost_shell_radius,
    const std::string& innermost_radius_name,
    const std::string& outermost_radius_name,
    const Options::Context& context = {});
}  // namespace domain::creators
