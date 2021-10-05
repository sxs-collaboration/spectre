// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Helpers/PointwiseFunctions/PostNewtonian/BinaryTrajectories.hpp"

#include <array>
#include <cmath>
#include <utility>

#include "Utilities/ConstantExpressions.hpp"

BinaryTrajectories::BinaryTrajectories(double initial_separation)
    : initial_separation_fourth_power_{square(square(initial_separation))} {}

double BinaryTrajectories::separation(const double time) const {
  return pow(initial_separation_fourth_power_ - 12.8 * time, 0.25);
}

double BinaryTrajectories::orbital_frequency(const double time) const {
  return pow(separation(time), -1.5);
}

std::pair<std::array<double, 3>, std::array<double, 3>>
BinaryTrajectories::positions(const double time) const {
  const double sep = separation(time);
  const double omega = orbital_frequency(time);
  const double x1 = 0.5 * sep * cos(omega * time);
  const double y1 = 0.5 * sep * sin(omega * time);
  return {{x1, y1, 0.0}, {-x1, -y1, 0.0}};
}
