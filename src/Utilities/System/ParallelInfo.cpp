// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <iomanip>
#include <sstream>

#include "Utilities/System/ParallelInfo.hpp"

namespace sys {

std::string pretty_wall_time(const double total_seconds) {
  // Subseconds don't really matter so just ignore them. This gives nice round
  // numbers.
  int total = static_cast<int>(total_seconds);
  const int day = total / (24 * 3600);

  total %= (24 * 3600);
  const int hour = total / 3600;

  total %= 3600;
  const int minutes = total / 60;

  total %= 60;
  const int seconds = total;

  std::stringstream ss{};
  ss << std::setfill('0');
  if (day > 0) {
    ss << std::setw(2) << day << "-";
  }

  // std::setw() isn't sticky so it has to be used for every insertion
  ss << std::setw(2) << hour << ":";
  ss << std::setw(2) << minutes << ":";
  ss << std::setw(2) << seconds;
  return ss.str();
}

std::string pretty_wall_time() { return pretty_wall_time(sys::wall_time()); }
}  // namespace sys
