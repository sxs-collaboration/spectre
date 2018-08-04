// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <sstream>
#include <string>

/*!
 * \ingroup UtilitiesGroup
 * \brief Get the streamed output of `t` as a `std::string`
 */
template <typename T>
std::string get_output(const T& t) noexcept {
  std::ostringstream os;
  os << t;
  return os.str();
}
