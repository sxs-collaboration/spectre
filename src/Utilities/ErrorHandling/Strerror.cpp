// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Utilities/ErrorHandling/Strerror.hpp"

#include <array>
#include <string>

#include "Utilities/ErrorHandling/StrerrorWrapper.h"

std::string strerror_threadsafe(const int errnum) {
  // The documentation gives no guidance on how much space to reserve,
  // but a typical English result is "No such file or directory".
  std::array<char, 1000> message{};
  if (spectre_strerror_r(errnum, message.data(), message.size()) != 0) {
    return "strerror failed";
  }
  return {message.data()};
}
