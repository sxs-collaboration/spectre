// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <stdexcept>

/// \ingroup ErrorHandlingGroup
/// Exception indicating convergence failure
class convergence_error : public std::runtime_error {
 public:
  explicit convergence_error(const std::string& message)
      : runtime_error(message) {}
};
