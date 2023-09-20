// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <stdexcept>

/// \ingroup ErrorHandlingGroup
/// Exception indicating an ASSERT failed
class SpectreAssert : public std::runtime_error {
 public:
  explicit SpectreAssert(const std::string& message) : runtime_error(message) {}
};

/// \ingroup ErrorHandlingGroup
/// Exception indicating an ERROR was triggered
class SpectreError : public std::runtime_error {
 public:
  explicit SpectreError(const std::string& message) : runtime_error(message) {}
};

/// \ingroup ErrorHandlingGroup
/// Exception indicating an ERROR was triggered because of an FPE
///
/// \note You cannot rely on catching this exception for recovering from
/// FPEs because not all compilers and hardware properly support throwing
/// exceptions on FPEs.
class SpectreFpe : public std::runtime_error {
 public:
  explicit SpectreFpe(const std::string& message) : runtime_error(message) {}
};

/// \ingroup ErrorHandlingGroup
/// Exception indicating convergence failure
class convergence_error : public std::runtime_error {
 public:
  explicit convergence_error(const std::string& message)
      : runtime_error(message) {}
};
