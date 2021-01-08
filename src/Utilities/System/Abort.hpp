// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

namespace sys {
/// \ingroup ErrorHandlingGroup
/// Abort the program with an error message.
///
/// \details This function calls CkExit with a non-zero argument to indicate a
/// failure, unless the SPECTRE_TRAP_ON_ERROR environmental variable is set, in
/// which case it raises SIGTRAP.
[[noreturn]] void abort(const std::string& message);
}  // namespace sys
