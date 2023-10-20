// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

namespace sys {

/// \ingroup UtilitiesGroup
/// \brief Exit the program normally.
/// This should only be called once over all processors.
[[noreturn]] void exit(int exit_code = 0);

}  // namespace sys
