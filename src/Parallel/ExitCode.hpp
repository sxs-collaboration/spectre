// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"

namespace Parallel {

/*!
 * \brief Exit code of an executable
 *
 * \warning Don't change the integer values of the enum cases unless you have a
 * very good reason to do so. The integer values are used by external code, so
 * this is a public interface that should remain stable.
 */
enum class ExitCode : int {
  /// Program is complete
  Complete = 0,
  /// Program aborted because of an error
  Abort = 1,
  /// Program is incomplete and should be continued from the last checkpoint
  ContinueFromCheckpoint = 2
};

namespace Tags {

/*!
 * \brief Exit code of an executable
 *
 * \see Parallel::ExitCode
 */
struct ExitCode : db::SimpleTag {
  using type = Parallel::ExitCode;
};

}  // namespace Tags
}  // namespace Parallel
