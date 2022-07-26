// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <iosfwd>
#include <optional>
#include <tuple>

namespace Parallel {

/// The possible options for altering the current execution of the algorithm,
/// used in the return type of iterable actions.
enum class AlgorithmExecution {
  /// Continue executing iterable actions.
  Continue,
  /// Temporarily stop executing iterable actions, but try the same
  /// action again after receiving data from other distributed
  /// objects.
  Retry,
  /// Stop the execution of iterable actions, but allow entry methods
  /// (communication) to explicitly request restarting the execution.
  Pause,
  /// Stop the execution of iterable actions and do not allow their execution
  /// until after a phase change. Simple actions will still execute.
  Halt
};

/// Return type for iterable actions.  The std::optional can be used to specify
/// the index of the action to be called next in the phase dependent action
/// list.  Passing std::nullopt will execute the next action in the list.
using iterable_action_return_t =
    std::tuple<AlgorithmExecution, std::optional<size_t>>;
}  // namespace Parallel
