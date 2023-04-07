// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

/// \cond
class DataVector;
template <typename Frame>
class Strahlkorper;
namespace domain {
enum class ObjectLabel;
}  // namespace domain
/// \endcond

/// \ingroup ControlSystemGroup
/// All tags that will be used in the LinkedMessageQueue's within control
/// systems.
///
/// These tags will be used to retrieve the results of the measurements that
/// were sent to the control system which have been placed inside a
/// LinkedMessageQueue.
namespace control_system::QueueTags {
/// \ingroup ControlSystemGroup
/// Holds the centers of each horizon from measurements as DataVectors
template <::domain::ObjectLabel Horizon>
struct Center {
  using type = DataVector;
};

/// \ingroup ControlSystemGroup
/// Holds a full strahlkorper from measurements
template <typename Frame>
struct Strahlkorper {
  using type = ::Strahlkorper<Frame>;
};
}  // namespace control_system::QueueTags
