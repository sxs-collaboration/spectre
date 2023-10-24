// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "ControlSystem/FutureMeasurements.hpp"
#include "DataStructures/DataBox/Tag.hpp"

namespace control_system::Tags {
/// Measurement times for a set of control systems sharing a
/// measurement.
template <typename ControlSystems>
struct FutureMeasurements : db::SimpleTag {
  using type = control_system::FutureMeasurements;
};
}  // namespace control_system::Tags
