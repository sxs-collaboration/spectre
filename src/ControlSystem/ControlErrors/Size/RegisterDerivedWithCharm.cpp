// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/Size/RegisterDerivedWithCharm.hpp"

#include <memory>
#include <pup.h>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace control_system::size {
void register_derived_with_charm() {
  register_classes_with_charm<States::Initial, States::AhSpeed,
                              States::DeltaR>();
}
}  // namespace control_system::size
