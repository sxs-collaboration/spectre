// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <memory>
#include <pup.h>

#include "ControlSystem/ControlErrors/Size/AhSpeed.hpp"
#include "ControlSystem/ControlErrors/Size/DeltaR.hpp"
#include "ControlSystem/ControlErrors/Size/Initial.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"

namespace control_system::size {
void register_derived_with_charm() {
  Parallel::register_classes_with_charm<States::Initial, States::AhSpeed,
                                        States::DeltaR>();
}
}  // namespace control_system::size
