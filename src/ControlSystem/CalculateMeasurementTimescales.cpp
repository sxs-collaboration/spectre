// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/CalculateMeasurementTimescales.hpp"

#include "ControlSystem/Controller.hpp"
#include "ControlSystem/TimescaleTuner.hpp"
#include "DataStructures/DataVector.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace control_system {
template <size_t DerivOrder, bool AllowDecrease>
DataVector calculate_measurement_timescales(
    const ::Controller<DerivOrder>& controller,
    const ::TimescaleTuner<AllowDecrease>& tuner,
    const int measurements_per_update) {
  return tuner.current_timescale() * controller.get_update_fraction() /
         static_cast<double>(measurements_per_update);
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define ALLOWDECREASE(data) BOOST_PP_TUPLE_ELEM(1, data)

#define INSTANTIATE(_, data)                              \
  template DataVector calculate_measurement_timescales(   \
      const ::Controller<DIM(data)>& controller,          \
      const ::TimescaleTuner<ALLOWDECREASE(data)>& tuner, \
      const int measurements_per_update);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (true, false))

#undef INSTANTIATE
#undef DIM
#undef ALLOWDECREASE

}  // namespace control_system
