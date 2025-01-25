// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "ControlSystem/ControlErrors/GridCenters.hpp"

#include <cstddef>


#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace control_system::ControlErrors {
GridCenters::GridCenters(const Options::Context& /*context*/) {}

DataVector GridCenters::impl(const DataVector& fot_positions_dv,
                             const DataVector& measured_grid_position_of_A,
                             const DataVector& measured_grid_position_of_B) {
  tnsr::I<DataVector, 3, Frame::Grid> measured_grid_positions_tnsr{2_st, 0.0};
  for (size_t i = 0; i < 3; i++) {
    measured_grid_positions_tnsr.get(i)[0] = measured_grid_position_of_A[i];
    measured_grid_positions_tnsr.get(i)[1] = measured_grid_position_of_B[i];
  }

  DataVector control_error{6, 0.0};
  for (size_t i = 0; i < 2; i++) {
    const auto& measured_grid_position =
        i == 0 ? measured_grid_position_of_A : measured_grid_position_of_B;
    for (size_t j = 0; j < 3; j++) {
      control_error[i * 3 + j] =
          measured_grid_position[j] - fot_positions_dv[i * 3 + j];
    }
  }

  return control_error;
}

void GridCenters::pup(PUP::er& /*p*/) {}
}  // namespace control_system::ControlErrors
