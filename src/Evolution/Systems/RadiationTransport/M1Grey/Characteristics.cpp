// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/RadiationTransport/M1Grey/Characteristics.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "Utilities/Gsl.hpp"

namespace RadiationTransport {
namespace M1Grey {

void characteristic_speeds(
    const gsl::not_null<std::array<DataVector, 4>*> pchar_speeds,
    const Scalar<DataVector>& lapse) {
  const size_t num_grid_points = get(lapse).size();
  auto& char_speeds = *pchar_speeds;
  if (char_speeds[0].size() != num_grid_points) {
    char_speeds[0] = DataVector(num_grid_points);
  }
  char_speeds[0] = 1.;
  if (char_speeds[1].size() != num_grid_points) {
    char_speeds[1] = DataVector(num_grid_points);
  }
  char_speeds[1] = -1.;
  for (size_t i = 2; i < 4; i++) {
    gsl::at(char_speeds, i) = char_speeds[0];
  }
}
}  // namespace M1Grey
}  // namespace RadiationTransport
