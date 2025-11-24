// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/PunctureField.hpp"

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/DynamicBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {

void puncture_field(
    gsl::not_null<Variables<tmpl::list<
        CurvedScalarWave::Tags::Psi, ::Tags::dt<CurvedScalarWave::Tags::Psi>,
        ::Tags::deriv<CurvedScalarWave::Tags::Psi, tmpl::size_t<3>,
                      Frame::Inertial>>>*>
        result,
    const tnsr::I<DataVector, 3, Frame::Inertial>& centered_coords,
    const tnsr::I<double, 3>& particle_position,
    const tnsr::I<double, 3>& particle_velocity,
    const tnsr::I<double, 3>& particle_acceleration, double bh_mass,
    size_t order, const std::array<double, 3>& bh_spin,
    const std::string& puncture_type) {
  if (puncture_type == "Schwarzschild") {
    ASSERT((bh_spin == std::array<double, 3>{0., 0., 0.}),
            "Only Schwarzschild puncture is fully implemented currently.");
    if (order == 0) {
      puncture_field_0(result, centered_coords, particle_position,
                       particle_velocity, particle_acceleration, bh_mass);
    } else if (order == 1) {
      puncture_field_1(result, centered_coords, particle_position,
                       particle_velocity, particle_acceleration, bh_mass);
    } else {
      ERROR(
          "The schwarzschild puncture field is only implemented up to "
          "expansion order 1 but "
          "you requested order "
          << order);
    }
  } else {
    if (order == 0) {
      puncture_field_kerr_0(result, centered_coords, particle_position,
                            particle_velocity, particle_acceleration, bh_mass,
                            bh_spin);
    } else {
      ERROR(
          "The kerr puncture field is only implemented up to "
          "expansion order "
          "0 but you requested order "
          << order);
    }
  }
}
}  // namespace CurvedScalarWave::Worldtube
