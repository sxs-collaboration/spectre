// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/UpdateAcceleration.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {
void UpdateAcceleration::apply(
    const gsl::not_null<
        Variables<tmpl::list<::Tags::dt<Tags::EvolvedPosition<Dim>>,
                             ::Tags::dt<Tags::EvolvedVelocity<Dim>>>>*>
        dt_evolved_vars,
    const std::array<tnsr::I<double, Dim>, 2>& pos_vel,
    const tnsr::I<double, Dim, Frame::Inertial>& geodesic_acc) {
  const auto& particle_velocity = pos_vel.at(1);
  for (size_t i = 0; i < Dim; ++i) {
    get<::Tags::dt<Tags::EvolvedPosition<Dim>>>(*dt_evolved_vars).get(i)[0] =
        particle_velocity.get(i);
    get<::Tags::dt<Tags::EvolvedVelocity<Dim>>>(*dt_evolved_vars).get(i)[0] =
        geodesic_acc.get(i);
  }
}
}  // namespace CurvedScalarWave::Worldtube
