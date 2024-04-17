// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/SingletonActions/UpdateAcceleration.hpp"

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/SelfForce.hpp"
#include "Evolution/Systems/CurvedScalarWave/Worldtube/Tags.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube {
void UpdateAcceleration::apply(
    const gsl::not_null<
        Variables<tmpl::list<::Tags::dt<Tags::EvolvedPosition<Dim>>,
                             ::Tags::dt<Tags::EvolvedVelocity<Dim>>>>*>
        dt_evolved_vars,
    const std::array<tnsr::I<double, Dim>, 2>& pos_vel,
    const tuples::TaggedTuple<gr::Tags::SpacetimeMetric<double, Dim>,
                              gr::Tags::InverseSpacetimeMetric<double, Dim>,
                              Tags::TimeDilationFactor>& background,
    const tnsr::I<double, Dim, Frame::Inertial>& geodesic_acc,
    const Scalar<double>& dt_psi_monopole,
    const tnsr::i<double, Dim, Frame::Inertial>& psi_dipole,
    const double charge, const std::optional<double> mass,
    const size_t max_iterations) {
  tnsr::I<double, Dim> self_force_acc(0.);
  const auto& particle_velocity = pos_vel.at(1);
  if (max_iterations > 0) {
    const auto& inverse_metric =
        get<gr::Tags::InverseSpacetimeMetric<double, Dim>>(background);
    const auto& dilation_factor = get<Tags::TimeDilationFactor>(background);
    self_force_acceleration(make_not_null(&self_force_acc), dt_psi_monopole,
                            psi_dipole, particle_velocity, charge, mass.value(),
                            inverse_metric, dilation_factor);
  }
  for (size_t i = 0; i < Dim; ++i) {
    get<::Tags::dt<Tags::EvolvedPosition<Dim>>>(*dt_evolved_vars).get(i)[0] =
        particle_velocity.get(i);
    get<::Tags::dt<Tags::EvolvedVelocity<Dim>>>(*dt_evolved_vars).get(i)[0] =
        geodesic_acc.get(i) + self_force_acc.get(i);
  }
}
}  // namespace CurvedScalarWave::Worldtube
