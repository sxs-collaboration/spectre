// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"

namespace Frame {
struct Fluid;
} // namespace Frame

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
namespace Particles::MonteCarlo {

struct Packet;

/// Elastic scattering, which is just redrawing the momentum from
/// an isotropic distribution in the fluid frame, at constant
/// fluid frame energy.
void scatter_packet(
    gsl::not_null<Packet*> packet,
    gsl::not_null<std::mt19937*> random_number_generator,
    const double& fluid_frame_energy,
    const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_jacobian,
    const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_inverse_jacobian);

void diffuse_packet(gsl::not_null<Packet*> /*packet*/,
                    const double& /*time_step*/);

}  // namespace Particles::MonteCarlo
