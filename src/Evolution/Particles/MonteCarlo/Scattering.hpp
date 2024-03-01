// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

namespace Frame {
struct Fluid;
}  // namespace Frame

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
namespace Particles::MonteCarlo {

struct Packet;

/// Precomputed quantities useful for the diffusion approximation
/// in high-scattering opacity regions.
/// We follow \cite Foucart:2017mbt
/// Note that r_d in that manuscript should be
/// (distance diffused)/(time elapsed)
/// i.e. the paper is missing a normalization
/// of r_d by delta_t. The upper bound of the integral in Eq (30)
/// should just be sqrt(3*tau/4)
/// All quantities in this struct are fixed; we could also just
/// hard-code them to save us the on-the-fly calculation, or
/// have a single instance of this struct in the global cache.
struct DiffusionMonteCarloParameters {
  DiffusionMonteCarloParameters();

  const double MinimumOpacityForDiffusion = 3.0;
  const double MaximumOpacityForCorrection = 11.0;
  /// Definition of the vector B_i, equation (35)
  const std::array<double, 21> BvsRhoForScattering{
      1000., 18.74, 7.52, 4.75, 3.51, 2.78, 2.32, 2.00,  1.77,   1.60,     1.47,
      1.36,  1.28,  1.21, 1.15, 1.10, 1.07, 1.04, 1.019, 1.0027, 1.0000001};
  /// Storage for the function r(P) implicitly defined by Eq (29)
  /// ScatteringRofP[i] = r(0.001*i)
  /// Calculation performed in constructor
  std::array<double, 1001> ScatteringRofP;
  /// Storage for the opacity dependent correction on the
  /// right-hand side of Eq (30)
  /// We use 101 points between the min and max opacities
  /// defined above.
  /// Calculation performed in constructor
  std::array<double, 101> OpacityDependentCorrectionToRofP;
};

/// Precompute quantities needed for evolving packets in the diffusion
/// approximation, i.e. the transport velocity in logical coordinates
/// and the coefficients defined by Eq (31-33)
void DiffusionPrecomputeForElement(
    gsl::not_null<DataVector*> prefactor_diffusion_time_vector,
    gsl::not_null<DataVector*> prefactor_diffusion_four_velocity,
    gsl::not_null<DataVector*> prefactor_diffusion_time_step,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);

/// Elastic scattering, which is just redrawing the momentum from
/// an isotropic distribution in the fluid frame, at constant
/// fluid frame energy.
std::array<double, 4> scatter_packet(
    gsl::not_null<Packet*> packet,
    gsl::not_null<std::mt19937*> random_number_generator,
    const double& fluid_frame_energy,
    const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_jacobian,
    const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_inverse_jacobian);

/// Evolve a packet for dt = time_step, assuming that we can use
/// the diffusion approximation.
void diffuse_packet(
    gsl::not_null<Packet*> packet,
    gsl::not_null<std::mt19937*> random_number_generator,
    gsl::not_null<double*> neutrino_energy, double time_step,
    const DiffusionMonteCarloParameters& diffusion_params,
    double scattering_opacity, const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& d_lapse,
    const tnsr::iJ<DataVector, 3, Frame::Inertial>& d_shift,
    const tnsr::iJJ<DataVector, 3, Frame::Inertial>& d_inv_spatial_metric,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>& mesh_velocity,
    const InverseJacobian<DataVector, 3, Frame::ElementLogical,
                          Frame::Inertial>&
        inverse_jacobian_logical_to_inertial,
    const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_jacobian,
    const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_inverse_jacobian,
    const DataVector& prefactor_diffusion_time_step,
    const DataVector& prefactor_diffusion_four_velocity,
    const DataVector& prefactor_diffusion_time_vector);

}  // namespace Particles::MonteCarlo
