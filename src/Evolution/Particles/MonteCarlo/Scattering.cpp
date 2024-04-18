// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Particles/MonteCarlo/Scattering.hpp"

#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "Evolution/Particles/MonteCarlo/CouplingTermsForPropagation.hpp"
#include "Evolution/Particles/MonteCarlo/EvolvePackets.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"

using hydro::units::nuclear::proton_mass;

namespace Particles::MonteCarlo {

// All equations refer to Foucart 2018 (10.1093/mnras/sty108)
DiffusionMonteCarloParameters::DiffusionMonteCarloParameters()
    : ScatteringRofP(std::array<double, 1001>()),
      OpacityDependentCorrectionToRofP(std::array<double, 101>()) {
  // Calculate function r(P), implicitly defined by Eq. (30)
  // We integrate using the trapezoid rule to find where
  // r(P) = i * 0.001
  const double dr = 1.e-5;
  double r = dr;
  double integrand_low = 0.0;
  double integrand_high = 4. / sqrt(M_PI) * square(r) * exp(-square(r));
  double integral = 0.5 * (integrand_low + integrand_high) * dr;
  gsl::at(ScatteringRofP, 0) = 0.0;
  const size_t p_pts = ScatteringRofP.size();
  for (size_t i = 1; i < p_pts - 1; i++) {
    while (integral * static_cast<double>(p_pts - 1) < static_cast<double>(i)) {
      r += dr;
      integrand_low = integrand_high;
      integrand_high = 4. / sqrt(M_PI) * square(r) * exp(-square(r));
      integral += 0.5 * (integrand_low + integrand_high) * dr;
    }
    gsl::at(ScatteringRofP, i) = r - 0.5 * dr;
  }
  // Technically, r(1) = infinity... we instead interpolate from the last
  // two points (and will later enforce causality to make sure that
  // we never `diffuse' packets faster than the speed of light).
  gsl::at(ScatteringRofP, p_pts - 1) =
      2.0 * gsl::at(ScatteringRofP, p_pts - 2) -
      gsl::at(ScatteringRofP, p_pts - 3);

  // For optical depth between MinimumOpacityForDiffusion and
  // MaximumOpacityForCorrection, we renormalize r(P)
  // so that there is a probability exp(-tau) of superluminal motion,
  // which is interpreted as the absence of scattering events.
  // See Eq.(30); we precompute here the numerical factor on the
  // rhs of Eq (30) as a function of the optical depth.
  const size_t tau_pts = OpacityDependentCorrectionToRofP.size();
  const double dTau =
      (MaximumOpacityForCorrection - MinimumOpacityForDiffusion) /
      (static_cast<double>(tau_pts) - 1.);
  for (size_t nt = 0; nt < tau_pts; nt++) {
    const double cTau =
        MinimumOpacityForDiffusion + static_cast<double>(nt) * dTau;
    const double TargetRho = sqrt(3. * cTau / 4.);
    // Find A such that F[TargetRho]=A
    // i.e. ScatteringRofP[A]=TargetRho
    // Note that as TargetRho > 0 and ScatteringRofP[0]=0
    // upper_bracket >= 1; so there is always a previous
    // element available.
    const auto upper_bracket = std::lower_bound(
        ScatteringRofP.begin(), ScatteringRofP.end(), TargetRho);
    if (upper_bracket == ScatteringRofP.end()) {
      gsl::at(OpacityDependentCorrectionToRofP, nt) = (1. - exp(-cTau));
    } else {
      // Linear interpolation between two closest points to solution.
      // The division by (p_pts - 1) is needed to get the normalization
      // right (i.e. a number in [0,1]).
      const auto lower_bracket = std::prev(upper_bracket);
      const auto lower_index = static_cast<size_t>(
          std::distance(ScatteringRofP.begin(), lower_bracket));
      const size_t upper_index = lower_index + 1;
      const double TargetF =
          ((gsl::at(ScatteringRofP, upper_index) - TargetRho) *
               static_cast<double>(lower_index) +
           (TargetRho - gsl::at(ScatteringRofP, lower_index)) *
               static_cast<double>(upper_index)) /
          (gsl::at(ScatteringRofP, upper_index) -
           gsl::at(ScatteringRofP, lower_index)) /
          static_cast<double>(p_pts - 1);
      gsl::at(OpacityDependentCorrectionToRofP, nt) =
          TargetF / (1. - exp(-cTau));
    }
  }
}

void DiffusionPrecomputeForElement(
    gsl::not_null<DataVector*> prefactor_diffusion_time_vector,
    gsl::not_null<DataVector*> prefactor_diffusion_four_velocity,
    gsl::not_null<DataVector*> prefactor_diffusion_time_step,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric) {
  // g_tt, stored in prefactor_diffusion_time_step
  // g_tt = \alpha^2 - \beta^i \beta_i
  *prefactor_diffusion_time_step =
      -square(get(lapse)) + get(dot_product(shift, shift, spatial_metric));
  // u_t stored in prefactor_diffusion_four_velocity
  // u_t = - W * (lapse - beta^i v_i)
  *prefactor_diffusion_four_velocity =
      get(dot_product(shift, lower_spatial_four_velocity)) -
      get(lorentz_factor) * get(lapse);
  // Precompute u_t^2 + g_tt
  *prefactor_diffusion_time_vector =
      square(*prefactor_diffusion_four_velocity) +
      (*prefactor_diffusion_time_step);

  for (size_t i = 0; i < prefactor_diffusion_time_vector->size(); i++) {
    // First, deal with case with small transport velocity. In that case,
    // Eq 32 is singular, but we also do not need to move packets with
    // the fluid in the diffusion step (as the fluid is not moving on
    // the grid). Set prefactor_diffusion_time_step to -1, which we will
    // check for when performing diffusion.
    if (fabs((*prefactor_diffusion_time_vector)[i]) < 1.e-10) {
      (*prefactor_diffusion_time_step)[i] = -1.0;
      (*prefactor_diffusion_time_vector)[i] =
          std::numeric_limits<double>::signaling_NaN();
      (*prefactor_diffusion_four_velocity)[i] =
          std::numeric_limits<double>::signaling_NaN();
      continue;
    }
    // Eq 31 stored in prefactor_diffusion_time_step
    // A/B = - (u_t + sqrt(u_t^2 + g_tt))/g_tt
    const double A_over_B = -((*prefactor_diffusion_four_velocity)[i] +
                              sqrt((*prefactor_diffusion_time_vector)[i])) /
                            (*prefactor_diffusion_time_step)[i];
    // Eq. 32 for neutrinos of 1MeV energy (the result scale linearly with
    // energy, and can thus be modified as needed during packet evolution) B = 1
    // / (1 - (A/B) u_t)
    (*prefactor_diffusion_four_velocity)[i] =
        1.0 / (1.0 - A_over_B * (*prefactor_diffusion_four_velocity)[i]);
    // A from Eq. 32 and 31
    // A= (A/B) * B
    (*prefactor_diffusion_time_vector)[i] =
        A_over_B * (*prefactor_diffusion_four_velocity)[i];
    // Eq. 33
    // prefactor_diffusion_time_step = (A + B u^t)/(B u^t) = 1 + A*alpha/(B*W)
    (*prefactor_diffusion_time_step)[i] =
        (get(lapse)[i] * A_over_B / get(lorentz_factor)[i]) + 1.0;
  }
}

std::array<double, 4> scatter_packet(
    const gsl::not_null<Packet*> packet,
    const gsl::not_null<std::mt19937*> random_number_generator,
    const double& fluid_frame_energy,
    const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_jacobian,
    const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_inverse_jacobian) {
  const size_t& idx = packet->index_of_closest_grid_point;
  std::uniform_real_distribution<double> rng_uniform_zero_to_one(0.0, 1.0);
  const double cos_theta =
      2.0 * rng_uniform_zero_to_one(*random_number_generator) - 1.0;
  const double phi =
      2.0 * M_PI * rng_uniform_zero_to_one(*random_number_generator);
  const double cos_phi = cos(phi);
  const double sin_phi = sin(phi);
  const double sin_theta = sqrt(1.0 - cos_theta * cos_theta);
  std::array<double, 4> four_momentum_fluid{
      fluid_frame_energy, sin_theta * cos_phi * fluid_frame_energy,
      sin_theta * sin_phi * fluid_frame_energy, cos_theta * fluid_frame_energy};

  packet->momentum_upper_t = 0.0;
  for (size_t d = 0; d < 4; d++) {
    packet->momentum_upper_t +=
        inertial_to_fluid_inverse_jacobian.get(0, d)[idx] *
        gsl::at(four_momentum_fluid, d);
  }
  for (size_t d = 0; d < 3; d++) {
    // Multiply by -1.0 because p_t = -p^t in an orthonormal frame
    packet->momentum.get(d) = -gsl::at(four_momentum_fluid, 0) *
                              inertial_to_fluid_jacobian.get(d + 1, 0)[idx];
    for (size_t dd = 0; dd < 3; dd++) {
      packet->momentum.get(d) +=
          gsl::at(four_momentum_fluid, dd + 1) *
          inertial_to_fluid_jacobian.get(d + 1, dd + 1)[idx];
    }
  }
  return std::array<double, 4>{cos_theta, sin_theta, cos_phi, sin_phi};
}

void diffuse_packet(
    const gsl::not_null<Packet*> packet,
    const gsl::not_null<std::mt19937*> random_number_generator,
    const gsl::not_null<double*> neutrino_energy,
    const gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
    coupling_tilde_s,
    const gsl::not_null<Scalar<DataVector>*> coupling_rho_ye,
    const double time_step,
    const DiffusionMonteCarloParameters& diffusion_params,
    const double absorption_opacity, const double scattering_opacity,
    const Scalar<DataVector>& lorentz_factor,
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
    const DataVector& prefactor_diffusion_time_vector) {
  const size_t idx = packet->index_of_closest_grid_point;
  double scattering_optical_depth = scattering_opacity * time_step *
                                    get(lapse)[idx] * get(lorentz_factor)[idx];
  if (scattering_optical_depth <
      diffusion_params.MinimumOpacityForDiffusion * (1.0 - 1.e-12)) {
    ERROR("Optical depth too low for diffusion approximation");
  }
  if (scattering_optical_depth < diffusion_params.MinimumOpacityForDiffusion) {
    scattering_optical_depth = diffusion_params.MinimumOpacityForDiffusion;
  }
  // Draw random number P in Eq (29)
  std::uniform_real_distribution<double> rng_uniform_zero_to_one(0.0, 1.0);
  double p_diffusion = rng_uniform_zero_to_one(*random_number_generator);
  // Correct for finite opacity, as in Eq (30)
  if (scattering_optical_depth < diffusion_params.MaximumOpacityForCorrection) {
    // Find location of current opacity in vector of stored value
    // As we only reach this point if the opacity is between
    // MinimumOpacityForDiffusion and MaximumOpacityForCorrection,
    // nti and nti+1 are guaranteed to be in bounds.
    const double nt =
        (scattering_optical_depth -
         diffusion_params.MinimumOpacityForDiffusion) /
        (diffusion_params.MaximumOpacityForCorrection -
         diffusion_params.MinimumOpacityForDiffusion) *
        static_cast<double>(
            diffusion_params.OpacityDependentCorrectionToRofP.size() - 1);
    const auto nti = static_cast<size_t>(floor(nt));
    // Interpolate linearly between closest value
    const double correction_factor =
        gsl::at(diffusion_params.OpacityDependentCorrectionToRofP, nti) *
            (static_cast<double>(nti) + 1. - nt) +
        gsl::at(diffusion_params.OpacityDependentCorrectionToRofP, nti + 1) *
            (nt - static_cast<double>(nti));
    p_diffusion *= correction_factor;
  }

  if (p_diffusion < 0.0) {
    ERROR("p_diffusion should be a positive number");
  }
  // Effective speed of the packet in the fluid frame as it diffuses
  double dr_over_dt_free = 1.0;
  // If p_diffusion >= 1.0, we assume no scattering (maintains causality)
  // Otherwise, apply normalization with optical depth to calculate dr/dt
  // i.e. Eq 28
  if (p_diffusion < 1.0) {
    // As p_diffusion is between 0 and 1 (not included) nri and nri+1
    // are guaranteed to be in bounds.
    const double nr =
        p_diffusion *
        static_cast<double>(diffusion_params.ScatteringRofP.size() - 1);
    const auto nri = static_cast<size_t>(floor(nr));
    dr_over_dt_free = sqrt(4.0 / 3.0 / scattering_optical_depth) *
                      (gsl::at(diffusion_params.ScatteringRofP, nri + 1) *
                           (nr - static_cast<double>(nri)) +
                       gsl::at(diffusion_params.ScatteringRofP, nri) *
                           (static_cast<double>(nri + 1) - nr));
    dr_over_dt_free = std::min(1.0, dr_over_dt_free);
  }

  // Evolution comoving with fluid
  // We first evolve the packet along a null geodesic in such a way as to end up
  // at the same location as the fluid after (1.0 - dr_over_dt_free) * time_step
  const double dt_comoving =
      (1.0 - dr_over_dt_free) * time_step * prefactor_diffusion_time_step[idx];
  // First case, prefactor_diffusion_time_step>0, which correpsonds to non-zero
  // transport velocities
  if (dt_comoving > 0.0) {
    // Calculate p_i = nu * (prefactor_diffusion_four_velocity* u_i +
    // prefactor_diffusion_time_vector * shift_i) We do not need p^t, which is
    // recalculated by evolve_single_packet_on_geodesic
    for (size_t i = 0; i < 3; i++) {
      packet->momentum[i] = (prefactor_diffusion_four_velocity[idx] *
                             lower_spatial_four_velocity.get(i)[idx]);
      for (size_t j = 0; j < 3; j++) {
        packet->momentum[i] += prefactor_diffusion_time_vector[idx] *
                               spatial_metric.get(i, j)[idx] *
                               shift.get(j)[idx];
      }
      packet->momentum[i] *= (*neutrino_energy);
    }
    evolve_single_packet_on_geodesic(packet, dt_comoving, lapse, shift, d_lapse,
                                     d_shift, d_inv_spatial_metric,
                                     inv_spatial_metric, mesh_velocity,
                                     inverse_jacobian_logical_to_inertial);
    // For the rest of the 'comoving time', the packet is modeled as remaining
    // stationary.
    packet->time += (1.0 - dr_over_dt_free) * time_step - dt_comoving;
    packet->renormalize_momentum(inv_spatial_metric, lapse);
    (*neutrino_energy) = compute_fluid_frame_energy(*packet, lorentz_factor,
                                                    lower_spatial_four_velocity,
                                                    lapse, inv_spatial_metric);
  } else {
    // Zero transport velocity; we only need to update the time.
    packet->time += (1.0 - dr_over_dt_free) * time_step;
    // ... and potentially correct for grid motion
    if (mesh_velocity.has_value()) {
      for (size_t i = 0; i < 3; i++) {
        for (size_t j = 0; j < 3; j++) {
          packet->coordinates[i] -=
              mesh_velocity.value().get(j)[idx] * (1.0 - dr_over_dt_free) *
              time_step * inverse_jacobian_logical_to_inertial.get(i, j)[idx];
        }
      }
    }
  }
  // Coupling with the fluid for a packet comoving with the fluid
  // Energy coupling
  const double energy_coupling = (1.0 - dr_over_dt_free) * time_step *
    get(lapse)[idx] * absorption_opacity * packet->number_of_neutrinos *
    (*neutrino_energy);
  coupling_tilde_tau->get()[idx] += energy_coupling;
  // Momentum coupling term
  for (size_t d = 0; d < 3; d++) {
    coupling_tilde_s->get(d)[idx] += energy_coupling /
      get(lorentz_factor)[idx] *
      lower_spatial_four_velocity.get(d)[idx];
  }
  // Lepton number coupling term
  if (packet->species < 2) {
    coupling_rho_ye->get()[idx] +=
      (packet->species == 0 ? 1.0 : -1.0) * proton_mass *
      energy_coupling / (*neutrino_energy) / get(lorentz_factor)[idx];
  }

  // Scatter in fluid frame. Output array contains (cos_theta, sin_theta,
  // cos_phi, sin_phi) for direction of momentum in fluid frame spherical
  // coordinates.
  const std::array<double, 4> fluid_frame_angles = scatter_packet(
      packet, random_number_generator, *neutrino_energy,
      inertial_to_fluid_jacobian, inertial_to_fluid_inverse_jacobian);

  // Free evolution away from fluid frame
  // We correct the remaining time so that all packets are evolved for the
  // same time in the fluid frame, and not the grid frame. This is needed
  // to recover the proper average velocity of the packets.
  const double dt_corr =
      std::clamp(get(lapse)[idx] / get(lorentz_factor)[idx] *
                     packet->momentum_upper_t / (*neutrino_energy),
                 0.0, 2.0);
  const double dt_free = dr_over_dt_free * time_step * dt_corr;
  evolve_single_packet_on_geodesic(
      packet, dt_free, lapse, shift, d_lapse, d_shift, d_inv_spatial_metric,
      inv_spatial_metric, mesh_velocity, inverse_jacobian_logical_to_inertial);
  packet->renormalize_momentum(inv_spatial_metric, lapse);
  (*neutrino_energy) = compute_fluid_frame_energy(*packet, lorentz_factor,
                                                  lower_spatial_four_velocity,
                                                  lapse, inv_spatial_metric);
  // Add coupling term for free propagation
  const double& lapse_packet = get(lapse)[idx];
  const double& lorentz_factor_packet = get(lorentz_factor)[idx];
  const std::array<double, 3> lower_spatial_four_velocity_packet = {
    lower_spatial_four_velocity.get(0)[idx],
    lower_spatial_four_velocity.get(1)[idx],
    lower_spatial_four_velocity.get(2)[idx]};
  AddCouplingTermsForPropagation(
    coupling_tilde_tau, coupling_tilde_s, coupling_rho_ye, *packet,
    dt_free, absorption_opacity, 0.0, (*neutrino_energy),
    lapse_packet, lorentz_factor_packet,
    lower_spatial_four_velocity_packet);


  // Get final momentum in fluid frame then transform to inertial frame
  const size_t b_pts = diffusion_params.BvsRhoForScattering.size();
  const double rb = dr_over_dt_free * static_cast<double>(b_pts - 1);
  const auto nb = static_cast<size_t>(floor(rb));
  // Two cases used to avoid potential out-of-bound indexing
  // when dr_over_dt_free is exactly 1 (its maximum potential value)
  const double B =
      (nb == b_pts - 1)
          ? gsl::at(diffusion_params.BvsRhoForScattering, nb)
          : gsl::at(diffusion_params.BvsRhoForScattering, nb) *
                    (static_cast<double>(nb) + 1.0 - rb) +
                gsl::at(diffusion_params.BvsRhoForScattering, nb + 1) *
                    (rb - static_cast<double>(nb));
  // Theta2, phi2 = angles for packet propagation in spherical coordinates
  // with axis along the direction of propagation of the packet after the
  // first scattering event; Eq. (34)
  double random_number = rng_uniform_zero_to_one(*random_number_generator);
  const double cth2 = std::clamp(
      B - (B + 1.) * exp(random_number * log((B - 1.) / (B + 1.))), -1.0, 1.0);
  const double sth2 = sqrt(1.0 - square(cth2));
  random_number =
      rng_uniform_zero_to_one(*random_number_generator) * 2.0 * M_PI;
  const double cph2 = cos(random_number);
  const double sph2 = sin(random_number);

  const auto& [cth, sth, cph, sph] = fluid_frame_angles;

  // Final fluid frame momentum (between Eq 33 and 34)
  std::array<double, 4> final_fluid_frame_momentum{
      (*neutrino_energy),
      (*neutrino_energy) *
          (-sth2 * cph2 * cth * cph + sth2 * sph2 * sph + cth2 * sth * cph),
      (*neutrino_energy) *
          (-sth2 * cph2 * cth * sph - sth2 * sph2 * cph + cth2 * sth * sph),
      (*neutrino_energy) * (sth2 * cph2 * sth + cth2 * cth)};
  // Convert from fluid to inertial frame
  packet->momentum_upper_t = 0.0;
  for (size_t d = 0; d < 4; d++) {
    packet->momentum_upper_t +=
        inertial_to_fluid_inverse_jacobian.get(0, d)[idx] *
        gsl::at(final_fluid_frame_momentum, d);
  }
  for (size_t d = 0; d < 3; d++) {
    // Multiply by -1.0 because p_t = -p^t in an orthonormal frame
    packet->momentum.get(d) = -gsl::at(final_fluid_frame_momentum, 0) *
                              inertial_to_fluid_jacobian.get(d + 1, 0)[idx];
    for (size_t dd = 0; dd < 3; dd++) {
      packet->momentum.get(d) +=
          gsl::at(final_fluid_frame_momentum, dd + 1) *
          inertial_to_fluid_jacobian.get(d + 1, dd + 1)[idx];
    }
  }
}

}  // namespace Particles::MonteCarlo
