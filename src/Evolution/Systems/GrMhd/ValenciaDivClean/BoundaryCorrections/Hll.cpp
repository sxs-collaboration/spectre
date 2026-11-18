// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GrMhd/ValenciaDivClean/BoundaryCorrections/Hll.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <pup.h>

#include <memory>
#include <optional>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/EagerMath/RaiseOrLowerIndex.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "PointwiseFunctions/Hydro/SoundSpeedSquared.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/CaptureForError.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"

namespace grmhd::ValenciaDivClean::BoundaryCorrections {
Hll::Hll(const double magnetic_field_magnitude_for_hydro,
         const double light_speed_density_cutoff)
    : magnetic_field_magnitude_for_hydro_(magnetic_field_magnitude_for_hydro),
      light_speed_density_cutoff_(light_speed_density_cutoff) {}

Hll::Hll(CkMigrateMessage* /*unused*/) {}

std::unique_ptr<evolution::BoundaryCorrection> Hll::get_clone() const {
  return std::make_unique<Hll>(*this);
}

void Hll::pup(PUP::er& p) {
  BoundaryCorrection::pup(p);
  p | magnetic_field_magnitude_for_hydro_;
  p | light_speed_density_cutoff_;
}

double Hll::dg_package_data(
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_normal_dot_flux_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> packaged_normal_dot_flux_tilde_phi,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*>
        packaged_largest_ingoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_fast_outgoing_char_speed,
    const gsl::not_null<Scalar<DataVector>*> packaged_fast_ingoing_char_speed,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        packaged_interface_unit_normal,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        packaged_interface_unit_normal_vector,

    const Scalar<DataVector>& tilde_d, const Scalar<DataVector>& tilde_ye,
    const Scalar<DataVector>& tilde_tau,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
    const Scalar<DataVector>& tilde_phi,

    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_d,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_ye,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_tau,
    const tnsr::Ij<DataVector, 3, Frame::Inertial>& flux_tilde_s,
    const tnsr::IJ<DataVector, 3, Frame::Inertial>& flux_tilde_b,
    const tnsr::I<DataVector, 3, Frame::Inertial>& flux_tilde_phi,

    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::i<DataVector, 3, Frame::Inertial>& spatial_velocity_one_form,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,

    const Scalar<DataVector>& rest_mass_density,
    const Scalar<DataVector>& electron_fraction,
    const Scalar<DataVector>& temperature,
    const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,

    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_covector,
    const tnsr::I<DataVector, 3, Frame::Inertial>& /*normal_vector*/,
    const std::optional<tnsr::I<DataVector, 3, Frame::Inertial>>&
    /*mesh_velocity*/,
    const std::optional<Scalar<DataVector>>& normal_dot_mesh_velocity,
    const EquationsOfState::EquationOfState<true, 3>& equation_of_state) const {
  const bool has_b_field =
      max(get(magnitude(tilde_b))) > magnetic_field_magnitude_for_hydro_;
  {
    Scalar<DataVector> shift_dot_normal = tilde_d;
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);

    // Initialize characteristic speeds to light speed
    // (default choice)
    get(*packaged_largest_outgoing_char_speed) =
        get(lapse) - get(shift_dot_normal);
    get(*packaged_largest_ingoing_char_speed) =
        -get(lapse) - get(shift_dot_normal);

    if (not has_b_field and
        (max(get(rest_mass_density)) > light_speed_density_cutoff_)) {
      // Since we have no magnetic field (and we have grid points not in the
      // atmosphere), we can reduce the char speeds to the hydro speeds only.
      // This makes the scheme less dissipative.
      //
      // Calculate sound speed squared
      const size_t num_points = get(rest_mass_density).size();
      Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                           ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                           ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
                           ::Tags::TempScalar<6>>>
          temp_buffer{num_points};
      auto& v_dot_normal = get<::Tags::TempScalar<0>>(temp_buffer);
      auto& v_squared = get<::Tags::TempScalar<1>>(temp_buffer);
      auto& discriminant = get(get<::Tags::TempScalar<2>>(temp_buffer));
      auto& one_minus_v2_cs2 = get<::Tags::TempScalar<3>>(temp_buffer);
      auto& one_minus_cs2 = get<::Tags::TempScalar<4>>(temp_buffer);
      auto& lapse_over_one_minus_v2_cs2 =
          get<::Tags::TempScalar<5>>(temp_buffer);
      auto& v_dot_normal_times_one_minus_cs2 =
          get<::Tags::TempScalar<6>>(temp_buffer);

      const Scalar<DataVector> sound_speed_squared{
          clamp(get(equation_of_state
                        .sound_speed_squared_from_density_and_temperature(
                            rest_mass_density, temperature, electron_fraction)),
                0.0, 1.0)};

      // Compute v_dot_normal, v^i n_i
      dot_product(make_not_null(&v_dot_normal), spatial_velocity,
                  normal_covector);

      // Compute v^2=v^i v_i
      dot_product(make_not_null(&v_squared), spatial_velocity,
                  spatial_velocity_one_form);
      get(v_squared) = clamp(get(v_squared), 0.0, 1.0 - 1.0e-8);

      // Calculate characteristic speeds in inertial frame
      //
      // Ideally we'd use the Lorentz factor instead of 1-v^2, but I (Nils
      // Deppe) don't have the bandwidth to change this right now.
      get(one_minus_v2_cs2) = 1.0 - get(v_squared) * get(sound_speed_squared);
      get(one_minus_cs2) = 1.0 - get(sound_speed_squared);
      discriminant =
          get(sound_speed_squared) * (1.0 - get(v_squared)) *
          (get(one_minus_v2_cs2) -
           get(v_dot_normal) * get(v_dot_normal) * get(one_minus_cs2));
      discriminant = max(discriminant, 0.0);
      discriminant = sqrt(discriminant);

      get(lapse_over_one_minus_v2_cs2) = get(lapse) / get(one_minus_v2_cs2);
      get(v_dot_normal_times_one_minus_cs2) =
          get(v_dot_normal) * get(one_minus_cs2);

      for (size_t i = 0; i < num_points; ++i) {
        if (get(rest_mass_density)[i] > light_speed_density_cutoff_) {
          get(*packaged_largest_outgoing_char_speed)[i] =
              get(lapse_over_one_minus_v2_cs2)[i] *
                  (get(v_dot_normal_times_one_minus_cs2)[i] + discriminant[i]) -
              get(shift_dot_normal)[i];
          get(*packaged_largest_ingoing_char_speed)[i] =
              get(lapse_over_one_minus_v2_cs2)[i] *
                  (get(v_dot_normal_times_one_minus_cs2)[i] - discriminant[i]) -
              get(shift_dot_normal)[i];
        }
      }
    }

    // Correct for mesh velocity
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_largest_outgoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
      get(*packaged_largest_ingoing_char_speed) -=
          get(*normal_dot_mesh_velocity);
    }
  }

  // The normal vector is raised with this side's spatial metric, because the
  // FD faces provide only the covector.
  const size_t num_points = get(rest_mass_density).size();
  Variables<
      tmpl::list<::Tags::TempScalar<0>, ::Tags::TempII<0, 3, Frame::Inertial>>>
      metric_buffer{num_points};
  auto& det_spatial_metric = get<::Tags::TempScalar<0>>(metric_buffer);
  auto& inv_spatial_metric =
      get<::Tags::TempII<0, 3, Frame::Inertial>>(metric_buffer);
  determinant_and_inverse(make_not_null(&det_spatial_metric),
                          make_not_null(&inv_spatial_metric), spatial_metric);
  *packaged_interface_unit_normal = normal_covector;
  raise_or_lower_index(packaged_interface_unit_normal_vector, normal_covector,
                       inv_spatial_metric);

  // Fast-magnetosonic speeds of this side, with its own lapse, shift and
  // spatial metric. All variables but the divergence cleaning pair use them,
  // combined with the neighbor's by the same two-sided min/max in
  // dg_boundary_terms. The estimate a^2 = c_s^2 + v_A^2 (1 - c_s^2) drops the
  // dependence on B.n and is an upper bound on the fast speed. Without a
  // magnetic field they are the largest speeds, i.e. the hydro speeds.
  *packaged_fast_outgoing_char_speed = *packaged_largest_outgoing_char_speed;
  *packaged_fast_ingoing_char_speed = *packaged_largest_ingoing_char_speed;
  // A face with no point above the cutoff keeps the light speeds.
  if (has_b_field and
      max(get(rest_mass_density)) > light_speed_density_cutoff_) {
    Variables<tmpl::list<::Tags::TempScalar<0>, ::Tags::TempScalar<1>,
                         ::Tags::TempScalar<2>, ::Tags::TempScalar<3>,
                         ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
                         ::Tags::TempScalar<6>, ::Tags::TempScalar<7>>>
        temp_buffer{num_points};
    auto& v_dot_normal = get<::Tags::TempScalar<0>>(temp_buffer);
    auto& v_squared = get<::Tags::TempScalar<1>>(temp_buffer);
    auto& det_times_b_squared = get<::Tags::TempScalar<2>>(temp_buffer);
    auto& b_dot_v = get<::Tags::TempScalar<3>>(temp_buffer);
    auto& shift_dot_normal = get<::Tags::TempScalar<4>>(temp_buffer);
    auto& a_squared = get(get<::Tags::TempScalar<5>>(temp_buffer));
    auto& one_minus_v2_a2 = get(get<::Tags::TempScalar<6>>(temp_buffer));
    auto& lapse_over_one_minus_v2_a2 =
        get(get<::Tags::TempScalar<7>>(temp_buffer));

    // The inputs are clamped as in the hydro branch above, so unphysical face
    // states cannot trap.
    dot_product(make_not_null(&v_dot_normal), spatial_velocity,
                normal_covector);
    dot_product(make_not_null(&v_squared), spatial_velocity,
                spatial_velocity_one_form);
    get(v_squared) = clamp(get(v_squared), 0.0, 1.0 - 1.0e-8);
    dot_product(make_not_null(&det_times_b_squared), tilde_b, tilde_b,
                spatial_metric);
    dot_product(make_not_null(&b_dot_v), tilde_b, spatial_velocity_one_form);
    // b^2 = B^2 / W^2 + (B^i v_i)^2 with B^i = tilde_B^i / sqrt(gamma) and
    // 1 / W^2 = 1 - v^2, so gamma b^2 needs no division. The Alfven speed is
    // v_A^2 = b^2 / (b^2 + rho h). The pressure comes from the temperature:
    // from the energy, a tabulated equation of state would root-find T.
    get(det_times_b_squared) =
        get(det_times_b_squared) * (1.0 - get(v_squared)) +
        square(get(b_dot_v));
    const Scalar<DataVector> specific_internal_energy =
        equation_of_state.specific_internal_energy_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction);
    const Scalar<DataVector> pressure =
        equation_of_state.pressure_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction);
    a_squared =
        get(det_times_b_squared) /
        (get(det_times_b_squared) +
         (get(rest_mass_density) * (1.0 + get(specific_internal_energy)) +
          get(pressure)) *
             get(det_spatial_metric));
    a_squared = clamp(a_squared, 0.0, 1.0);
    const Scalar<DataVector> sound_speed_squared{clamp(
        get(equation_of_state.sound_speed_squared_from_density_and_temperature(
            rest_mass_density, temperature, electron_fraction)),
        0.0, 1.0)};
    a_squared =
        get(sound_speed_squared) + a_squared * (1.0 - get(sound_speed_squared));

    // The dispersion relation of the hydro branch with a^2 for c_s^2
    one_minus_v2_a2 = 1.0 - get(v_squared) * a_squared;
    auto& discriminant = get(b_dot_v);
    discriminant =
        a_squared * (1.0 - get(v_squared)) *
        (one_minus_v2_a2 - square(get(v_dot_normal)) * (1.0 - a_squared));
    discriminant = max(discriminant, 0.0);
    discriminant = sqrt(discriminant);
    dot_product(make_not_null(&shift_dot_normal), shift, normal_covector);
    lapse_over_one_minus_v2_a2 = get(lapse) / one_minus_v2_a2;
    get(v_dot_normal) *= 1.0 - a_squared;
    get(*packaged_fast_outgoing_char_speed) =
        lapse_over_one_minus_v2_a2 * (get(v_dot_normal) + discriminant) -
        get(shift_dot_normal);
    get(*packaged_fast_ingoing_char_speed) =
        lapse_over_one_minus_v2_a2 * (get(v_dot_normal) - discriminant) -
        get(shift_dot_normal);
    if (normal_dot_mesh_velocity.has_value()) {
      get(*packaged_fast_outgoing_char_speed) -= get(*normal_dot_mesh_velocity);
      get(*packaged_fast_ingoing_char_speed) -= get(*normal_dot_mesh_velocity);
    }
    // Keep the light speed in the atmosphere, as the largest speeds do.
    for (size_t pt = 0; pt < num_points; ++pt) {
      if (get(rest_mass_density)[pt] <= light_speed_density_cutoff_) {
        get(*packaged_fast_outgoing_char_speed)[pt] =
            get(*packaged_largest_outgoing_char_speed)[pt];
        get(*packaged_fast_ingoing_char_speed)[pt] =
            get(*packaged_largest_ingoing_char_speed)[pt];
      }
    }
  }

  *packaged_tilde_d = tilde_d;
  *packaged_tilde_ye = tilde_ye;
  *packaged_tilde_tau = tilde_tau;
  *packaged_tilde_s = tilde_s;
  *packaged_tilde_b = tilde_b;
  *packaged_tilde_phi = tilde_phi;

  normal_dot_flux(packaged_normal_dot_flux_tilde_d, normal_covector,
                  flux_tilde_d);
  normal_dot_flux(packaged_normal_dot_flux_tilde_ye, normal_covector,
                  flux_tilde_ye);
  normal_dot_flux(packaged_normal_dot_flux_tilde_tau, normal_covector,
                  flux_tilde_tau);
  normal_dot_flux(packaged_normal_dot_flux_tilde_s, normal_covector,
                  flux_tilde_s);
  normal_dot_flux(packaged_normal_dot_flux_tilde_b, normal_covector,
                  flux_tilde_b);
  normal_dot_flux(packaged_normal_dot_flux_tilde_phi, normal_covector,
                  flux_tilde_phi);

  using std::max;
  return max(max(abs(get(*packaged_largest_outgoing_char_speed))),
             max(abs(get(*packaged_largest_ingoing_char_speed))));
}

void Hll::dg_boundary_terms(
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_d,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_ye,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_s,
    const gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
        boundary_correction_tilde_b,
    const gsl::not_null<Scalar<DataVector>*> boundary_correction_tilde_phi,
    const Scalar<DataVector>& tilde_d_int,
    const Scalar<DataVector>& tilde_ye_int,
    const Scalar<DataVector>& tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_int,
    const Scalar<DataVector>& tilde_phi_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_int,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_int,
    const Scalar<DataVector>& largest_outgoing_char_speed_int,
    const Scalar<DataVector>& largest_ingoing_char_speed_int,
    const Scalar<DataVector>& fast_outgoing_char_speed_int,
    const Scalar<DataVector>& fast_ingoing_char_speed_int,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_int,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        interface_unit_normal_vector_int,
    const Scalar<DataVector>& tilde_d_ext,
    const Scalar<DataVector>& tilde_ye_ext,
    const Scalar<DataVector>& tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b_ext,
    const Scalar<DataVector>& tilde_phi_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_d_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_ye_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_tau_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_s_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>& normal_dot_flux_tilde_b_ext,
    const Scalar<DataVector>& normal_dot_flux_tilde_phi_ext,
    const Scalar<DataVector>& largest_outgoing_char_speed_ext,
    const Scalar<DataVector>& largest_ingoing_char_speed_ext,
    const Scalar<DataVector>& fast_outgoing_char_speed_ext,
    const Scalar<DataVector>& fast_ingoing_char_speed_ext,
    const tnsr::i<DataVector, 3, Frame::Inertial>& interface_unit_normal_ext,
    const tnsr::I<DataVector, 3, Frame::Inertial>&
        interface_unit_normal_vector_ext,
    const dg::Formulation dg_formulation) {
  const size_t num_points = get(tilde_d_int).size();
  const bool weak = dg_formulation == dg::Formulation::WeakInertial;

  Variables<tmpl::list<
      ::Tags::TempScalar<0>, ::Tags::TempScalar<1>, ::Tags::TempScalar<2>,
      ::Tags::TempScalar<3>, ::Tags::TempScalar<4>, ::Tags::TempScalar<5>,
      ::Tags::TempScalar<6>, ::Tags::TempScalar<7>, ::Tags::TempScalar<8>,
      ::Tags::TempScalar<9>, ::Tags::TempScalar<10>, ::Tags::TempScalar<11>,
      ::Tags::TempScalar<12>, ::Tags::TempScalar<13>,
      ::Tags::Tempi<0, 3, Frame::Inertial>,
      ::Tags::TempI<0, 3, Frame::Inertial>>>
      temp_buffer{num_points};
  auto& lambda_max = get(get<::Tags::TempScalar<0>>(temp_buffer));
  auto& lambda_min = get(get<::Tags::TempScalar<1>>(temp_buffer));
  auto& lambda_lprod = get(get<::Tags::TempScalar<2>>(temp_buffer));
  auto& one_over_lambda_dl = get(get<::Tags::TempScalar<3>>(temp_buffer));
  auto& fast_max = get(get<::Tags::TempScalar<4>>(temp_buffer));
  auto& fast_min = get(get<::Tags::TempScalar<5>>(temp_buffer));
  auto& fast_lprod = get(get<::Tags::TempScalar<6>>(temp_buffer));
  auto& one_over_fast_dl = get(get<::Tags::TempScalar<7>>(temp_buffer));

  // Light-speed bounds for the divergence cleaning pair (Phi and the normal
  // magnetic field).
  lambda_max = max(0., get(largest_outgoing_char_speed_int),
                   -get(largest_ingoing_char_speed_ext));
  lambda_min = min(0., get(largest_ingoing_char_speed_int),
                   -get(largest_outgoing_char_speed_ext));
  // Fast-magnetosonic bounds for the other variables, from the two sides' own
  // speeds. The exterior packaged its speeds against its own outward normal,
  // hence the signs. In the atmosphere these are the light speeds.
  fast_max = max(0., get(fast_outgoing_char_speed_int),
                 -get(fast_ingoing_char_speed_ext));
  fast_min = min(0., get(fast_ingoing_char_speed_int),
                 -get(fast_outgoing_char_speed_ext));
  lambda_lprod = lambda_max * lambda_min;
  one_over_lambda_dl = 1.0 / max(lambda_max - lambda_min, 1.0e-30);
  fast_lprod = fast_max * fast_min;
  one_over_fast_dl = 1.0 / max(fast_max - fast_min, 1.0e-30);

  // The HLL flux of one component for the bounds l_max, l_min, with
  // lprod = l_max l_min.
  const auto hll = [&weak](const gsl::not_null<DataVector*> result,
                           const DataVector& l_max, const DataVector& l_min,
                           const DataVector& lprod,
                           const DataVector& one_over_dl,
                           const DataVector& u_int, const DataVector& nf_int,
                           const DataVector& u_ext, const DataVector& nf_ext) {
    if (weak) {
      *result = (l_max * nf_int + l_min * nf_ext + lprod * (u_ext - u_int)) *
                one_over_dl;
    } else {
      *result =
          (l_min * (nf_int + nf_ext) + lprod * (u_ext - u_int)) * one_over_dl;
    }
  };

  hll(make_not_null(&get(*boundary_correction_tilde_d)), fast_max, fast_min,
      fast_lprod, one_over_fast_dl, get(tilde_d_int),
      get(normal_dot_flux_tilde_d_int), get(tilde_d_ext),
      get(normal_dot_flux_tilde_d_ext));
  hll(make_not_null(&get(*boundary_correction_tilde_ye)), fast_max, fast_min,
      fast_lprod, one_over_fast_dl, get(tilde_ye_int),
      get(normal_dot_flux_tilde_ye_int), get(tilde_ye_ext),
      get(normal_dot_flux_tilde_ye_ext));
  hll(make_not_null(&get(*boundary_correction_tilde_tau)), fast_max, fast_min,
      fast_lprod, one_over_fast_dl, get(tilde_tau_int),
      get(normal_dot_flux_tilde_tau_int), get(tilde_tau_ext),
      get(normal_dot_flux_tilde_tau_ext));
  hll(make_not_null(&get(*boundary_correction_tilde_phi)), lambda_max,
      lambda_min, lambda_lprod, one_over_lambda_dl, get(tilde_phi_int),
      get(normal_dot_flux_tilde_phi_int), get(tilde_phi_ext),
      get(normal_dot_flux_tilde_phi_ext));
  for (size_t i = 0; i < 3; ++i) {
    hll(make_not_null(&boundary_correction_tilde_s->get(i)), fast_max, fast_min,
        fast_lprod, one_over_fast_dl, tilde_s_int.get(i),
        normal_dot_flux_tilde_s_int.get(i), tilde_s_ext.get(i),
        normal_dot_flux_tilde_s_ext.get(i));
  }
  // Without a magnetic field on either side, or in the atmosphere, the two
  // pairs of bounds are equal and the split below would recombine into the
  // plain HLL flux of the other variables.
  if (std::equal(fast_max.begin(), fast_max.end(), lambda_max.begin()) and
      std::equal(fast_min.begin(), fast_min.end(), lambda_min.begin())) {
    for (size_t i = 0; i < 3; ++i) {
      hll(make_not_null(&boundary_correction_tilde_b->get(i)), fast_max,
          fast_min, fast_lprod, one_over_fast_dl, tilde_b_int.get(i),
          normal_dot_flux_tilde_b_int.get(i), tilde_b_ext.get(i),
          normal_dot_flux_tilde_b_ext.get(i));
    }
    return;
  }
  // The normal part of the magnetic field uses the light-speed bounds, the
  // tangential part the fast bounds: G(B^i) = G(B_n) n^i + G(B_t^i). The two
  // sides' normals are opposite up to interpolation, so the split is along
  // their average m, with n_i n^j = m_i m^j / (m_k m^k), which makes the
  // correction change sign exactly under the int<->ext swap. Since G is
  // linear, only m^j needs the division.
  auto& n_lower = get<::Tags::Tempi<0, 3, Frame::Inertial>>(temp_buffer);
  auto& n_upper = get<::Tags::TempI<0, 3, Frame::Inertial>>(temp_buffer);
  for (size_t i = 0; i < 3; ++i) {
    n_lower.get(i) = 0.5 * (interface_unit_normal_int.get(i) -
                            interface_unit_normal_ext.get(i));
    n_upper.get(i) = 0.5 * (interface_unit_normal_vector_int.get(i) -
                            interface_unit_normal_vector_ext.get(i));
  }
  auto& one_over_n_squared = get(get<::Tags::TempScalar<8>>(temp_buffer));
  one_over_n_squared =
      1.0 / (n_lower.get(0) * n_upper.get(0) + n_lower.get(1) * n_upper.get(1) +
             n_lower.get(2) * n_upper.get(2));
  for (size_t i = 0; i < 3; ++i) {
    n_upper.get(i) *= one_over_n_squared;
  }
  auto& bn_int = get(get<::Tags::TempScalar<9>>(temp_buffer));
  auto& bn_ext = get(get<::Tags::TempScalar<10>>(temp_buffer));
  auto& nfbn_int = get(get<::Tags::TempScalar<11>>(temp_buffer));
  auto& nfbn_ext = get(get<::Tags::TempScalar<12>>(temp_buffer));
  auto& g_bn = get(get<::Tags::TempScalar<13>>(temp_buffer));
  bn_int = get<0>(tilde_b_int) * get<0>(n_lower) +
           get<1>(tilde_b_int) * get<1>(n_lower) +
           get<2>(tilde_b_int) * get<2>(n_lower);
  bn_ext = get<0>(tilde_b_ext) * get<0>(n_lower) +
           get<1>(tilde_b_ext) * get<1>(n_lower) +
           get<2>(tilde_b_ext) * get<2>(n_lower);
  nfbn_int = get<0>(normal_dot_flux_tilde_b_int) * get<0>(n_lower) +
             get<1>(normal_dot_flux_tilde_b_int) * get<1>(n_lower) +
             get<2>(normal_dot_flux_tilde_b_int) * get<2>(n_lower);
  nfbn_ext = get<0>(normal_dot_flux_tilde_b_ext) * get<0>(n_lower) +
             get<1>(normal_dot_flux_tilde_b_ext) * get<1>(n_lower) +
             get<2>(normal_dot_flux_tilde_b_ext) * get<2>(n_lower);
  hll(make_not_null(&g_bn), lambda_max, lambda_min, lambda_lprod,
      one_over_lambda_dl, bn_int, nfbn_int, bn_ext, nfbn_ext);
  // The tangential part, B_t^i = B^i - B_n n^i on each side, in one
  // expression per component.
  for (size_t i = 0; i < 3; ++i) {
    const DataVector& n_i = n_upper.get(i);
    if (weak) {
      boundary_correction_tilde_b->get(i) =
          (fast_max * (normal_dot_flux_tilde_b_int.get(i) - nfbn_int * n_i) +
           fast_min * (normal_dot_flux_tilde_b_ext.get(i) - nfbn_ext * n_i) +
           fast_lprod * ((tilde_b_ext.get(i) - bn_ext * n_i) -
                         (tilde_b_int.get(i) - bn_int * n_i))) *
              one_over_fast_dl +
          g_bn * n_i;
    } else {
      boundary_correction_tilde_b->get(i) =
          (fast_min * ((normal_dot_flux_tilde_b_int.get(i) - nfbn_int * n_i) +
                       (normal_dot_flux_tilde_b_ext.get(i) - nfbn_ext * n_i)) +
           fast_lprod * ((tilde_b_ext.get(i) - bn_ext * n_i) -
                         (tilde_b_int.get(i) - bn_int * n_i))) *
              one_over_fast_dl +
          g_bn * n_i;
    }
  }
}

bool operator==(const Hll& lhs, const Hll& rhs) {
  return lhs.magnetic_field_magnitude_for_hydro_ ==
             rhs.magnetic_field_magnitude_for_hydro_ and
         lhs.light_speed_density_cutoff_ == rhs.light_speed_density_cutoff_;
}
bool operator!=(const Hll& lhs, const Hll& rhs) { return not(lhs == rhs); }

// NOLINTNEXTLINE
PUP::able::PUP_ID Hll::my_PUP_ID = 0;
}  // namespace grmhd::ValenciaDivClean::BoundaryCorrections
