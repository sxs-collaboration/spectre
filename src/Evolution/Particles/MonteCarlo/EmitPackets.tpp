// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>

#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/TemplatedLocalFunctions.hpp"
#include "PointwiseFunctions/Hydro/Units.hpp"

using hydro::units::nuclear::proton_mass;

namespace Particles::MonteCarlo {

template <size_t EnergyBins, size_t NeutrinoSpecies>
void TemplatedLocalFunctions<EnergyBins, NeutrinoSpecies>::emit_packets(
    const gsl::not_null<std::vector<Packet>*> packets,
    const gsl::not_null<std::mt19937*> random_number_generator,
    const gsl::not_null<Scalar<DataVector>*> coupling_tilde_tau,
    const gsl::not_null<tnsr::i<DataVector, 3, Frame::Inertial>*>
        coupling_tilde_s,
    const gsl::not_null<Scalar<DataVector>*> coupling_rho_ye,
    const double& time_start_step, const double& time_step, const Mesh<3>& mesh,
    const std::array<std::array<DataVector, EnergyBins>, NeutrinoSpecies>&
        emission_in_cell,
    const std::array<DataVector, NeutrinoSpecies>& single_packet_energy,
    const std::array<double, EnergyBins>& energy_at_bin_center,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Jacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_jacobian,
    const InverseJacobian<DataVector, 4, Frame::Inertial, Frame::Fluid>&
        inertial_to_fluid_inverse_jacobian) {
  std::uniform_real_distribution<double> rng_uniform_zero_to_one(0.0, 1.0);
  // We begin by determining how many packets to create for each cell, neutrino
  // species, and energy group.
  const size_t grid_size = mesh.number_of_grid_points();
  std::array<std::array<std::vector<size_t>, EnergyBins>, NeutrinoSpecies>
      number_of_packets_to_create_per_cell;
  size_t number_of_packets_to_create_total = 0;
  for (size_t s = 0; s < NeutrinoSpecies; s++) {
    for (size_t g = 0; g < EnergyBins; g++) {
      gsl::at(gsl::at(number_of_packets_to_create_per_cell, s), g)
          .resize(grid_size);
      for (size_t i = 0; i < grid_size; i++) {
        const double& emission_this_cell =
            gsl::at(gsl::at(emission_in_cell, s), g)[i];
        const double packets_to_create_double =
            emission_this_cell / gsl::at(single_packet_energy, s)[i];
        size_t packets_to_create_int = floor(packets_to_create_double);
        if (rng_uniform_zero_to_one(*random_number_generator) <
            packets_to_create_double -
                static_cast<double>(packets_to_create_int)) {
          packets_to_create_int++;
        }
        gsl::at(gsl::at(number_of_packets_to_create_per_cell, s), g)[i] =
            packets_to_create_int;
        number_of_packets_to_create_total += packets_to_create_int;

        // Coupling to hydro variables. These will need to be divided by the
        // appropriate coordinate volume when the coupling is performed, as here
        // we consider total emission numbers, rather than emission densities.
        coupling_tilde_tau->get()[i] -=
            emission_this_cell * get(lorentz_factor)[i];
        for (int d = 0; d < 3; d++) {
          coupling_tilde_s->get(d)[i] -=
              emission_this_cell * lower_spatial_four_velocity.get(d)[i];
        }
        if (NeutrinoSpecies >= 2 && s < 2) {
          coupling_rho_ye->get()[i] -=
              (s == 0 ? 1.0 : -1.0) * emission_this_cell /
              gsl::at(energy_at_bin_center, g) * proton_mass;
        }
      }
    }
  }

  // With the number of packets known, resize the vector holding the packets
  Packet default_packet(0, 0.0, 0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0);
  const size_t initial_size = packets->size();
  packets->resize(initial_size + number_of_packets_to_create_total,
                  default_packet);

  // Allocate memory for some temporary data.
  double time_normalized = 0.0;
  std::array<double, 3> coord_normalized{{0.0, 0.0, 0.0}};
  std::array<double, 3> three_momentum_normalized{{0.0, 0.0, 0.0}};
  std::array<double, 4> four_momentum_fluid_frame{{0.0, 0.0, 0.0, 0.0}};
  const Index<3> extents = mesh.extents();
  const std::array<double, 3> logical_dx{2.0 / static_cast<double>(extents[0]),
                                         2.0 / static_cast<double>(extents[1]),
                                         2.0 / static_cast<double>(extents[2])};
  std::array<double, 3> coord_bottom_cell{{0.0, 0.0, 0.0}};

  // Now create the packets
  size_t next_packet_index = initial_size;
  for (size_t x_idx = 0; x_idx < extents[0]; x_idx++) {
    gsl::at(coord_bottom_cell, 0) =
        -1.0 + 2.0 * static_cast<double>(x_idx) * gsl::at(logical_dx, 0);
    for (size_t y_idx = 0; y_idx < extents[1]; y_idx++) {
      gsl::at(coord_bottom_cell, 1) =
          -1.0 + 2.0 * static_cast<double>(y_idx) * gsl::at(logical_dx, 1);
      for (size_t z_idx = 0; z_idx < extents[2]; z_idx++) {
        const size_t idx = mesh.storage_index(Index<3>{{x_idx, y_idx, z_idx}});
        gsl::at(coord_bottom_cell, 2) =
            -1.0 + 2.0 * static_cast<double>(z_idx) * gsl::at(logical_dx, 2);
        for (size_t s = 0; s < NeutrinoSpecies; s++) {
          for (size_t g = 0; g < EnergyBins; g++) {
            for (size_t p = 0;
                 p < gsl::at(gsl::at(number_of_packets_to_create_per_cell, s),
                             g)[idx];
                 p++) {
              detail::draw_single_packet(&time_normalized, &coord_normalized,
                                         &three_momentum_normalized,
                                         random_number_generator);
              Packet& current_packet = (*packets)[next_packet_index];
              current_packet.species = s;
              current_packet.time =
                  time_start_step + time_normalized * time_step;
              current_packet.number_of_neutrinos =
                  gsl::at(gsl::at(single_packet_energy, s), idx) /
                  gsl::at(energy_at_bin_center, g);
              current_packet.index_of_closest_grid_point = idx;
              // 4_momentum_fluid_frame contains p^mu in an orthornormal
              // coordinate system moving with the fluid.
              gsl::at(four_momentum_fluid_frame, 0) =
                  gsl::at(energy_at_bin_center, g);
              for (size_t d = 0; d < 3; d++) {
                current_packet.coordinates.get(d) =
                    gsl::at(coord_bottom_cell, d) +
                    gsl::at(coord_normalized, d) * gsl::at(logical_dx, d);
                gsl::at(four_momentum_fluid_frame, d + 1) =
                    gsl::at(three_momentum_normalized, d) *
                    gsl::at(four_momentum_fluid_frame, 0);
              }

              // The packet stores p^t and p_i in the inertial frame. We
              // transform the momentum accordingly.
              current_packet.momentum_upper_t = 0.0;
              for (size_t d = 0; d < 4; d++) {
                current_packet.momentum_upper_t +=
                    inertial_to_fluid_inverse_jacobian.get(0, d)[idx] *
                    gsl::at(four_momentum_fluid_frame, d);
              }
              for (size_t d = 0; d < 3; d++) {
                // Multiply by -1.0 because p_t = -p^t in an orthonormal frame
                current_packet.momentum.get(d) =
                    (-1.0) * gsl::at(four_momentum_fluid_frame, 0) *
                    inertial_to_fluid_jacobian.get(d + 1, 0)[idx];
                for (size_t dd = 0; dd < 3; dd++) {
                  current_packet.momentum.get(d) +=
                      gsl::at(four_momentum_fluid_frame, dd + 1) *
                      inertial_to_fluid_jacobian.get(d + 1, dd + 1)[idx];
                }
              }
              next_packet_index++;
            }
          }
        }
      }
    }
  }
}

}  // namespace Particles::MonteCarlo
