// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/Gsl.hpp"

/// Items related to the evolution of particles
/// Items related to Monte-Carlo radiation transport
namespace Particles::MonteCarlo {

/// Struct representing a single Monte Carlo packet of neutrinos
struct Packet {
  /// Constructor
  Packet(const size_t& species_,
         const double& number_of_neutrinos_,
         const size_t& index_of_closest_grid_point_, const double& time_,
         const double& coord_x_, const double& coord_y_, const double& coord_z_,
         const double& p_upper_t_, const double& p_x_, const double& p_y_,
         const double& p_z_)
      : species(species_),
        number_of_neutrinos(number_of_neutrinos_),
        index_of_closest_grid_point(index_of_closest_grid_point_),
        time(time_),
        momentum_upper_t(p_upper_t_) {
    coordinates[0] = coord_x_;
    coordinates[1] = coord_y_;
    coordinates[2] = coord_z_;
    momentum[0] = p_x_;
    momentum[1] = p_y_;
    momentum[2] = p_z_;
  }

  /// Species of neutrinos (in the code, just an index used to access the
  /// right interaction rates; typically \f$0=\nu_e, 1=\nu_a, 2=\nu_x\f$)
  size_t species;

  /// Number of neutrinos represented by current packet
  /// Note that this number is rescaled so that
  /// `Energy_of_packet = N * Energy_of_neutrinos`
  /// with the packet energy in G=Msun=c=1 units but
  /// the neutrino energy in MeV!
  double number_of_neutrinos;

  /// Index of the closest point on the FD grid.
  size_t index_of_closest_grid_point;

  /// Current time
  double time;

  /// Stores \f$p^t\f$
  double momentum_upper_t;

  /// Coordinates of the packet, in element logical coordinates
  tnsr::I<double, 3, Frame::ElementLogical> coordinates;

  /// Spatial components of the 4-momentum \f$p_i\f$, in Inertial coordinates
  tnsr::i<double, 3, Frame::Inertial> momentum;

  /*!
   * Recalculte \f$p^t\f$ using the fact that the 4-momentum is a null vector
   * \f{align}{
   * p^t = \sqrt{\gamma^{ij} p_i p_j}/\alpha
   * \f}
   */
  void renormalize_momentum(
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& lapse);
};

/*!
 * Calculate energy of neutrinos in a frame comoving with the fluid
 *
 * \f{align}{
 * E = W \alpha p^t - \gamma^{ij} u_i p_j
 * \f}
 */
double compute_fluid_frame_energy(
    const Packet& packet, const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric);

}  // namespace Particles::MonteCarlo
