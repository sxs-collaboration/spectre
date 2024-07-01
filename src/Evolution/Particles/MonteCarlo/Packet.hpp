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

  /// Default constructor needed to make Packet puppable
  /// For the same reason, we have default initialization
  /// for a number of member variables to unphysical values.
  Packet()
      : species{0},
        number_of_neutrinos{0},
        index_of_closest_grid_point{0},
        time{-1.0},
        momentum_upper_t{0.0} {
    coordinates[0] = 0.0;
    coordinates[1] = 0.0;
    coordinates[2] = 0.0;
    momentum[0] = 0.0;
    momentum[1] = 0.0;
    momentum[2] = 0.0;
  }

  /// Species of neutrinos (in the code, just an index used to access the
  /// right interaction rates; typically \f$0=\nu_e, 1=\nu_a, 2=\nu_x\f$)
  size_t species = std::numeric_limits<size_t>::max();

  /// Number of neutrinos represented by current packet
  /// Note that this number is rescaled so that
  /// `Energy_of_packet = N * Energy_of_neutrinos`
  /// with the packet energy in G=Msun=c=1 units but
  /// the neutrino energy in MeV!
  double number_of_neutrinos = std::numeric_limits<double>::signaling_NaN();

  /// Index of the closest point on the FD grid.
  size_t index_of_closest_grid_point = std::numeric_limits<size_t>::max();

  /// Current time
  double time = std::numeric_limits<double>::signaling_NaN();

  /// Stores \f$p^t\f$
  double momentum_upper_t = std::numeric_limits<double>::signaling_NaN();

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

  void pup(PUP::er& p);

  /// Overloaded comparison operator. Useful for test; in an actual simulation
  /// two distinct packets are not truly "identical" as they represent different
  /// particles.
  bool operator==(const Packet& rhs) const {
    return (this->species == rhs.species) and
           (this->number_of_neutrinos == rhs.number_of_neutrinos) and
           (this->index_of_closest_grid_point ==
            rhs.index_of_closest_grid_point) and
           (this->time == rhs.time) and
           (this->momentum_upper_t == rhs.momentum_upper_t) and
           (this->coordinates == rhs.coordinates) and
           (this->momentum == rhs.momentum);
  };
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
