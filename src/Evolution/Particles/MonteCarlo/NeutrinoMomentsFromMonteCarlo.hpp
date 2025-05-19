// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DgSubcell/Tags/Jacobians.hpp"
#include "Evolution/DgSubcell/Tags/Mesh.hpp"
#include "Evolution/Particles/MonteCarlo/Packet.hpp"
#include "Evolution/Particles/MonteCarlo/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl

class DataVector;

template <size_t Dim>
class Mesh;
/// \endcond

namespace Particles::MonteCarlo {

void inertial_frame_energy_density_function(
    gsl::not_null<Scalar<DataVector>*> inertial_frame_energy_density,
    const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial);

namespace Tags {
/// Simple tag containing the inertial frame energy
/// density on the grid for Monte Carlo packets
struct InertialFrameEnergyDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags

struct InertialFrameEnergyDensityCompute : Tags::InertialFrameEnergyDensity,
                                           db::ComputeTag {
  using base = Tags::InertialFrameEnergyDensity;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Particles::MonteCarlo::Tags::PacketsOnElement,
      gr::Tags::Lapse<DataVector>, gr::Tags::SqrtDetSpatialMetric<DataVector>,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial>;

  static void function(
      gsl::not_null<return_type*> inertial_frame_energy_density,
      const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
      const Scalar<DataVector>& sqrt_determinant_spatial_metric,
      const Mesh<3>& mesh,
      const Scalar<DataVector>& det_inv_jacobian_logical_to_inertial);
};

template <size_t Nspecies>
void inertial_frame_energy_density_per_species_function(
    gsl::not_null<tnsr::i<DataVector, Nspecies, Frame::Inertial>*>
        inertial_frame_energy_density,
    const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial);

namespace Tags {
/// Simple tag containing the inertial frame energy
/// density on the grid for Monte Carlo packets
/// Each component correspond to a different neutrino species.
template <size_t Nspecies>
struct InertialFrameEnergyDensityPerSpecies : db::SimpleTag {
  using type = tnsr::i<DataVector, Nspecies, Frame::Inertial>;
};
}  // namespace Tags

template <size_t Nspecies>
struct InertialFrameEnergyDensityPerSpeciesCompute
    : Tags::InertialFrameEnergyDensityPerSpecies<Nspecies>,
      db::ComputeTag {
  using base = Tags::InertialFrameEnergyDensityPerSpecies<Nspecies>;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Particles::MonteCarlo::Tags::PacketsOnElement,
      gr::Tags::Lapse<DataVector>, gr::Tags::SqrtDetSpatialMetric<DataVector>,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial>;

  static void function(
      gsl::not_null<return_type*> inertial_frame_energy_density,
      const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
      const Scalar<DataVector>& sqrt_determinant_spatial_metric,
      const Mesh<3>& mesh,
      const Scalar<DataVector>& det_inv_jacobian_logical_to_inertial);
};

void inertial_frame_lepton_number_density_function(
    gsl::not_null<Scalar<DataVector>*> inertial_frame_lepton_number_density,
    const std::vector<Packet>& packets,
    const Scalar<DataVector>& lorentz_factor,
    const tnsr::i<DataVector, 3, Frame::Inertial>& lower_spatial_four_velocity,
    const Scalar<DataVector>& lapse,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
    const Scalar<DataVector>& sqrt_determinant_spatial_metric,
    const Mesh<3>& mesh,
    const Scalar<DataVector>& det_jacobian_logical_to_inertial);

namespace Tags {
/// Simple tag containing the net electron number density
/// in neutrinos (i.e. number of electron neutrinos minus
/// number of electron antineutrinos)
struct InertialFrameLeptonNumberDensity : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace Tags

struct InertialFrameLeptonNumberDensityCompute
    : Tags::InertialFrameLeptonNumberDensity,
      db::ComputeTag {
  using base = Tags::InertialFrameLeptonNumberDensity;
  using return_type = typename base::type;
  using argument_tags = tmpl::list<
      Particles::MonteCarlo::Tags::PacketsOnElement,
      hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
      gr::Tags::Lapse<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      evolution::dg::subcell::Tags::Mesh<3>,
      evolution::dg::subcell::fd::Tags::DetInverseJacobianLogicalToInertial>;

  static void function(
      gsl::not_null<return_type*> inertial_frame_lepton_number_density,
      const std::vector<Packet>& packets,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::I<DataVector, 3, Frame::Inertial>& spatial_velocity,
      const Scalar<DataVector>& lapse,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
      const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric,
      const Scalar<DataVector>& sqrt_determinant_spatial_metric,
      const Mesh<3>& mesh,
      const Scalar<DataVector>& det_inv_jacobian_logical_to_inertial);
};

}  // namespace Particles::MonteCarlo
