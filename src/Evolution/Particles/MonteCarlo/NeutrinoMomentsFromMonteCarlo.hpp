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

void inertial_frame_energy_density(
    gsl::not_null<Scalar<DataVector>*> fluid_frame_energy_density,
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
      gsl::not_null<return_type*> inertial_frame_density,
      const std::vector<Packet>& packets, const Scalar<DataVector>& lapse,
      const Scalar<DataVector>& sqrt_determinant_spatial_metric,
      const Mesh<3>& mesh,
      const Scalar<DataVector>& det_inv_jacobian_logical_to_inertial);
};

}  // namespace Particles::MonteCarlo
