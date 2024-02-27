// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines tags related to domain quantities

#pragma once

#include "Domain/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

namespace Frame {
struct Fluid;
struct Inertial;
}  // namespace Frame

namespace Particles::MonteCarlo {

// Inverse Jacobian of the map from inertial coordinate to an orthonormal frame
// comoving with the fluid. That frame uses the 4-velocity as its time axis, and
// constructs the other members of the tetrads using Gram-Schmidt's algorithm.
struct InverseJacobianInertialToFluidCompute
    : domain::Tags::InverseJacobian<4, Frame::Inertial, Frame::Fluid>,
      db::ComputeTag {
  using base = domain::Tags::InverseJacobian<4, typename Frame::Inertial,
                                             typename Frame::Fluid>;
  using return_type = typename base::type;
  using argument_tags =
      tmpl::list<hydro::Tags::SpatialVelocity<DataVector, 3, Frame::Inertial>,
                 hydro::Tags::LorentzFactor<DataVector>,
                 gr::Tags::Lapse<DataVector>,
                 gr::Tags::Shift<DataVector, 3, Frame::Inertial>,
                 gr::Tags::SpatialMetric<DataVector, 3> >;

  static void function(gsl::not_null<return_type*> inv_jacobian,
                       const tnsr::I<DataVector, 3>& spatial_velocity,
                       const Scalar<DataVector>& lorentz_factor,
                       const Scalar<DataVector>& lapse,
                       const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
                       const tnsr::ii<DataVector, 3>& spatial_metric);
};

}  // namespace Particles::MonteCarlo
