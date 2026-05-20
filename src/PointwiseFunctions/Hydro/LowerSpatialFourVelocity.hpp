// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/Hydro/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"

namespace hydro::Tags {

/*!
 * \brief Computes $u_i=W \gamma_{ij} v^j$, where $W$ is the Lorentz factor,
 * $\gamma_{ij}$ is the spatial metric, and $v^j$ is the spatial velocity.
 *
 */
struct LowerSpatialFourVelocityCompute
    : LowerSpatialFourVelocity<DataVector, 3, Frame::Inertial>,
      db::ComputeTag {
  using base = LowerSpatialFourVelocity<DataVector, 3, Frame::Inertial>;
  using argument_tags =
      tmpl::list<SpatialVelocity<DataVector, 3, Frame::Inertial>,
                 gr::Tags::SpatialMetric<DataVector, 3>,
                 LorentzFactor<DataVector>>;
  static void function(const gsl::not_null<tnsr::i<DataVector, 3>*> result,
                       const tnsr::I<DataVector, 3>& spatial_velocity,
                       const tnsr::ii<DataVector, 3>& spatial_metric,
                       const Scalar<DataVector>& lorentz_factor);
};

}  // namespace hydro::Tags
