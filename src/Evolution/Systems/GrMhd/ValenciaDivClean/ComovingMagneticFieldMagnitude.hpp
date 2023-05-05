// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace grmhd::ValenciaDivClean::Tags {
/*!
 * \brief Compute the magnitude of the comoving magnetic field.
 *
 * \f{align}
 *  \sqrt{b^2} = \left( \frac{B^2}{W^2} + (B^iv_i)^2 \right)^{1/2}
 * \f}
 *
 * \note This ComputeTag is for observation and monitoring purpose, not related
 * to the actual time evolution.
 *
 */
struct ComovingMagneticFieldMagnitudeCompute
    : hydro::Tags::ComovingMagneticFieldMagnitude<DataVector>,
      db::ComputeTag {
  using argument_tags = tmpl::list<hydro::Tags::MagneticField<DataVector, 3>,
                                   hydro::Tags::SpatialVelocity<DataVector, 3>,
                                   hydro::Tags::LorentzFactor<DataVector>,
                                   gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = Scalar<DataVector>;
  using base = hydro::Tags::ComovingMagneticFieldMagnitude<DataVector>;

  static void function(
      gsl::not_null<Scalar<DataVector>*> comoving_magnetic_field_magnitude,
      const tnsr::I<DataVector, 3>& magnetic_field,
      const tnsr::I<DataVector, 3>& spatial_velocity,
      const Scalar<DataVector>& lorentz_factor,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);
};
}  // namespace grmhd::ValenciaDivClean::Tags
