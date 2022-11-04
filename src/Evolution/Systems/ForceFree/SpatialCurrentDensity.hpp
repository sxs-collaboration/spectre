// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

namespace ForceFree {

/*!
 * \brief Specifies electric current density (Ohm's law) that imposes the
 * force-free condition.
 *
 * We use the relaxation approach by \cite Alic2012 :
 *
 * \f{align}{
 *  J^i = q \frac{\epsilon^{ijk}_{(3)}E_jB_k}{B_lB^l}
 *      + \eta \left[
 *          \frac{E_jB^j}{B_lB^l}B^i
 *          + \frac{\mathcal{R}(E_lE^l-B_lB^l)}{B_lB^l}E^i
 *      \right]
 * \f}
 *
 * where \f$q\f$ is electric charge density, \f$E^i\f$ is electric field,
 * \f$B^i\f$ is magnetic field, \f$\eta\f$ is parallel conductivity.
 *
 * \f$\epsilon_{(3)}^{ijk}\f$ is the spatial Levi-Civita tensor defined as
 *
 * \f{align*}
 *  \epsilon_{(3)}^{ijk} = \frac{1}{\sqrt{\gamma}} [ijk]
 * \f}
 *
 * where \f$\gamma\f$ is the determinant of spatial metric and \f$[ijk]\f$ is
 * the antisymmetric symbol with \f$[123]=+1\f$.
 *
 * \f$\mathcal{R}(x)\f$ is the ramp (or rectifier) function
 *
 * \f{align*}
 *  \mathcal{R}(x) = \left\{\begin{array}{lc}
 *          x, & \text{if } x \geq 0 \\
 *          0, & \text{if } x < 0 \\
 * \end{array}\right\} = \max (x, 0)
 * \f}
 *
 */
struct SpatialCurrentDensity {
  using return_tags = tmpl::list<Tags::SpatialCurrentDensity>;

  using argument_tags =
      tmpl::list<Tags::TildeQ, Tags::TildeE, Tags::TildeB,
                 Tags::ParallelConductivity, gr::Tags::SqrtDetSpatialMetric<>,
                 gr::Tags::SpatialMetric<3>>;

  static void apply(
      gsl::not_null<tnsr::I<DataVector, 3, Frame::Inertial>*>
          spatial_current_density,
      const Scalar<DataVector>& tilde_q,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_e,
      const tnsr::I<DataVector, 3, Frame::Inertial>& tilde_b,
      const double parallel_conductivity,
      const Scalar<DataVector>& sqrt_det_spatial_metric,
      const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric);
};

}  // namespace ForceFree
