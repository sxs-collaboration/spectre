// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Evolution/Systems/ForceFree/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ForceFree {
namespace Tags {
/*!
 * \brief Compute the largest characteristic speed of the GRFFE system with
 * divergence cleaning.
 *
 * Wave speeds of the fast modes of the GRFFE system are the speed of light.
 *
 * \f{align*}
 *  \lambda_\pm = \beta^i\beta_i \pm \alpha
 * \f}
 *
 * where \f$\alpha\f$ is the lapse and \f$\beta^i\f$ is the shift. Therefore the
 * largest characteristic speed is \f$\lambda_\text{max} =
 * \sqrt{\beta_i\beta^i}+\alpha\f$.
 */
struct LargestCharacteristicSpeedCompute : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<DataVector>, gr::Tags::Shift<DataVector, 3>,
                 gr::Tags::SpatialMetric<DataVector, 3>>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(gsl::not_null<double*> speed,
                       const Scalar<DataVector>& lapse,
                       const tnsr::I<DataVector, 3>& shift,
                       const tnsr::ii<DataVector, 3>& spatial_metric);
};

}  // namespace Tags
}  // namespace ForceFree
