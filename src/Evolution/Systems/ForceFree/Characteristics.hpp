// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
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
 *  \lambda_\pm = - n_i \beta^i \pm \alpha
 * \f}
 *
 * where \f$\alpha\f$ is the lapse, \f$\beta^i\f$ is the shift, and \f$n_i\f$ is
 * the spatial unit normal.
 */
struct LargestCharacteristicSpeedCompute : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using argument_tags =
      tmpl::list<gr::Tags::Lapse<>, gr::Tags::Shift<3>,
                 ::Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<3>>>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(gsl::not_null<double*> speed,
                       const Scalar<DataVector>& lapse,
                       const tnsr::I<DataVector, 3>& shift,
                       const tnsr::i<DataVector, 3>& unit_normal);
};

}  // namespace Tags
}  // namespace ForceFree
