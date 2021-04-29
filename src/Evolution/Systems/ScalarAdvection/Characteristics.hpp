// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataVector.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace ScalarAdvection {
namespace Tags {
/*!
 * \brief Compute the largest characteristic speed of the ScalarAdvection system
 *
 * \f{align*}
 * \lambda_\text{maxabs} = \sqrt{v^iv_i}
 * \f}
 *
 * where \f$v^i\f$ is the velocity field.
 */
template <size_t Dim>
struct LargestCharacteristicSpeedCompute : LargestCharacteristicSpeed,
                                           db::ComputeTag {
  using argument_tags = tmpl::list<Tags::VelocityField<Dim>>;
  using return_type = double;
  using base = LargestCharacteristicSpeed;
  static void function(const gsl::not_null<double*> speed,
                       const tnsr::I<DataVector, Dim>& velocity_field) noexcept;
};
}  // namespace Tags
}  // namespace ScalarAdvection
