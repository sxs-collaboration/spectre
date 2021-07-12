// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/Systems/ScalarAdvection/Tags.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace ScalarAdvection::Tags {
/*!
 * \brief Compute the advection velocity field \f$v\f$ of the ScalarAdvection
 * system
 *
 * - For 1D problem, \f$v(x) = 1.0\f$
 * - For 2D problem, \f$v(x,y) = (0.5-y,-0.5+x)\f$
 *
 */
template <size_t Dim>
struct VelocityFieldCompute : VelocityField<Dim>, db::ComputeTag {
  using argument_tags =
      tmpl::list<domain::Tags::Coordinates<Dim, Frame::Inertial>>;
  using return_type = tnsr::I<DataVector, Dim>;
  using base = VelocityField<Dim>;

  static void function(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> velocity_field,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          inertial_coords) noexcept;
};
}  // namespace ScalarAdvection::Tags
