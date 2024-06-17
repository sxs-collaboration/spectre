// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
namespace gsl {
  template <typename T>
  class not_null;
}  // namespace gsl

class DataVector;

template<size_t Dim>
class Mesh;
/// \endcond

namespace Particles::MonteCarlo {

/// Proper 4-volume of a cell for a time step time_step.
/// We assume that determinant_spatial_metric is given
/// in inertial coordinates, hence the need for
/// det_jacobian_logical_to_inertial
void cell_proper_four_volume_finite_difference(
  gsl::not_null<Scalar<DataVector>* > cell_proper_four_volume,
  const Scalar<DataVector>& lapse,
  const Scalar<DataVector>& determinant_spatial_metric,
  double time_step,
  const Mesh<3>& mesh,
  const Scalar<DataVector>& det_jacobian_logical_to_inertial);

/// 3-volume of a cell in inertial coordinate. Note that this is
/// the coordinate volume, not the proper volume. This quantity
/// is needed as a normalization factor for the terms coupling
/// Monte-Carlo transport to the fluid evolution (as we evolved
/// densitized fluid variables in inertial coordinates).
void cell_inertial_coordinate_three_volume_finite_difference(
  gsl::not_null<Scalar<DataVector>* > cell_inertial_three_volume,
  const Mesh<3>& mesh,
  const Scalar<DataVector>& det_jacobian_logical_to_inertial);

} // namespace Particles::MonteCarlo
