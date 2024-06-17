// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/TypeAliases.hpp"

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

void cell_light_crossing_time(
    gsl::not_null<Scalar<DataVector>*> cell_light_crossing_time,
    const Mesh<3>& mesh,
    const tnsr::I<DataVector, 3, Frame::Inertial>& inertial_coordinates,
    const Scalar<DataVector>& lapse,
    const tnsr::I<DataVector, 3, Frame::Inertial>& shift,
    const tnsr::II<DataVector, 3, Frame::Inertial>& inv_spatial_metric);

}  // namespace Particles::MonteCarlo
