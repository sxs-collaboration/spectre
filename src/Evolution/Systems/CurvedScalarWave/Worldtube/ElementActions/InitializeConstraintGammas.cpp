// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/CurvedScalarWave/Worldtube/ElementActions/InitializeConstraintGammas.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace CurvedScalarWave::Worldtube::Initialization {

template <size_t Dim>
void InitializeConstraintDampingGammas<Dim>::apply(
    const gsl::not_null<Scalar<DataVector>*> gamma1,
    const gsl::not_null<Scalar<DataVector>*> gamma2,
    const tnsr::I<DataVector, Dim, Frame::Inertial>& inertial_coords) {
  const size_t number_of_grid_points = get<0>(inertial_coords).size();
  *gamma1 = Scalar<DataVector>{number_of_grid_points, 0.};
  const double amplitude = 10.;
  const double sigma = 1.e-1;
  const double constant = 1.e-4;
  const auto radius = magnitude(inertial_coords);
  get(*gamma2) = amplitude * exp(-square(sigma * radius.get())) + constant;
}

template struct InitializeConstraintDampingGammas<1>;
template struct InitializeConstraintDampingGammas<2>;
template struct InitializeConstraintDampingGammas<3>;
}  // namespace CurvedScalarWave::Worldtube::Initialization
