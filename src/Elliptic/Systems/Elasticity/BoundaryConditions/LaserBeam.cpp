// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Elliptic/Systems/Elasticity/BoundaryConditions/LaserBeam.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"

namespace Elasticity::BoundaryConditions {

void LaserBeam::apply(
    const gsl::not_null<tnsr::I<DataVector, 3>*> /*displacement*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress,
    const tnsr::I<DataVector, 3>& x,
    const tnsr::i<DataVector, 3>& face_normal) const {
  const auto n_dot_x = get<0>(face_normal) * get<0>(x) +
                       get<1>(face_normal) * get<1>(x) +
                       get<2>(face_normal) * get<2>(x);
  const auto r_sq = square(get<0>(x)) + square(get<1>(x)) + square(get<2>(x)) -
                    square(n_dot_x);
  const DataVector beam_profile =
      exp(-r_sq / square(beam_width_)) / (M_PI * square(beam_width_));
  get<0>(*n_dot_minus_stress) = -beam_profile * get<0>(face_normal);
  get<1>(*n_dot_minus_stress) = -beam_profile * get<1>(face_normal);
  get<2>(*n_dot_minus_stress) = -beam_profile * get<2>(face_normal);
}

void LaserBeam::apply_linearized(
    const gsl::not_null<tnsr::I<DataVector, 3>*> /*displacement*/,
    const gsl::not_null<tnsr::I<DataVector, 3>*> n_dot_minus_stress) {
  get<0>(*n_dot_minus_stress) = 0.;
  get<1>(*n_dot_minus_stress) = 0.;
  get<2>(*n_dot_minus_stress) = 0.;
}

bool operator==(const LaserBeam& lhs, const LaserBeam& rhs) {
  return lhs.beam_width() == rhs.beam_width();
}

bool operator!=(const LaserBeam& lhs, const LaserBeam& rhs) {
  return not(lhs == rhs);
}

PUP::able::PUP_ID LaserBeam::my_PUP_ID = 0;  // NOLINT

}  // namespace Elasticity::BoundaryConditions
