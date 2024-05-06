// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/WorldtubeConstraint.hpp"

namespace Cce {
void ComputeKGWorldtubeConstraint::apply(
    gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
        evolution_kg_constraint,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& cauchy_kg_psi,
    const Spectral::Swsh::SwshInterpolator& interpolator, const size_t l_max,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& volume_psi) {
  SpinWeighted<ComplexDataVector, 0> evolution_kg_psi;
  interpolator.interpolate(make_not_null(&evolution_kg_psi),
                           get(cauchy_kg_psi));

  const SpinWeighted<ComplexDataVector, 0> surface_psi;
  make_const_view(make_not_null(&surface_psi), get(volume_psi), 0,
                  Spectral::Swsh::number_of_swsh_collocation_points(l_max));
  get(*evolution_kg_constraint) = evolution_kg_psi - surface_psi;
}
}  // namespace Cce
