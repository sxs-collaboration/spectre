// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce {
/*!
 * \brief Compute a worldtube constraint for Klein-Gordon Cce
 *
 * \details Both Cauchy and characteristic systems evolve the scalar field
 * independently. The results have to be consistent at the worldtube. The
 * difference (`Tags::KleinGordonWorldtubeConstraint`) between the Cauchy
 * worldtube data `Tags::BoundaryValue<Tags::KleinGordonPsi>` and the evolved
 * volume data `Tags::KleinGordonPsi` (at the worldtube)
 * indicates the accuracy of the simulation.
 */
struct ComputeKGWorldtubeConstraint {
  using return_tags = tmpl::list<Tags::KleinGordonWorldtubeConstraint>;
  using argument_tags = tmpl::list<
      Tags::BoundaryValue<Tags::KleinGordonPsi>,
      Spectral::Swsh::Tags::SwshInterpolator<Tags::CauchyAngularCoords>,
      Tags::LMax, Tags::KleinGordonPsi>;

  static void apply(
      gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*>
          evolution_kg_constraint,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& cauchy_kg_psi,
      const Spectral::Swsh::SwshInterpolator& interpolator, size_t l_max,
      const Scalar<SpinWeighted<ComplexDataVector, 0>>& volume_psi);
};
}  // namespace Cce
