// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/CurvedScalarWave/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace CurvedScalarWave {
namespace Tags {
/*!
 * \brief Compute items to compute constraint damping parameters.
 *
 * \details For the evolution system with constraint damping parameters
 * to be symmetric-hyperbolic, we need \f$\gamma_1 \gamma_2 = 0\f$. When
 * \f$\gamma_1 = 0\f$, Ref. \cite Holst2004wt shows that the one-index
 * constraint decays exponentially on a time-scale \f$ 1/\gamma_2\f$.
 * Conversely, they also show that using \f$\gamma_2 > 0\f$ leads to
 * exponential suppression of constraint violations.
 *
 * Can be retrieved using `CurvedScalarWave::Tags::ConstraintGamma1`
 * and `CurvedScalarWave::Tags::ConstraintGamma2`.
 */
struct ConstraintGamma1Compute : ConstraintGamma1, db::ComputeTag {
  using argument_tags = tmpl::list<Psi>;
  static auto function(const Scalar<DataVector>& used_for_size) noexcept {
    return make_with_value<type>(used_for_size, 0.);
  }
  using base = ConstraintGamma1;
};
/// \copydoc ConstraintGamma1Compute
struct ConstraintGamma2Compute : ConstraintGamma2, db::ComputeTag {
  using argument_tags = tmpl::list<Psi>;
  static auto function(const Scalar<DataVector>& used_for_size) noexcept {
    return make_with_value<type>(used_for_size, 1.);
  }
  using base = ConstraintGamma2;
};
}  // namespace Tags
}  // namespace CurvedScalarWave
