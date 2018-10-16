// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/FaceNormal.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"

/// \cond
class DataVector;
namespace Tags {
template <typename Tag>
struct Normalized;
}  // namespace Tags
/// \endcond

// IWYU pragma: no_forward_declare Tensor

namespace GeneralizedHarmonic {
/*!
 * \ingroup GeneralizedHarmonic
 * \brief Compute the characteristic speeds for the generalized harmonic system.
 *
 * Computes the speeds as described in "A New Generalized Harmonic
 * Evolution System" by Lindblom et. al, https://arxiv.org/abs/gr-qc/0512093.
 * The characteristic fields' names used here differ from this paper:
 *
 * \f{align*}
 * \mathrm{SpECTRE} && \mathrm{Lindblom} \\
 * u^{\psi}_{ab} && u^0_{ab} \\
 * u^{\pm}_{ab} && u^{1\pm}_{ab} \\
 * u^0_{iab} && u^2_{iab}
 * \f}
 *
 * The speeds \f$v\f$, are given by:
 * \f{align*}
 * v_{\psi} =& -(1 + \gamma_1) n_k N^k \\
 * v_{0} =& -n_k N^k \\
 * v_{\pm} =& -n_k N^k \pm N
 * \f}
 *
 * where \f$N, N^k\f$ are the lapse and shift respectively, \f$\gamma_1\f$ is a
 * constraint damping parameter, and \f$n_k\f$ is the unit normal to the
 * surface.
 */
template <size_t Dim, typename Frame>
struct CharacteristicSpeedsCompute : db::ComputeTag {
  using argument_tags = tmpl::list<
      Tags::ConstraintGamma1, gr::Tags::Lapse<DataVector>,
      gr::Tags::Shift<Dim, Frame, DataVector>,
      ::Tags::Normalized<::Tags::UnnormalizedFaceNormal<Dim, Frame>>>;

  using volume_tags = tmpl::list<>;

  static typename Tags::CharacteristicSpeeds<Dim, Frame>::type function(
      const Scalar<DataVector>& gamma_1, const Scalar<DataVector>& lapse,
      const tnsr::I<DataVector, Dim, Frame>& shift,
      const tnsr::i<DataVector, Dim, Frame>& normal) noexcept;
};
}  // namespace GeneralizedHarmonic
