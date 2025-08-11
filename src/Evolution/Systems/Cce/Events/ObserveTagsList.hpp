// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "Evolution/Systems/Cce/LinearOperators.hpp"
#include "Evolution/Systems/Cce/NewmanPenrose.hpp"
#include "Evolution/Systems/Cce/OptionTags.hpp"
#include "Evolution/Systems/Cce/SwshDerivatives.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Events {
template <typename Tag, bool IncludeSecondDeriv = true>
// clang-format off
  using zero_one_two_radial_derivs = tmpl::flatten<tmpl::list<
      Tag,
      Tags::Dy<Tag>,
      tmpl::conditional_t<IncludeSecondDeriv,
                          Tags::Dy<Tags::Dy<Tag>>,
                          tmpl::list<>>>>;
  using spin_weighted_tags_to_observe = tmpl::flatten<
      tmpl::list<zero_one_two_radial_derivs<Tags::BondiBeta>,
                 zero_one_two_radial_derivs<Tags::BondiU>,
                 zero_one_two_radial_derivs<Tags::BondiQ>,
                 zero_one_two_radial_derivs<Tags::BondiW>,
                 zero_one_two_radial_derivs<Tags::BondiH, false>,
                 zero_one_two_radial_derivs<Tags::BondiJ>,
                 zero_one_two_radial_derivs<Tags::Du<Tags::BondiJ>>,
                 Tags::BondiR,
                 Tags::Psi0,
                 Tags::Psi1,
                 Tags::Psi2,
                 Tags::NewmanPenroseAlpha,
                 Tags::NewmanPenroseBeta,
                 Tags::NewmanPenroseGamma,
                 Tags::NewmanPenroseEpsilon,
                 // Tags::NewmanPenroseKappa,
                 // in our choice of tetrad, \kappa=0
                 Tags::NewmanPenroseTau,
                 Tags::NewmanPenroseSigma,
                 Tags::NewmanPenroseRho,
                 Tags::NewmanPenrosePi,
                 Tags::NewmanPenroseNu,
                 Tags::NewmanPenroseMu,
                 Tags::NewmanPenroseLambda,
                 Tags::EthRDividedByR,
                 Tags::DuRDividedByR>>;
// clang-format on

using available_volume_tags_to_observe =
    tmpl::push_back<spin_weighted_tags_to_observe,
                    Tags::ComplexInertialRetardedTime, Tags::OneMinusY>;

using compute_tags_for_observation_box = tmpl::list<
    Tags::Psi0Compute, Tags::Psi1Compute, Tags::Psi2Compute,
    Tags::SwshDerivativeCompute<Tags::BondiJ, Spectral::Swsh::Tags::Eth>,
    Tags::SwshDerivativeCompute<Tags::BondiW, Spectral::Swsh::Tags::Eth>,
    Tags::NewmanPenroseAlphaCompute, Tags::NewmanPenroseBetaCompute,
    Tags::NewmanPenroseGammaCompute, Tags::NewmanPenroseEpsilonCompute,
    // Tags::NewmanPenroseKappaCompute,
    // in our choice of tetrad, \kappa=0
    Tags::NewmanPenroseTauCompute, Tags::NewmanPenroseSigmaCompute,
    Tags::NewmanPenroseRhoCompute, Tags::NewmanPenrosePiCompute,
    Tags::NewmanPenroseNuCompute, Tags::NewmanPenroseMuCompute,
    Tags::NewmanPenroseLambdaCompute,
    Tags::SwshDerivativeCompute<Tags::NewmanPenrosePi,
                                Spectral::Swsh::Tags::Eth>,
    Tags::SwshDerivativeCompute<Tags::NewmanPenrosePi,
                                Spectral::Swsh::Tags::Ethbar>,
    Tags::DyCompute<Tags::NewmanPenrosePi>,
    Tags::DyCompute<Tags::NewmanPenroseMu>>;
}  // namespace Cce::Events
