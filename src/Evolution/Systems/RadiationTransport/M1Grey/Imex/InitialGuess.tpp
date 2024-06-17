// Distributed under the MIT License.
// See LICENSE.txt for details.
#pragma once

#include "Evolution/Systems/RadiationTransport/M1Grey/Imex/InitialGuess.hpp"

#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/EagerMath/DotProduct.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/Gsl.hpp"

namespace M1Grey::Imex {

template <typename inhomogeneous_types>
static std::vector<imex::GuessResult> apply(
    [[maybe_unused]] gsl::not_null<Scalar<DataVector>*> tilde_e,
    [[maybe_unused]] gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    [[maybe_unused]] const Scalar<DataVector>& tilde_j,
    [[maybe_unused]] const tnsr::i<DataVector, 3>& tilde_h_spatial,
    [[maybe_unused]] const Scalar<DataVector>& lapse,
    [[maybe_unused]] const tnsr::ii<DataVector, 3, Frame::Inertial>&
        spatial_metric,
    [[maybe_unused]] const Variables<inhomogeneous_types>& inhomogeneous_terms,
    [[maybe_unused]] const double implicit_weight) {
  //  Note the empty return guesses the explicit result
  return {};
}

using NeutrinoSpecies = neutrinos::ElectronNeutrinos<1>;

using tilde_e_tag =
    RadiationTransport::M1Grey::Tags::TildeE<Frame::Inertial, NeutrinoSpecies>;
using tilde_s_tag =
    RadiationTransport::M1Grey::Tags::TildeS<Frame::Inertial, NeutrinoSpecies>;
using return_tags = tmpl::list<tilde_e_tag, tilde_s_tag>;

// explicit instantiation
template std::vector<imex::GuessResult> apply<return_tags>(
    gsl::not_null<Scalar<DataVector>*> tilde_e,
    gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
    const Scalar<DataVector>& tilde_j,
    const tnsr::i<DataVector, 3>& tilde_h_spatial,
    const Scalar<DataVector>& lapse,
    const tnsr::ii<DataVector, 3, Frame::Inertial>& spatial_metric,
    const Variables<return_tags>& inhomogeneous_terms, double implicit_weight);

}  // namespace M1Grey::Imex
