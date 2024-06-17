// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <vector>

#include "DataStructures/Tags/TempTensor.hpp"
#include "Evolution/Imex/GuessResult.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/Gsl.hpp"

/// \cond
class DataVector;
template <typename>
class Variables;
namespace gsl {
template <typename T>
class not_null;
}  // namespace gsl
/// \endcond

namespace M1Grey::Imex {

template <typename NeutrinoSpeciesList>
struct InitialGuess;

template <typename... NeutrinoSpecies>
struct InitialGuess<tmpl::list<NeutrinoSpecies...>> {
  using return_tags =
      tmpl::list<RadiationTransport::M1Grey::Tags::TildeE<Frame::Inertial,
                                                          NeutrinoSpecies>...,
                 RadiationTransport::M1Grey::Tags::TildeS<Frame::Inertial,
                                                          NeutrinoSpecies>...>;
  using argument_tags = tmpl::list<
      RadiationTransport::M1Grey::Tags::TildeJ<NeutrinoSpecies>...,
      RadiationTransport::M1Grey::Tags::TildeHSpatial<Frame::Inertial,
                                                      NeutrinoSpecies>...,
      gr::Tags::Lapse<DataVector>, gr::Tags::SpatialMetric<DataVector, 3>>;

  // This is a stub implementation for initial guesses for the M1 system, should
  // a more accurate guess in the future be needed. Currently, the M1 system
  // just uses the explicit solution as an initial guess.
  template <typename return_tags>
  static std::vector<imex::GuessResult> apply(
      [[maybe_unused]] gsl::not_null<Scalar<DataVector>*> tilde_e,
      [[maybe_unused]] gsl::not_null<tnsr::i<DataVector, 3>*> tilde_s,
      [[maybe_unused]] const Scalar<DataVector>& tilde_j,
      [[maybe_unused]] const tnsr::i<DataVector, 3>& tilde_h_spatial,
      [[maybe_unused]] const Scalar<DataVector>& lapse,
      [[maybe_unused]] const tnsr::ii<DataVector, 3, Frame::Inertial>&
          spatial_metric,
      [[maybe_unused]] const Variables<return_tags>& inhomogeneous_terms,
      [[maybe_unused]] double implicit_weight);
};

}  // namespace M1Grey::Imex
