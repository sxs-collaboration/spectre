// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Imex/InitialGuess.tpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Framework/TestingFramework.hpp"
#include "Utilities/Gsl.hpp"

using NeutrinoSpecies = neutrinos::ElectronNeutrinos<1>;

using tilde_e_tag =
    RadiationTransport::M1Grey::Tags::TildeE<Frame::Inertial, NeutrinoSpecies>;
using tilde_s_tag =
    RadiationTransport::M1Grey::Tags::TildeS<Frame::Inertial, NeutrinoSpecies>;
using return_tags = tmpl::list<tilde_e_tag, tilde_s_tag>;

SPECTRE_TEST_CASE("Evolution.Systems.RadiationTransport.M1Grey.InitialGuess",
                  "[Unit][M1Grey]") {
  // initialize
  const size_t num_points = 3;
  Scalar<DataVector> tilde_e_old{num_points, 6.1};

  tnsr::i<DataVector, 3> tilde_s_old;
  get<0>(tilde_s_old) = DataVector(num_points, 0.0);
  get<1>(tilde_s_old) = DataVector(num_points, 1.1);
  get<2>(tilde_s_old) = DataVector(num_points, 2.2);

  // old values to compare against
  Scalar<DataVector> tilde_e_base = tilde_e_old;
  tnsr::i<DataVector, 3> tilde_s_base = tilde_s_old;
  Scalar<DataVector>* tilde_e = &tilde_e_base;
  tnsr::i<DataVector, 3>* tilde_s = &tilde_s_base;

  // arguments for calculation
  const Scalar<DataVector> tilde_j(num_points, 0.0);
  const tnsr::i<DataVector, 3> tilde_h_spatial(num_points, 2.1);
  const Scalar<DataVector> lapse(num_points, 0.8);
  tnsr::ii<DataVector, 3, Frame::Inertial> spatial_metric;

  get<0, 0>(spatial_metric) = DataVector(num_points, 1.0);
  get<1, 1>(spatial_metric) = DataVector(num_points, 1.0);
  get<2, 2>(spatial_metric) = DataVector(num_points, 1.0);

  Variables<return_tags> inhomogeneous_terms;
  const double implicit_weight = 1.0;

  auto null_result = M1Grey::Imex::apply<return_tags>(
      tilde_e, tilde_s, tilde_j, tilde_h_spatial, lapse, spatial_metric,
      inhomogeneous_terms, implicit_weight);

  // tilde_e and tilde_s should remain unchanged
  CHECK(*tilde_e == tilde_e_old);
  CHECK(*tilde_s == tilde_s_old);
  // nothing should be returned
  CHECK(null_result.empty());
}
