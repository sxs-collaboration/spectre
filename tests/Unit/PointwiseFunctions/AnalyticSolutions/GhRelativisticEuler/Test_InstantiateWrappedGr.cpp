// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/CheckWrappedGrConsistency.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.GhRelativisticEuler.WrappedGr",
    "[Unit][PointwiseFunctions]") {
  const double t = 2.0;
  const tnsr::I<DataVector, 3, Frame::Inertial> x_3d{DataVector{3.0, 4.0}};
  const tnsr::I<DataVector, 2, Frame::Inertial> x_2d{DataVector{3.0, 4.0}};
  const tnsr::I<DataVector, 1, Frame::Inertial> x_1d{DataVector{3.0, 4.0}};
  check_wrapped_gr_solution_consistency(
      gh::Solutions::WrappedGr<
          RelativisticEuler::Solutions::FishboneMoncriefDisk>{1.0, 0.23, 6.0,
                                                              12.0, 0.001, 1.4},
      RelativisticEuler::Solutions::FishboneMoncriefDisk{1.0, 0.23, 6.0, 12.0,
                                                         0.001, 1.4},
      x_3d, t);

  check_wrapped_gr_solution_consistency(
      gh::Solutions::WrappedGr<RelativisticEuler::Solutions::TovStar>{
          1.e-3,
          std::make_unique<EquationsOfState::PolytropicFluid<true>>(8.0, 2.0)},
      RelativisticEuler::Solutions::TovStar{
          1.e-3,
          std::make_unique<EquationsOfState::PolytropicFluid<true>>(8.0, 2.0)},
      x_3d, t);
}
