// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/CheckWrappedGrConsistency.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/AlfvenWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/BondiMichel.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/KomissarovShock.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GrMhd/SmoothFlow.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.GhGrMhd.WrappedGr",
                  "[Unit][PointwiseFunctions]") {
  const double time = 2.0;
  const tnsr::I<DataVector, 3, Frame::Inertial> coords{DataVector{3.0, 4.0}};

  check_wrapped_gr_solution_consistency(
      gh::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>{1.1, 50.0, 1.3,
                                                              1.5, 0.24},
      grmhd::Solutions::BondiMichel{1.1, 50.0, 1.3, 1.5, 0.24}, coords, time);
}
