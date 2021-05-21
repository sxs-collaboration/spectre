// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/GeneralRelativity/CheckWrappedGrConsistency.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/Tov.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/FishboneMoncriefDisk.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/SmoothFlow.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

SPECTRE_TEST_CASE(
    "Unit.PointwiseFunctions.AnalyticSolutions.GhRelativisticEuler.WrappedGr",
    "[Unit][PointwiseFunctions]") {
  const double t = 2.0;
  const tnsr::I<DataVector, 3, Frame::Inertial> x_3d{DataVector{3.0, 4.0}};
  const tnsr::I<DataVector, 2, Frame::Inertial> x_2d{DataVector{3.0, 4.0}};
  const tnsr::I<DataVector, 1, Frame::Inertial> x_1d{DataVector{3.0, 4.0}};
  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<
      RelativisticEuler::Solutions::FishboneMoncriefDisk>{
        1.0, 0.23, 6.0, 12.0, 0.001, 1.4},
      RelativisticEuler::Solutions::FishboneMoncriefDisk{
        1.0, 0.23, 6.0, 12.0, 0.001, 1.4}, x_3d, t);

  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<
          RelativisticEuler::Solutions::SmoothFlow<1>>{
          std::array<double, 1>{-0.3}, std::array<double, 1>{0.4}, 1.23, 1.334,
          0.78},
      RelativisticEuler::Solutions::SmoothFlow<1>{std::array<double, 1>{-0.3},
                                                  std::array<double, 1>{0.4},
                                                  1.23, 1.334, 0.78},
      x_1d, t);

  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<
          RelativisticEuler::Solutions::SmoothFlow<2>>{
          std::array<double, 2>{-0.3, 0.1}, std::array<double, 2>{0.4, -0.24},
          1.23, 1.334, 0.78},
      RelativisticEuler::Solutions::SmoothFlow<2>{
          std::array<double, 2>{-0.3, 0.1}, std::array<double, 2>{0.4, -0.24},
          1.23, 1.334, 0.78},
      x_2d, t);

  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<
          RelativisticEuler::Solutions::SmoothFlow<3>>{
          std::array<double, 3>{-0.3, 0.1, -0.002},
          std::array<double, 3>{0.4, -0.24, 0.054}, 1.23, 1.334, 0.78},
      RelativisticEuler::Solutions::SmoothFlow<3>{
          std::array<double, 3>{-0.3, 0.1, -0.002},
          std::array<double, 3>{0.4, -0.24, 0.054}, 1.23, 1.334, 0.78},
      x_3d, t);

  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<
          RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution>>{
          1.e-3, 8.0, 2.0},
      RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution>{
          1.e-3, 8.0, 2.0},
      x_3d, t);
}
