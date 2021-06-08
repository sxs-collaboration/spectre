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
  const double t = 2.0;
  const tnsr::I<DataVector, 3, Frame::Inertial> x{DataVector{3.0, 4.0}};

  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<grmhd::Solutions::AlfvenWave>{
          2.2, 1.23, 0.2, 1.4, {{0.0, 0.0, 2.0}}, {{0.75, 0.0, 0.0}}},
      grmhd::Solutions::AlfvenWave{
          2.2, 1.23, 0.2, 1.4, {{0.0, 0.0, 2.0}}, {{0.75, 0.0, 0.0}}},
      x, t);

  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<grmhd::Solutions::BondiMichel>{
          1.1, 50.0, 1.3, 1.5, 0.24},
      grmhd::Solutions::BondiMichel{1.1, 50.0, 1.3, 1.5, 0.24}, x, t);

  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<
          grmhd::Solutions::KomissarovShock>{
          4. / 3., 1., 3.323, 10., 55.36,
          std::array<double, 3>{{0.8370659816473115, 0., 0.}},
          std::array<double, 3>{{0.6202085442748952, -0.44207111995019704, 0.}},
          std::array<double, 3>{{10., 18.28, 0.}},
          std::array<double, 3>{{10., 14.49, 0.}}, 0.5},
      grmhd::Solutions::KomissarovShock{
          4. / 3., 1., 3.323, 10., 55.36,
          std::array<double, 3>{{0.8370659816473115, 0., 0.}},
          std::array<double, 3>{{0.6202085442748952, -0.44207111995019704, 0.}},
          std::array<double, 3>{{10., 18.28, 0.}},
          std::array<double, 3>{{10., 14.49, 0.}}, 0.5},
      x, t);

  check_wrapped_gr_solution_consistency(
      GeneralizedHarmonic::Solutions::WrappedGr<grmhd::Solutions::SmoothFlow>{
          {{0.24, 0.11, 0.04}}, {{0.14, 0.42, -0.03}}, 1.3, 1.5, 0.24},
      grmhd::Solutions::SmoothFlow{
          {{0.24, 0.11, 0.04}}, {{0.14, 0.42, -0.03}}, 1.3, 1.5, 0.24},
      x, t);
}
