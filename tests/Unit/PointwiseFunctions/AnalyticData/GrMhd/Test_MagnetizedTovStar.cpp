// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.MagTovStar",
                  "[Unit][PointwiseFunctions]") {
  const auto mag_tov_opts =
      TestHelpers::test_creation<grmhd::AnalyticData::MagnetizedTovStar>(
          "CentralDensity: 1.28e-3\n"
          "PolytropicConstant: 100.0\n"
          "PolytropicExponent: 2.0\n"
          "PressureExponent: 2\n"
          "VectorPotentialAmplitude: 2500\n"
          "CutoffPressureFraction: 0.04\n");
  const RelativisticEuler::Solutions::TovStar<gr::Solutions::TovSolution> tov{
      1.28e-3, 100.0, 2.0};
  const auto mag_tov = serialize_and_deserialize(mag_tov_opts);
  CHECK(mag_tov == grmhd::AnalyticData::MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                                          2, 0.04, 2500.0));
  CHECK(mag_tov != grmhd::AnalyticData::MagnetizedTovStar(2.28e-3, 100.0, 2.0,
                                                          2, 0.04, 2500.0));
  CHECK(mag_tov != grmhd::AnalyticData::MagnetizedTovStar(1.28e-3, 200.0, 2.0,
                                                          2, 0.04, 2500.0));
  CHECK(mag_tov != grmhd::AnalyticData::MagnetizedTovStar(1.28e-3, 100.0, 3.0,
                                                          2, 0.04, 2500.0));
  CHECK(mag_tov != grmhd::AnalyticData::MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                                          3, 0.04, 2500.0));
  CHECK(mag_tov != grmhd::AnalyticData::MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                                          2, 0.05, 2500.0));
  CHECK(mag_tov != grmhd::AnalyticData::MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                                          2, 0.04, 3500.0));

  std::unique_ptr<EquationsOfState::EquationOfState<true, 1>> eos =
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0);

  const gr::Solutions::TovSolution tov_soln{*eos, 1.28e-3};
  const Mesh<3> mesh{{{5, 5, 5}},
                     {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
                       Spectral::Basis::Legendre}},
                     {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Gauss,
                       Spectral::Quadrature::Gauss}}};
  const auto log_coords = logical_coordinates(mesh);

  tnsr::I<DataVector, 3, Frame::Inertial> in_coords{
      mesh.number_of_grid_points(), 0.0};
  const double scale = 1.0e-2;
  in_coords.get(0) = scale * (log_coords.get(0) + 1.1);
  InverseJacobian<DataVector, 3, Frame::Logical, Frame::Inertial> inv_jac{
      mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < 3; ++i) {
    inv_jac.get(i, i) = 1.0 / scale;
  }

  // Check that the non-magnetic field tags match the unmagnetized solution.
  using tov_tags =
      tmpl::remove<tmpl::append<hydro::grmhd_tags<DataVector>,
                                gr::tags_for_hydro<3, DataVector>>,
                   hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>;
  tmpl::for_each<tov_tags>(
      [tov_values = tov.variables(in_coords, 0.0, tov_tags{}),
       mag_tov_values =
           mag_tov.variables(in_coords, tov_tags{})](auto tag_v) noexcept {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(mag_tov_values), get<tag>(tov_values));
      });

  // Verify that the resulting magnetic field has (approximately) vanishing
  // covariant divergence
  const auto b_field =
      get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(
          mag_tov.variables(in_coords, tmpl::list<hydro::Tags::MagneticField<
                                           DataVector, 3, Frame::Inertial>>{}));
  const auto sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(mag_tov.variables(
          in_coords, tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataVector>>{}));
  auto tilde_b = b_field;
  for (size_t i = 0; i < 3; ++i) {
    tilde_b.get(i) *= get(sqrt_det_spatial_metric);
  }

  const auto div_tilde_b = divergence(tilde_b, mesh, inv_jac);
  CHECK(max(abs(get(div_tilde_b))) < 1.0e-14);
}
