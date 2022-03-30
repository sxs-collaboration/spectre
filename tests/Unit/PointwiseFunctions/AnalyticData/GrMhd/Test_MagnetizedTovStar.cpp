// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>

#include "Domain/LogicalCoordinates.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GrMhd/MagnetizedTovStar.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/RelativisticEuler/TovStar.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/GetOutput.hpp"

namespace grmhd::AnalyticData {
namespace {

static_assert(
    not is_analytic_solution_v<MagnetizedTovStar>,
    "MagnetizedTovStar should be analytic_data, and not an analytic_solution");
static_assert(
    is_analytic_data_v<MagnetizedTovStar>,
    "MagnetizedTovStar should be analytic_data, and not an analytic_solution");

using TovCoordinates = gr::Solutions::TovCoordinates;

void test_equality() {
  const MagnetizedTovStar mag_tov_original{
      1.28e-3, 100.0, 2.0, TovCoordinates::Schwarzschild, 2, 0.04, 2500.0};
  const auto mag_tov = serialize_and_deserialize(mag_tov_original);
  CHECK(mag_tov == MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                     TovCoordinates::Schwarzschild, 2, 0.04,
                                     2500.0));
  CHECK(mag_tov != MagnetizedTovStar(2.28e-3, 100.0, 2.0,
                                     TovCoordinates::Schwarzschild, 2, 0.04,
                                     2500.0));
  CHECK(mag_tov != MagnetizedTovStar(1.28e-3, 200.0, 2.0,
                                     TovCoordinates::Schwarzschild, 2, 0.04,
                                     2500.0));
  CHECK(mag_tov != MagnetizedTovStar(1.28e-3, 100.0, 3.0,
                                     TovCoordinates::Schwarzschild, 2, 0.04,
                                     2500.0));
  CHECK(mag_tov != MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                     TovCoordinates::Isotropic, 2, 0.04,
                                     2500.0));
  CHECK(mag_tov != MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                     TovCoordinates::Schwarzschild, 3, 0.04,
                                     2500.0));
  CHECK(mag_tov != MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                     TovCoordinates::Schwarzschild, 2, 0.05,
                                     2500.0));
  CHECK(mag_tov != MagnetizedTovStar(1.28e-3, 100.0, 2.0,
                                     TovCoordinates::Schwarzschild, 2, 0.04,
                                     3500.0));
}

void test_magnetized_tov_star(const TovCoordinates coord_system) {
  Parallel::register_classes_with_charm<
      grmhd::AnalyticData::MagnetizedTovStar>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          grmhd::AnalyticData::MagnetizedTovStar>(
          "MagnetizedTovStar:\n"
          "  CentralDensity: 1.28e-3\n"
          "  PolytropicConstant: 100.0\n"
          "  PolytropicExponent: 2.0\n"
          "  Coordinates: " +
          get_output(coord_system) +
          "\n"
          "  PressureExponent: 2\n"
          "  VectorPotentialAmplitude: 2500\n"
          "  CutoffPressureFraction: 0.04\n");
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& mag_tov =
      dynamic_cast<const grmhd::AnalyticData::MagnetizedTovStar&>(
          *deserialized_option_solution);

  const RelativisticEuler::Solutions::TovStar tov{1.28e-3, 100.0, 2.0,
                                                  coord_system};

  std::unique_ptr<EquationsOfState::EquationOfState<true, 1>> eos =
      std::make_unique<EquationsOfState::PolytropicFluid<true>>(100.0, 2.0);

  const Mesh<3> mesh{
      {{5, 5, 5}},
      {{Spectral::Basis::Legendre, Spectral::Basis::Legendre,
        Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::GaussLobatto,
        Spectral::Quadrature::GaussLobatto}}};
  const auto log_coords = logical_coordinates(mesh);

  // Coordinates where we check the data. Includes the origin.
  tnsr::I<DataVector, 3, Frame::Inertial> in_coords{
      mesh.number_of_grid_points(), 0.0};
  const double scale = 1.0e-2;
  in_coords.get(0) = scale * (log_coords.get(0) + 1.);
  InverseJacobian<DataVector, 3, Frame::ElementLogical, Frame::Inertial>
      inv_jac{mesh.number_of_grid_points(), 0.0};
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
       mag_tov_values = mag_tov.variables(in_coords, tov_tags{})](auto tag_v) {
        using tag = tmpl::type_from<decltype(tag_v)>;
        CHECK_ITERABLE_APPROX(get<tag>(mag_tov_values), get<tag>(tov_values));
      });

  // Verify that the resulting magnetic field has (approximately) vanishing
  // covariant divergence, but is non-zero overall.
  INFO("Check magnetic field");
  const auto vars = mag_tov.variables(
      in_coords,
      tmpl::list<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>,
                 gr::Tags::SqrtDetSpatialMetric<DataVector>>{});
  const auto& b_field =
      get<hydro::Tags::MagneticField<DataVector, 3, Frame::Inertial>>(vars);
  const auto& sqrt_det_spatial_metric =
      get<gr::Tags::SqrtDetSpatialMetric<DataVector>>(vars);
  auto tilde_b = b_field;
  double b_field_l2norm = 0.;
  for (size_t i = 0; i < 3; ++i) {
    tilde_b.get(i) *= get(sqrt_det_spatial_metric);
    b_field_l2norm += sum(square(b_field.get(i)));
  }

  b_field_l2norm = sqrt(b_field_l2norm);
  CHECK(b_field_l2norm != approx(0.));
  const auto div_tilde_b = divergence(tilde_b, mesh, inv_jac);
  CHECK(max(abs(get(div_tilde_b))) < 1.0e-14);
}

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticData.GrMhd.MagTovStar",
                  "[Unit][PointwiseFunctions]") {
  test_equality();
  test_magnetized_tov_star(TovCoordinates::Schwarzschild);
  test_magnetized_tov_star(TovCoordinates::Isotropic);
}

}  // namespace
}  // namespace grmhd::AnalyticData
