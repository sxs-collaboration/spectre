// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <utility>

#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/System.hpp"
#include "Evolution/Systems/RadiationTransport/M1Grey/Tags.hpp"
#include "Evolution/Systems/RadiationTransport/Tags.hpp"
#include "Helpers/Evolution/Imex/TestSector.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "PointwiseFunctions/Hydro/Tags.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.RadiationTransport.M1Grey.System.Imex",
                  "[Unit][M1Grey]") {
  using DummySpecies = neutrinos::ElectronNeutrinos<1>;

  using system = RadiationTransport::M1Grey::System<tmpl::list<DummySpecies>>;
  using sector = tmpl::front<system::implicit_sectors>;

  // Metric
  Scalar<DataVector> lapse{};
  get(lapse) = DataVector{1.0};
  tnsr::ii<DataVector, 3> spatial_metric{};
  get<0, 0>(spatial_metric) = DataVector{3.0};
  get<0, 1>(spatial_metric) = DataVector{0.0};
  get<0, 2>(spatial_metric) = DataVector{0.0};
  get<1, 1>(spatial_metric) = DataVector{4.0};
  get<1, 2>(spatial_metric) = DataVector{0.0};
  get<2, 2>(spatial_metric) = DataVector{5.0};
  tnsr::II<DataVector, 3> inverse_spatial_metric{};
  Scalar<DataVector> sqrt_det_spatial_metric{};
  determinant_and_inverse(make_not_null(&sqrt_det_spatial_metric),
                          make_not_null(&inverse_spatial_metric),
                          spatial_metric);
  get(sqrt_det_spatial_metric) = sqrt(get(sqrt_det_spatial_metric));

  // neutrino properties
  Scalar<DataVector> emissivity{};
  get(emissivity) = DataVector{2.0};
  Scalar<DataVector> absorption_opacity{};
  get(absorption_opacity) = DataVector{4.0};
  Scalar<DataVector> scattering_opacity{};
  get(scattering_opacity) = DataVector{3.0};
  // fluid velocity (nonzero)
  tnsr::I<DataVector, 3> fluid_velocity{};
  get<0>(fluid_velocity) = DataVector{0.01};
  get<1>(fluid_velocity) = DataVector{0.02};
  get<2>(fluid_velocity) = DataVector{0.03};

  Scalar<DataVector> lorentz_factor{1_st};
  tenex::evaluate(
      make_not_null(&lorentz_factor),
      1.0 / sqrt(1.0 - fluid_velocity(ti::I) * fluid_velocity(ti::J) *
                           spatial_metric(ti::i, ti::j)));

  tnsr::I<DataVector, 3> fluid_velocity_zero_vel{};
  get<0>(fluid_velocity_zero_vel) = DataVector{0.0};
  get<1>(fluid_velocity_zero_vel) = DataVector{0.0};
  get<2>(fluid_velocity_zero_vel) = DataVector{0.0};

  Scalar<DataVector> lorentz_factor_zero_vel{1_st};
  tenex::evaluate(make_not_null(&lorentz_factor_zero_vel),
                  1.0 / sqrt(1.0 - fluid_velocity_zero_vel(ti::I) *
                                       fluid_velocity_zero_vel(ti::J) *
                                       spatial_metric(ti::i, ti::j)));

  const double stencil_size = 1.0e-4;
  const double tolerance = 1.0e-7;

  // has to match tags_from_evolution in System.hpp :: M1Solve
  const tuples::TaggedTuple<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
      RadiationTransport::M1Grey::Tags::GreyEmissivity<DummySpecies>,
      RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity<DummySpecies>,
      RadiationTransport::M1Grey::Tags::GreyScatteringOpacity<DummySpecies>,
      hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>>
      list_of_values = {lapse,
                        spatial_metric,
                        sqrt_det_spatial_metric,
                        inverse_spatial_metric,
                        emissivity,
                        absorption_opacity,
                        scattering_opacity,
                        lorentz_factor,
                        fluid_velocity};

  const tuples::TaggedTuple<
      gr::Tags::Lapse<DataVector>,
      gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
      gr::Tags::SqrtDetSpatialMetric<DataVector>,
      gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
      RadiationTransport::M1Grey::Tags::GreyEmissivity<DummySpecies>,
      RadiationTransport::M1Grey::Tags::GreyAbsorptionOpacity<DummySpecies>,
      RadiationTransport::M1Grey::Tags::GreyScatteringOpacity<DummySpecies>,
      hydro::Tags::LorentzFactor<DataVector>,
      hydro::Tags::SpatialVelocity<DataVector, 3>>
      list_of_values_zero_vel = {lapse,
                                 spatial_metric,
                                 sqrt_det_spatial_metric,
                                 inverse_spatial_metric,
                                 emissivity,
                                 absorption_opacity,
                                 scattering_opacity,
                                 lorentz_factor_zero_vel,
                                 fluid_velocity_zero_vel};

  using sector_variables_tag = Tags::Variables<sector::tensors>;
  using SectorVariables = sector_variables_tag::type;

  // note, here, explicit means not implicit, in the time stepper sense
  SectorVariables explicit_values(1);
  Scalar<DataVector>& tilde_e_vals = get<
      RadiationTransport::M1Grey::Tags::TildeE<Frame::Inertial, DummySpecies>>(
      explicit_values);
  get(tilde_e_vals) = DataVector{1.001};

  const double momentum_density_optically_thin = 2.0;
  const double momentum_density_optically_intermediate = 1.0;
  const double momentum_density_optically_thick = 5.0e-3;

  const std::vector<double> momentum_densities = {
      momentum_density_optically_thin, momentum_density_optically_intermediate,
      momentum_density_optically_thick};

  // loop over different momentum densities for each optical regime
  for (const double momentum_density : momentum_densities) {
    tnsr::i<DataVector, 3>& tilde_s_vals =
        get<RadiationTransport::M1Grey::Tags::TildeS<Frame::Inertial,
                                                     DummySpecies>>(
            explicit_values);
    get<0>(tilde_s_vals) = DataVector{0.0};
    get<1>(tilde_s_vals) = DataVector{momentum_density};
    get<2>(tilde_s_vals) = DataVector{0.0};

    // zero velocity tests
    TestHelpers::imex::test_sector<sector>(
        stencil_size, tolerance, explicit_values, list_of_values_zero_vel);

    // nonzero velocity tests
    TestHelpers::imex::test_sector<sector>(stencil_size, tolerance,
                                           explicit_values, list_of_values);
  }
}
