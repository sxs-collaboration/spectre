// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <limits>
#include <memory>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/DeterminantAndInverse.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/AnalyticChristoffel.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Dispatch.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/Gauges.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/GaugeSourceFunctions/RegisterDerived.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/GeneralRelativity/GeneralizedHarmonic/Christoffel.hpp"
#include "PointwiseFunctions/GeneralRelativity/Lapse.hpp"
#include "PointwiseFunctions/GeneralRelativity/Shift.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalOneForm.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpacetimeNormalVector.hpp"
#include "PointwiseFunctions/GeneralRelativity/SpatialMetric.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"

namespace {
template <size_t Dim>
auto make_coord_map() {
  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  using Affine3D =
      domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;
  if constexpr (Dim == 1) {
    Affine affine_map{-1.0, 1.0, 2.0, 3.0};
    return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
        affine_map);
  } else if constexpr (Dim == 2) {
    Affine affine_map{-1.0, 1.0, 2.0, 3.0};
    Affine2D product_map{affine_map, affine_map};
    return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
        product_map);
  } else {
    Affine affine_map{-1.0, 1.0, 2.0, 3.0};
    Affine3D product_map{affine_map, affine_map, affine_map};
    return domain::make_coordinate_map<Frame::ElementLogical, Frame::Inertial>(
        product_map);
  }
}

template <size_t Dim>
struct Metavariables {
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<
        tmpl::pair<gh::gauges::GaugeCondition,
                   tmpl::list<gh::gauges::AnalyticChristoffel>>,
        tmpl::pair<
            evolution::initial_data::InitialData,
            tmpl::list<
                gh::Solutions::WrappedGr<gr::Solutions::KerrSchild>,
                gh::Solutions::WrappedGr<gr::Solutions::GaugeWave<Dim>>>>>;
  };
};

template <size_t Dim>
void test_gauge_wave(const Mesh<Dim>& mesh) {
  const auto gauge_condition = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<gh::gauges::GaugeCondition>,
                                 Metavariables<Dim>>("AnalyticChristoffel:\n"
                                                     "  AnalyticPrescription:\n"
                                                     "    GaugeWave:\n"
                                                     "      Amplitude: 0.0012\n"
                                                     "      Wavelength: 1.4\n")
          ->get_clone());

  const size_t num_points = mesh.number_of_grid_points();

  const double time = 1.2;
  const auto coord_map = make_coord_map<Dim>();
  const auto logical_coords = logical_coordinates(mesh);
  const tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords =
      coord_map(logical_coords);
  const auto inverse_jacobian = coord_map.inv_jacobian(logical_coords);

  tnsr::a<DataVector, Dim, Frame::Inertial> gauge_h(num_points);
  tnsr::ab<DataVector, Dim, Frame::Inertial> d4_gauge_h(num_points);
  // Used dispatch with defaulted arguments that we don't need for Analytic
  // gauge.
  gh::gauges::dispatch(make_not_null(&gauge_h), make_not_null(&d4_gauge_h), {},
                       {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, mesh, time,
                       inertial_coords, inverse_jacobian, *gauge_condition);

  CHECK_ITERABLE_APPROX(
      gauge_h, (tnsr::a<DataVector, Dim, Frame::Inertial>(num_points, 0.0)));
  CHECK_ITERABLE_APPROX(d4_gauge_h, (tnsr::ab<DataVector, Dim, Frame::Inertial>(
                                        num_points, 0.0)));
}

void test_ks(const Mesh<3>& mesh) {
  const auto gauge_condition = serialize_and_deserialize(
      TestHelpers::test_creation<std::unique_ptr<gh::gauges::GaugeCondition>,
                                 Metavariables<3>>(
          "AnalyticChristoffel:\n"
          "  AnalyticPrescription:\n"
          "    KerrSchild:\n"
          "      Mass: 1.2\n"
          "      Spin: [0.1, 0.2, 0.3]\n"
          "      Center: [-0.1, -0.2, -0.4]\n")
          ->get_clone());

  const size_t num_points = mesh.number_of_grid_points();

  const double time = 1.2;
  const auto coord_map = make_coord_map<3>();
  const auto logical_coords = logical_coordinates(mesh);
  const tnsr::I<DataVector, 3, Frame::Inertial> inertial_coords =
      coord_map(logical_coords);
  const auto inverse_jacobian = coord_map.inv_jacobian(logical_coords);

  tnsr::a<DataVector, 3, Frame::Inertial> gauge_h(num_points);
  tnsr::ab<DataVector, 3, Frame::Inertial> d4_gauge_h(num_points);
  // Used dispatch with defaulted arguments that we don't need for Analytic
  // gauge.
  gh::gauges::dispatch(make_not_null(&gauge_h), make_not_null(&d4_gauge_h), {},
                       {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, {}, mesh, time,
                       inertial_coords, inverse_jacobian, *gauge_condition);

  const gh::Solutions::WrappedGr<gr::Solutions::KerrSchild> kerr_schild{
      1.2, {0.1, 0.2, 0.3}, {-0.1, -0.2, -0.4}};
  const auto [pi, phi, spacetime_metric] = kerr_schild.variables(
      inertial_coords, time,
      tmpl::list<gh::Tags::Pi<DataVector, 3>, gh::Tags::Phi<DataVector, 3>,
                 gr::Tags::SpacetimeMetric<DataVector, 3>>{});
  const auto spatial_metric = gr::spatial_metric(spacetime_metric);
  const auto inverse_spatial_metric =
      determinant_and_inverse(spatial_metric).second;
  const auto shift = gr::shift(spacetime_metric, inverse_spatial_metric);
  const auto lapse = gr::lapse(shift, spacetime_metric);
  const auto spacetime_normal_one_form =
      gr::spacetime_normal_one_form<DataVector, 3, Frame::Inertial>(lapse);
  const auto spacetime_normal_vector =
      gr::spacetime_normal_vector(lapse, shift);
  const auto inverse_spacetime_metric =
      determinant_and_inverse(spacetime_metric).second;
  auto expected_gauge_h = gh::trace_christoffel(
      spacetime_normal_one_form, spacetime_normal_vector,
      inverse_spatial_metric, inverse_spacetime_metric, pi, phi);
  for (auto& t : expected_gauge_h) {
    t *= -1.0;
  }

  CHECK_ITERABLE_APPROX(gauge_h, expected_gauge_h);
  // Compute numerical spatial derivative
  tnsr::ab<DataVector, 3, Frame::Inertial> expected_d4_gauge_h{num_points};
  tnsr::ia<DataVector, 3, Frame::Inertial> expected_di_gauge_h{};
  for (size_t i = 0; i < 3; ++i) {
    for (size_t a = 0; a < 3 + 1; ++a) {
      expected_di_gauge_h.get(i, a).set_data_ref(
          make_not_null(&expected_d4_gauge_h.get(i + 1, a)));
    }
  }
  partial_derivative(make_not_null(&expected_di_gauge_h), expected_gauge_h,
                     mesh, inverse_jacobian);
  // Set time derivative to zero. We are assuming a static solution.
  for (size_t a = 0; a < 4; ++a) {
    expected_d4_gauge_h.get(0, a) = 0.0;
  }
  CHECK_ITERABLE_APPROX(d4_gauge_h, expected_d4_gauge_h);
}
}  // namespace

SPECTRE_TEST_CASE(
    "Unit.Evolution.Systems.GeneralizedHarmonic.Gauge.AnalyticChristoffel",
    "[Unit][Evolution]") {
  gh::gauges::register_derived_with_charm();
  for (const auto& basis_and_quadrature :
       {std::pair{Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto},
        {Spectral::Basis::FiniteDifference,
         Spectral::Quadrature::CellCentered}}) {
    test_gauge_wave<1>(
        {5, basis_and_quadrature.first, basis_and_quadrature.second});
    test_gauge_wave<2>(
        {5, basis_and_quadrature.first, basis_and_quadrature.second});
    test_gauge_wave<3>(
        {5, basis_and_quadrature.first, basis_and_quadrature.second});
    test_ks({5, basis_and_quadrature.first, basis_and_quadrature.second});
  }
}
