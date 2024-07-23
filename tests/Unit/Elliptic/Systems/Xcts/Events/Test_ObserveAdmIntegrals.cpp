// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Elliptic/Systems/Xcts/Events/ObserveAdmIntegrals.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"

namespace {

using KerrSchild = Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>;

void test_local_adm_integrals(const double distance,
                                  const size_t refinement_level,
                                  const size_t polynomial_order) {
  // Define black hole parameters
  const double mass = 1;
  const double boost_speed = 0.5;
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));

  // Get Kerr-Schild solution
  const std::array<double, 3> dimensionless_spin{{0., 0., 0.}};
  const std::array<double, 3> center{{0., 0., 0.}};
  const std::array<double, 3> boost_velocity{{0., 0., boost_speed}};
  const KerrSchild solution{mass, dimensionless_spin, center, boost_velocity};

  // Set up domain
  const double horizon_kerrschild_radius =
      mass * (1. + sqrt(1. - dot(dimensionless_spin, dimensionless_spin)));
  domain::creators::Sphere shell{
      /* inner_radius */ horizon_kerrschild_radius,
      /* outer_radius */ distance,
      /* interior */ domain::creators::Sphere::Excision{},
      refinement_level,
      polynomial_order + 1,
      true,
      {},
      {},
      domain::CoordinateMaps::Distribution::Logarithmic};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const Mesh<3> mesh{polynomial_order + 1, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const Mesh<2> face_mesh{polynomial_order + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize "reduced" integral
  tnsr::I<double, 3> reduced_integral;
  for (int I = 0; I < 3; I++) {
    reduced_integral.get(I) = 0.;
  }

  // Compute integral by summing over each element
  for (const auto& element_id : element_ids) {
    // Get element information
    const auto& current_block = blocks.at(element_id.block_id());
    const auto current_element = domain::Initialization::create_initial_element(
        element_id, current_block, initial_ref_levels);
    const ElementMap<3, Frame::Inertial> logical_to_inertial_map(
        element_id, current_block.stationary_map().get_clone());

    // Get coordinates
    const auto logical_coords = logical_coordinates(mesh);
    const auto inertial_coords = logical_to_inertial_map(logical_coords);
    const auto inv_jacobian =
        logical_to_inertial_map.inv_jacobian(logical_coords);

    // Get required fields
    const auto background_fields = solution.variables(
        inertial_coords,
        tmpl::list<
            Xcts::Tags::ConformalFactor<DataVector>,
            gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
            gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
            Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
            gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>,
            gr::Tags::TraceExtrinsicCurvature<DataVector>>{});
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(background_fields);
    const auto& spatial_metric =
        get<gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& inv_spatial_metric =
        get<gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& inv_conformal_metric =
        get<Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& extrinsic_curvature =
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(background_fields);

    // Compute face normal (related to the conformal metric)
    auto direction = Direction<3>::upper_zeta();
    auto face_normal =
        unnormalized_face_normal(face_mesh, logical_to_inertial_map, direction);
    const auto& face_inv_conformal_metric =
        dg::project_tensor_to_boundary(inv_conformal_metric, mesh, direction);
    const auto face_normal_magnitude =
        magnitude(face_normal, face_inv_conformal_metric);
    for (size_t d = 0; d < 3; ++d) {
      face_normal.get(d) /= get(face_normal_magnitude);
    }
    DirectionMap<3, tnsr::i<DataVector, 3>> faces_map_normal(
        {std::make_pair(direction, face_normal)});

    // Compute integral
    tnsr::I<double, 3> element_integral;
    Events::local_adm_integrals(
        make_not_null(&element_integral), conformal_factor, spatial_metric,
        inv_spatial_metric, extrinsic_curvature, trace_extrinsic_curvature,
        inv_jacobian, mesh, current_element, faces_map_normal);
    for (int I = 0; I < 3; I++) {
      reduced_integral.get(I) += element_integral.get(I);
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  CHECK(reduced_integral.get(0) == custom_approx(0.));
  CHECK(reduced_integral.get(1) == custom_approx(0.));
  CHECK(reduced_integral.get(2) ==
        custom_approx(lorentz_factor * mass * boost_speed));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.Xcts.ObserveAdmIntegrals",
                  "[Unit][Elliptic]") {
  const size_t P = 6;
  const size_t L = 1;
  for (double distance : std::array<double, 3>{{1.e3, 1.e5, 1.e10}}) {
    test_local_adm_integrals(distance, L, P);
  }
}
