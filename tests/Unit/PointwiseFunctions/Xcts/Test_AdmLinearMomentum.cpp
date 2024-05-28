// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Slice.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/AreaElement.hpp"
#include "Domain/CoordinateMaps/Composition.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/KerrHorizonConforming.hpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/CreateInitialElement.hpp"
#include "Domain/Creators/Sphere.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/ElementToBlockLogicalMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/IndexToSliceAt.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Framework/CheckWithRandomValues.hpp"
#include "Framework/SetupLocalPythonEnvironment.hpp"
#include "NumericalAlgorithms/LinearOperators/DefiniteIntegral.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/HarmonicSchwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/Xcts/AdmLinearMomentum.hpp"

namespace {

using KerrSchild = Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>;

void test_linear_momentum_surface_integral(const double distance,
                                           const size_t refinement_level,
                                           const size_t polynomial_order) {
  // Define black hole parameters
  const double mass = 1;
  const double boost_speed = 0.5;
  const double lorentz_factor = 1. / sqrt(1. - square(boost_speed));

  // Get Kerr-Schild solution
  const std::array<double, 3> dimensionless_spin{{0., 0., 0.}};
  const std::array<double, 3> boost_velocity{{0., 0., boost_speed}};
  const KerrSchild solution{mass, dimensionless_spin,
                            /* center */ std::array<double, 3>{{0., 0., 0.}},
                            boost_velocity};

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
  const Mesh<2> face_mesh{polynomial_order + 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};

  // Initialize surface integral
  tnsr::I<double, 3> surface_integral;
  for (int I = 0; I < 3; I++) {
    surface_integral.get(I) = 0.;
  }

  // Compute integrals by summing over each element
  for (const auto& element_id : element_ids) {
    // Skip elements not at the outer boundary
    const size_t radial_dimension = 2;
    const auto radial_segment_id = element_id.segment_id(radial_dimension);
    if (radial_segment_id.index() !=
        two_to_the(radial_segment_id.refinement_level()) - 1) {
      continue;
    }

    // Get element information
    const auto& current_block = blocks.at(element_id.block_id());
    const auto current_element = domain::Initialization::create_initial_element(
        element_id, current_block, initial_ref_levels);
    const ElementMap<3, Frame::Inertial> logical_to_inertial_map(
        element_id, current_block.stationary_map().get_clone());

    // Loop over external boundaries
    for (auto boundary_direction : current_element.external_boundaries()) {
      // Skip interfaces not at the outer boundary
      if (boundary_direction != Direction<3>::upper_zeta()) {
        continue;
      }

      // Get coordinates
      const auto logical_coords =
          interface_logical_coordinates(face_mesh, boundary_direction);
      const auto inertial_coords = logical_to_inertial_map(logical_coords);

      // Get required fields
      const auto background_fields = solution.variables(
          inertial_coords,
          tmpl::list<
              Xcts::Tags::ConformalFactor<DataVector>,
              gr::Tags::SpatialMetric<DataVector, 3, Frame::Inertial>,
              gr::Tags::InverseSpatialMetric<DataVector, 3, Frame::Inertial>,
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
      const auto& extrinsic_curvature =
          get<gr::Tags::ExtrinsicCurvature<DataVector, 3, Frame::Inertial>>(
              background_fields);
      const auto& trace_extrinsic_curvature =
          get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(background_fields);

      // Compute the inverse extrinsic curvature
      tnsr::II<DataVector, 3> inv_extrinsic_curvature;
      tenex::evaluate<ti::I, ti::J>(make_not_null(&inv_extrinsic_curvature),
                                    inv_spatial_metric(ti::I, ti::K) *
                                        inv_spatial_metric(ti::J, ti::L) *
                                        extrinsic_curvature(ti::k, ti::l));

      // Compute curved area element
      const auto inv_jacobian =
          logical_to_inertial_map.inv_jacobian(logical_coords);
      const auto sqrt_det_spatial_metric =
          Scalar<DataVector>(sqrt(get(determinant(spatial_metric))));
      const auto curved_area_element =
          area_element(inv_jacobian, boundary_direction, inv_spatial_metric,
                       sqrt_det_spatial_metric);

      // Compute face normal
      auto face_normal = unnormalized_face_normal(
          face_mesh, logical_to_inertial_map, boundary_direction);
      const auto face_normal_magnitude =
          magnitude(face_normal, inv_spatial_metric);
      for (size_t d = 0; d < 3; ++d) {
        face_normal.get(d) /= get(face_normal_magnitude);
      }

      // Compute surface integrand
      const auto surface_integrand =
          Xcts::adm_linear_momentum_surface_integrand(
              conformal_factor, inv_spatial_metric, inv_extrinsic_curvature,
              trace_extrinsic_curvature);
      const auto contracted_integrand = tenex::evaluate<ti::I>(
          surface_integrand(ti::I, ti::J) * face_normal(ti::j));

      // Compute contribution to surface integral
      for (int I = 0; I < 3; I++) {
        surface_integral.get(I) += definite_integral(
            contracted_integrand.get(I) * get(curved_area_element), face_mesh);
      }
    }
  }

  // Check result
  auto custom_approx = Approx::custom().epsilon(10. / distance).scale(1.0);
  CHECK(surface_integral.get(0) == custom_approx(0.));
  CHECK(surface_integral.get(1) == custom_approx(0.));
  CHECK(surface_integral.get(2) ==
        custom_approx(lorentz_factor * mass * boost_speed));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.AdmLinearMomentum",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment
  local_python_env{"PointwiseFunctions/Xcts"}; const DataVector
  used_for_size{5};

  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          gsl::not_null<tnsr::II<DataVector, 3>*>, const Scalar<DataVector>&,
          const tnsr::II<DataVector, 3>&, const tnsr::II<DataVector, 3>&,
          const Scalar<DataVector>&)>(
          &Xcts::adm_linear_momentum_surface_integrand),
      "AdmLinearMomentum", {"adm_linear_momentum_surface_integrand"},
      {{{-1., 1.}}}, used_for_size);

  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          gsl::not_null<tnsr::I<DataVector, 3>*>,
          const tnsr::II<DataVector, 3>&, const Scalar<DataVector>&,
          const tnsr::i<DataVector, 3>&, const tnsr::ii<DataVector, 3>&,
          const tnsr::II<DataVector, 3>&, const tnsr::Ijj<DataVector, 3>&,
          const tnsr::i<DataVector, 3>&)>(
          &Xcts::adm_linear_momentum_volume_integrand),
      "AdmLinearMomentum", {"adm_linear_momentum_volume_integrand"},
      {{{-1, 1.}}}, used_for_size);

  const size_t P = 6;
  const size_t L = 1;
  for (double distance : std::array<double, 3>{{1.e3, 1.e5, 1.e10}}) {
    test_linear_momentum_surface_integral(distance, L, P);
  }
}
