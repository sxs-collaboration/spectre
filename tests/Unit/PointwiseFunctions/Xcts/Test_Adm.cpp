// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <iostream>
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
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrHorizon.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/KerrSchild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/Schwarzschild.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Xcts/WrappedGr.hpp"
#include "PointwiseFunctions/Xcts/Adm.hpp"

using KerrSchild = Xcts::Solutions::WrappedGr<gr::Solutions::KerrSchild>;
// using KerrSchild = Xcts::Solutions::Schwarzschild;
namespace {

void test_mass_integral(const double distance, const size_t refinements,
                        const size_t points) {
  // int distances[5] = {10,100,1000,10000,100000};
  // int refinements[5] = {0,1,2,3,4};
  // for (int distance : distances){

  //   for(int refinement : refinements){
  const double mass = 0.45;
  const std::array<double, 3> dimensionless_spin{{0., 0., 0.}};
  const double horizon_kerrschild_radius =
      mass * (1. + sqrt(1. - dot(dimensionless_spin, dimensionless_spin)));
  const KerrSchild solution{mass, dimensionless_spin, {{0., 0., 0.}}};
  domain::creators::Sphere shell{
      horizon_kerrschild_radius,
      distance * horizon_kerrschild_radius,
      domain::creators::Sphere::Excision{},
      refinements,
      points,
      true,
      {},
      {},
      domain::CoordinateMaps::Distribution::Logarithmic};
  const auto shell_domain = shell.create_domain();
  const auto& blocks = shell_domain.blocks();
  const auto& initial_ref_levels = shell.initial_refinement_levels();
  const auto element_ids = initial_element_ids(initial_ref_levels);
  const auto excision_sphere =
      shell_domain.excision_spheres().at("ExcisionSphere");
  const Mesh<2> face_mesh{points, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
  const Mesh<3> mesh{points, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  // const auto kerr_conforming_map =
  //     domain::CoordinateMap<Frame::Grid, Frame::Inertial,
  //                     domain::CoordinateMaps::KerrHorizonConforming>(
  //     domain::CoordinateMaps::KerrHorizonConforming{mass,
  //                                                 dimensionless_spin});
  double mass_integral_surface = 0.;
  double mass_integral_volume = 0.;
  for (const auto& element_id : element_ids) {
    const auto& current_block = blocks.at(element_id.block_id());
    const auto logical_coords = logical_coordinates(mesh);
    const ElementMap<3, Frame::Inertial> logical_to_grid_map(
        element_id, current_block.stationary_map().get_clone());
    const tnsr::I<DataVector, 3> inertial_coords =
        logical_to_grid_map(logical_coords);
    const auto inv_jacobian = logical_to_grid_map.inv_jacobian(logical_coords);
    const auto det_jacobian = determinant(inv_jacobian);
    const auto background_fields = solution.variables(
        inertial_coords, mesh, inv_jacobian,
        tmpl::list<
            Xcts::Tags::ConformalFactor<DataVector>,
            Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>,
            Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>,
            Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                       Frame::Inertial>,
            Xcts::Tags::ConformalRicciScalar<DataVector>,
            gr::Tags::ExtrinsicCurvature<DataVector, 3>,
            gr::Tags::TraceExtrinsicCurvature<DataVector>,
            gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>>{});
    const auto& conformal_factor =
        get<Xcts::Tags::ConformalFactor<DataVector>>(background_fields);
    const auto& conformal_metric =
        get<Xcts::Tags::ConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& inv_conformal_metric =
        get<Xcts::Tags::InverseConformalMetric<DataVector, 3, Frame::Inertial>>(
            background_fields);
    const auto& conformal_christoffel_second_kind =
        get<Xcts::Tags::ConformalChristoffelSecondKind<DataVector, 3,
                                                       Frame::Inertial>>(
            background_fields);
    const auto& conformal_ricci_scalar =
        get<Xcts::Tags::ConformalRicciScalar<DataVector>>(background_fields);
    const auto& extrinsic_curvature =
        get<gr::Tags::ExtrinsicCurvature<DataVector, 3>>(background_fields);
    const auto& trace_extrinsic_curvature =
        get<gr::Tags::TraceExtrinsicCurvature<DataVector>>(background_fields);
    const auto& energy_density =
        get<gr::Tags::Conformal<gr::Tags::EnergyDensity<DataVector>, 0>>(
            background_fields);
    const auto vol_integrand = Xcts::adm_mass_volume_integrand(
        conformal_factor, conformal_metric, inv_conformal_metric,
        conformal_christoffel_second_kind, conformal_ricci_scalar,
        extrinsic_curvature, trace_extrinsic_curvature, energy_density, mesh,
        inv_jacobian);
    mass_integral_volume +=
        definite_integral(get(vol_integrand) / get(det_jacobian), mesh);
    const auto element_abutting_direction =
        excision_sphere.abutting_direction(element_id);
    if (element_abutting_direction.has_value()) {
      const auto face_logical_coords = interface_logical_coordinates(
          face_mesh, element_abutting_direction.value());
      const domain::CoordinateMaps::Composition logical_to_inertial_map{
          domain::element_to_block_logical_map(element_id),
          current_block.stationary_map().get_clone()};
      const auto face_inverse_jacobian =
          logical_to_inertial_map.inv_jacobian(face_logical_coords);
      const auto direction = Direction<3>::lower_zeta();
      const size_t slice_index = index_to_slice_at(mesh.extents(), direction);
      const auto face_background_fields = solution.variables(
          logical_to_inertial_map(face_logical_coords),
          tmpl::list<gr::Tags::InverseSpatialMetric<DataVector, 3>>{});
      const auto& face_inv_conformal_metric =
          data_on_slice(inv_conformal_metric, mesh.extents(),
                        direction.dimension(), slice_index);
      const auto& face_conformal_christoffel_second_kind =
          data_on_slice(conformal_christoffel_second_kind, mesh.extents(),
                        direction.dimension(), slice_index);
      ;
      const auto& face_inv_spatial_metric =
          get<gr::Tags::InverseSpatialMetric<DataVector, 3>>(
              face_background_fields);
      const auto face_sqrt_det_conformal_metric = Scalar<DataVector>(
          1 / sqrt(get(determinant(face_inv_conformal_metric))));
      const auto curved_area_element = area_element(
          face_inverse_jacobian, element_abutting_direction.value(),
          face_inv_conformal_metric, face_sqrt_det_conformal_metric);
      const auto deriv = data_on_slice(
          partial_derivative(conformal_factor, mesh, inv_jacobian),
          mesh.extents(), direction.dimension(), slice_index);
      const auto conformal_factor_deriv = tenex::evaluate<ti::I>(
          face_inv_conformal_metric(ti::J, ti::I) * deriv(ti::j));
      // const auto christoffel =
      // -divergence(inv_conformal_metric,mesh,inv_jacobian);
      const tnsr::I<DataVector, 3> integrand = Xcts::adm_mass_surface_integrand(
          conformal_factor_deriv, face_inv_conformal_metric,
          face_conformal_christoffel_second_kind);
      auto face_normal = unnormalized_face_normal(
          face_mesh, logical_to_inertial_map, direction);
      const auto face_normal_magnitude =
          magnitude(face_normal, face_inv_conformal_metric);
      for (size_t d = 0; d < 3; ++d) {
        face_normal.get(d) /= get(face_normal_magnitude);
      }
      const auto contracted =
          tenex::evaluate(integrand(ti::I) * face_normal(ti::i));
      mass_integral_surface += definite_integral(
          -get(contracted) * get(curved_area_element), face_mesh);
    }
  }
  auto custom_approx = Approx::custom().epsilon(1e-2);
  std::cout << "Volume Integral:" << mass_integral_volume << "\n";
  std::cout << "Surface Integral:" << mass_integral_surface << "\n";
  std::cout << "Total Integral:"
            << mass_integral_volume + mass_integral_surface;
  CHECK(mass_integral_volume + mass_integral_surface == custom_approx(0.45));
}
//  }
//}
}  // namespace

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.Xcts.Adm",
                  "[Unit][PointwiseFunctions]") {
  pypp::SetupLocalPythonEnvironment local_python_env{"PointwiseFunctions/Xcts"};
  const DataVector used_for_size{5};
  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          gsl::not_null<Scalar<DataVector>*>, const Scalar<DataVector>&,
          const tnsr::ii<DataVector, 3>&, const tnsr::II<DataVector, 3>&,
          const Scalar<DataVector>&, const tnsr::ii<DataVector, 3>&,
          const Scalar<DataVector>&, const Scalar<DataVector>&,
          const Scalar<DataVector>&)>(&Xcts::adm_mass_volume_integrand),
      "Adm", {"adm_mass_volume_integrand"}, {{{-1., 1.}}}, used_for_size);
  pypp::check_with_random_values<1>(
      static_cast<void (*)(
          gsl::not_null<tnsr::I<DataVector, 3>*>, const tnsr::I<DataVector, 3>&,
          const tnsr::II<DataVector, 3>&, const tnsr::Ijj<DataVector, 3>&)>(
          &Xcts::adm_mass_surface_integrand),
      "Adm", {"adm_mass_surface_integrand"}, {{{-1., 1.}}}, used_for_size);
  const double distance = 2e6;
  const size_t refinement = 2;
  for (size_t point : std::array<size_t, 6>{{2, 4, 6, 8, 10, 12}}) {
    test_mass_integral(distance, refinement, point);
  }
}
