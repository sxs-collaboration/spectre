// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Wedge.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Elliptic/DiscontinuousGalerkin/DgOperator.hpp"
#include "Elliptic/Systems/Poisson/FirstOrderSystem.hpp"
#include "Elliptic/Systems/Poisson/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ApplyMassMatrix.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MetricIdentityJacobian.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/NormalDotFlux.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "NumericalAlgorithms/LinearOperators/WeakDivergence.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Poisson/Lorentzian.hpp"
#include "Utilities/TMPL.hpp"

SPECTRE_TEST_CASE("Unit.Elliptic.DG.LargeOuterRadius", "[Unit][Elliptic]") {
  // This test works for the StrongLogical and WeakInertial formulations, but
  // breaks for the StrongInertial formulation.
  const ::dg::Formulation formulation = ::dg::Formulation::StrongLogical;
  // The test also breaks if the constant is set to 1. _and_ the numeric first
  // derivative is used. It works fine if the analytic first derivative is used
  // or if the constant is set to 0.
  const Poisson::Solutions::Lorentzian<3> solution{/* plus_constant */ 1.};
  const bool use_numeric_first_deriv = false;
  // Set up the grid
  using Wedge3D = domain::CoordinateMaps::Wedge<3>;
  const double inner_radius = 1e2;
  CAPTURE(inner_radius);
  const double outer_radius = 1e9;
  CAPTURE(outer_radius);
  auto block_map =
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          Wedge3D{inner_radius, outer_radius, 1., 1.,
                  OrientationMap<3>::create_aligned(), true,
                  Wedge3D::WedgeHalves::Both,
                  domain::CoordinateMaps::Distribution::Inverse});
  const ElementId<3> element_id{0, {{{2, 0}, {2, 0}, {0, 0}}}};
  CAPTURE(element_id);
  const Mesh<3> mesh{12, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  CAPTURE(mesh);

  // Quantify grid stretching by looking at the midpoint of the element
  const ElementMap<3, Frame::Inertial> element_map{element_id,
                                                   std::move(block_map)};
  const auto midpoint_radius = magnitude(
      element_map(tnsr::I<double, 3, Frame::ElementLogical>{{{0., 0., 0.}}}));
  CAPTURE(midpoint_radius);
  Approx custom_approx = Approx::custom().epsilon(1.0e-6).scale(1.0);
  const size_t num_points = mesh.number_of_grid_points();
  const auto logical_coords = logical_coordinates(mesh);
  const auto inertial_coords = element_map(logical_coords);
  const auto inv_jacobian = element_map.inv_jacobian(logical_coords);
  const auto det_jacobian = determinant(element_map.jacobian(logical_coords));

  // Take first derivative
  using Field = Poisson::Tags::Field<DataVector>;
  using FieldDeriv = ::Tags::deriv<Poisson::Tags::Field<DataVector>,
                                   tmpl::size_t<3>, Frame::Inertial>;
  using Flux = ::Tags::Flux<Poisson::Tags::Field<DataVector>, tmpl::size_t<3>,
                            Frame::Inertial>;
  using FixedSource = ::Tags::FixedSource<Field>;
  const auto vars = solution.variables(
      inertial_coords, tmpl::list<Field, FieldDeriv, FixedSource>{});
  const auto& u = get<Field>(vars);
  const auto& du_analytic = get<FieldDeriv>(vars);
  const auto& f = get<FixedSource>(vars);
  const auto du_numeric = partial_derivative(u, mesh, inv_jacobian);
  CHECK_ITERABLE_CUSTOM_APPROX(du_numeric, du_analytic, custom_approx);

  // Take second derivative
  CAPTURE(formulation);
  const bool massive = true;
  CAPTURE(massive);
  tnsr::I<DataVector, 3> flux{num_points};
  Poisson::flat_cartesian_fluxes(
      make_not_null(&flux), use_numeric_first_deriv ? du_numeric : du_analytic);
  if (formulation == ::dg::Formulation::StrongInertial) {
    Scalar<DataVector> ddu{num_points, 0.};
    divergence(make_not_null(&ddu), flux, mesh, inv_jacobian);
    DataVector lhs = -get(ddu);
    DataVector rhs = get(f);
    // Massive scheme multiplies by Jacobian determinant
    if (massive) {
      lhs *= get(det_jacobian);
      rhs *= get(det_jacobian);
    }
    CHECK_ITERABLE_CUSTOM_APPROX(lhs, rhs, custom_approx);
  } else if (formulation == ::dg::Formulation::StrongLogical) {
    Scalar<DataVector> ddu{num_points, 0.};
    tnsr::I<DataVector, 3, Frame::ElementLogical> logical_flux =
        transform::first_index_to_different_frame(flux, inv_jacobian);
    for (size_t d = 0; d < 3; ++d) {
      logical_flux.get(d) *= get(det_jacobian);
    }
    logical_divergence(make_not_null(&ddu), logical_flux, mesh);
    DataVector lhs = -get(ddu);
    DataVector rhs = get(f);
    // Massive scheme multiplies by Jacobian determinant
    if (massive) {
      ::dg::apply_mass_matrix(make_not_null(&lhs), mesh);
      ::dg::apply_mass_matrix(make_not_null(&rhs), mesh);
      rhs *= get(det_jacobian);
    } else {
      lhs /= get(det_jacobian);
    }
    CHECK_ITERABLE_CUSTOM_APPROX(lhs, rhs, custom_approx);
  } else {
    auto det_times_inv_jacobian = inv_jacobian;
    for (auto& component : det_times_inv_jacobian) {
      component *= get(det_jacobian);
    }
    Variables<tmpl::list<Flux>> fluxes{mesh.number_of_grid_points()};
    get<Flux>(fluxes) = flux;
    Variables<tmpl::list<Field>> lhs{mesh.number_of_grid_points()};
    weak_divergence(make_not_null(&lhs), fluxes, mesh, det_times_inv_jacobian);
    DataVector rhs = get(f);
    if (massive) {
      ::dg::apply_mass_matrix(make_not_null(&lhs), mesh);
      ::dg::apply_mass_matrix(make_not_null(&rhs), mesh);
      rhs *= get(det_jacobian);
    } else {
      lhs /= get(det_jacobian);
    }
    // Add boundary corrections
    for (const auto direction : Direction<3>::all_directions()) {
      CAPTURE(direction);
      // Compute face geometry
      const auto face_mesh = mesh.slice_away(direction.dimension());
      const size_t face_num_points = face_mesh.number_of_grid_points();
      const auto face_logical_coords =
          interface_logical_coordinates(face_mesh, direction);
      const auto face_inertial_coords = element_map(face_logical_coords);
      auto face_normal =
          unnormalized_face_normal(face_mesh, element_map, direction);
      const auto face_normal_magnitude = magnitude(face_normal);
      for (size_t d = 0; d < 3; ++d) {
        face_normal.get(d) /= get(face_normal_magnitude);
      }
      auto face_jacobian = face_normal_magnitude;
      const auto det_jacobian_face =
          determinant(element_map.jacobian(face_logical_coords));
      get(face_jacobian) *= get(det_jacobian_face);
      // Compute boundary correction
      const auto vars_on_face =
          solution.variables(face_inertial_coords, tmpl::list<FieldDeriv>{});
      const auto& du_on_face = get<FieldDeriv>(vars_on_face);
      Variables<tmpl::list<Flux>> fluxes_on_face{face_num_points};
      Poisson::flat_cartesian_fluxes(make_not_null(&get<Flux>(fluxes_on_face)),
                                     du_on_face);
      auto boundary_correction =
          normal_dot_flux<tmpl::list<Field>>(face_normal, fluxes_on_face);
      ASSERT(massive, "Only massive scheme supported here");
      ::dg::apply_mass_matrix(make_not_null(&boundary_correction), face_mesh);
      boundary_correction *= get(face_jacobian);
      boundary_correction *= -1.;
      if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
        add_slice_to_data(make_not_null(&lhs), boundary_correction,
                          mesh.extents(), direction.dimension(),
                          index_to_slice_at(mesh.extents(), direction));
      } else {
        ::dg::lift_boundary_terms_gauss_points(
            make_not_null(&lhs), boundary_correction, mesh, direction);
      }
    }
    CHECK_ITERABLE_CUSTOM_APPROX(get(get<Field>(lhs)), rhs, custom_approx);
  }
}
