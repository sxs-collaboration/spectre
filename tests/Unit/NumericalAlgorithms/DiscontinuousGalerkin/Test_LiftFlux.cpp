// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <memory>
#include <pup.h>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/Direction.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TypeTraits.hpp"

#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/SurfaceJacobian.hpp"
#include "Framework/TestHelpers.hpp"

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.LiftFlux",
                  "[Unit][NumericalAlgorithms]") {
  {
    const Mesh<2> mesh{{{3, 5}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

    using Affine = domain::CoordinateMaps::Affine;
    const Affine xi_map(-1., 1., -5., 7.);
    const Affine eta_map(-1., 1., 2., 5.);
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
            domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>(xi_map,
                                                                   eta_map));
    const double element_length = (eta_map(std::array<double, 1>{{1.}}) -
                                   eta_map(std::array<double, 1>{{-1.}}))[0];

    const double weight =
        Spectral::quadrature_weights(mesh.slice_through(1))[0];
    const auto boundary_mesh = mesh.slice_through(0);

    const auto magnitude_of_face_normal = magnitude(unnormalized_face_normal(
        boundary_mesh, coordinate_map, Direction<2>::lower_eta()));

    Variables<tmpl::list<Tags::NormalDotNumericalFlux<Var>>> flux(
        boundary_mesh.number_of_grid_points());
    get(get<Tags::NormalDotNumericalFlux<Var>>(flux)) =
        DataVector({2., 3., 5.});

    const Variables<tmpl::list<Var>> expected =
        -2. / (element_length * weight) * flux;

    CHECK(dg::lift_flux(flux, mesh.extents(1), magnitude_of_face_normal) ==
          expected);
  }
  {
    const Mesh<1> mesh{
        {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

    using Affine = domain::CoordinateMaps::Affine;
    const Affine xi_map(-1., 1., 0., M_PI_2);
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(xi_map);

    const auto boundary_mesh = mesh.slice_through();

    const auto magnitude_of_face_normal = magnitude(unnormalized_face_normal(
        boundary_mesh, coordinate_map, Direction<1>::upper_xi()));
    CAPTURE(magnitude_of_face_normal);

    Variables<tmpl::list<Tags::NormalDotNumericalFlux<Var>>> flux(
        boundary_mesh.number_of_grid_points());
    get(get<Tags::NormalDotNumericalFlux<Var>>(flux)) = DataVector({1.});

    Variables<tmpl::list<Var>> expected_lifted_flux(
        mesh.number_of_grid_points());
    get(get<Var>(expected_lifted_flux)) =
        DataVector({2.54647909, -1.13882007, 1.13882007, -10.18591636});
    Variables<tmpl::list<Var>> expected_lifted_flux_mass_lumping(
        boundary_mesh.number_of_grid_points());
    get(get<Var>(expected_lifted_flux_mass_lumping)) =
        DataVector({-7.63943727});

    Approx custom_approx = Approx::custom().epsilon(1e-5).scale(1.);
    CHECK_VARIABLES_CUSTOM_APPROX(
        dg::lift_flux_no_mass_lumping(flux, mesh, Direction<1>::upper_xi(),
                                      magnitude_of_face_normal),
        expected_lifted_flux, custom_approx);
    CHECK_VARIABLES_CUSTOM_APPROX(
        dg::lift_flux(flux, mesh.extents(0), magnitude_of_face_normal),
        expected_lifted_flux_mass_lumping, custom_approx);
  }
  {
    const Mesh<1> mesh{
        {{4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

    using Affine = domain::CoordinateMaps::Affine;
    const Affine xi_map(-1., 1., -4., 0.);
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(xi_map);
    const auto direction = Direction<1>::upper_xi();

    const auto boundary_mesh = mesh.slice_through();

    const auto surface_jacobian = Scalar<DataVector>(
        get(determinant(coordinate_map.jacobian(
            interface_logical_coordinates(boundary_mesh, direction)))) /
        get(magnitude(unnormalized_face_normal(boundary_mesh, coordinate_map,
                                               direction))));
    CAPTURE(surface_jacobian);

    Variables<tmpl::list<Tags::NormalDotNumericalFlux<Var>>> flux(
        boundary_mesh.number_of_grid_points());
    get(get<Tags::NormalDotNumericalFlux<Var>>(flux)) = DataVector({1.});

    Variables<tmpl::list<Var>> expected_lifted_flux(
        boundary_mesh.number_of_grid_points());
    get(get<Var>(expected_lifted_flux)) = DataVector({-1.});

    CHECK_VARIABLES_APPROX(dg::lift_flux_massive_no_mass_lumping(
                               flux, boundary_mesh, surface_jacobian),
                           expected_lifted_flux);
  }
  {
    const Mesh<2> mesh{{{3, 3}},
                       Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};

    using Affine = domain::CoordinateMaps::Affine;
    const Affine xi_map(-1., 1., -4., 0.);
    const Affine eta_map(-1., 1., -3., 0.);
    const auto coordinate_map =
        domain::make_coordinate_map<Frame::Logical, Frame::Grid>(
            domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>(xi_map,
                                                                   eta_map));
    {
      const auto direction = Direction<2>::upper_xi();
      CAPTURE(direction);

      const auto boundary_mesh = mesh.slice_through(direction.dimension());

      const auto face_normal_magnitude = magnitude(
          unnormalized_face_normal(boundary_mesh, coordinate_map, direction));
      const auto jac_det_on_face = determinant(coordinate_map.jacobian(
          interface_logical_coordinates(boundary_mesh, direction)));
      const auto surface_jacobian =
          domain::surface_jacobian(jac_det_on_face, face_normal_magnitude);
      CAPTURE(face_normal_magnitude);
      CAPTURE(jac_det_on_face);
      CAPTURE(surface_jacobian);
      CHECK(false);

      Variables<tmpl::list<Tags::NormalDotNumericalFlux<Var>>> flux(
          boundary_mesh.number_of_grid_points());
      get(get<Tags::NormalDotNumericalFlux<Var>>(flux)) =
          DataVector({1., 3., 9.});

      Variables<tmpl::list<Var>> expected_lifted_flux(
          boundary_mesh.number_of_grid_points());
      get(get<Var>(expected_lifted_flux)) = DataVector({-0.1, -6.8, -4.1});

      CHECK_VARIABLES_APPROX(dg::lift_flux_massive_no_mass_lumping(
                                 flux, boundary_mesh, surface_jacobian),
                             expected_lifted_flux);
    }
  }
}
