// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Determinant.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Wedge2D.hpp"
#include "Domain/CoordinateMaps/Wedge3D.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Evolution/DgSubcell/Reconstruction.hpp"
#include "Helpers/Evolution/DgSubcell/ProjectionTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/Overloader.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags {
template <typename Tag>
struct Prefix : db::PrefixTag {
  using tag = Tag;
  using type = typename Tag::type;
};

struct Scalar : db::SimpleTag {
  using type = ::Scalar<DataVector>;
};
template <size_t Dim>
struct Vector : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim>;
};
}  // namespace Tags

template <size_t Dim>
auto make_map() {
  if constexpr (Dim == 1) {
    using domain::make_coordinate_map_base;
    using domain::CoordinateMaps::Affine;
    return make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
        Affine(-1.0, 1.0, 2.0, 5.0));
  } else if constexpr (Dim == 2) {
    using domain::make_coordinate_map_base;
    using domain::CoordinateMaps::Affine;
    using domain::CoordinateMaps::ProductOf2Maps;
    using domain::CoordinateMaps::Wedge2D;
    return make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
        ProductOf2Maps<Affine, Affine>(Affine(-1.0, 1.0, -1.0, -0.8),
                                       Affine(-1.0, 1.0, -1.0, -0.8)),
        Wedge2D(0.5, 0.75, 1.0, 1.0, {}, false));
  } else {
    using domain::make_coordinate_map_base;
    using domain::CoordinateMaps::Affine;
    using domain::CoordinateMaps::ProductOf3Maps;
    using domain::CoordinateMaps::Wedge3D;
    return make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
        ProductOf3Maps<Affine, Affine, Affine>(Affine(-1.0, 1.0, -1.0, -0.8),
                                               Affine(-1.0, 1.0, -1.0, -0.8),
                                               Affine(-1.0, 1.0, 0.8, 1.0)),
        Wedge3D(0.5, 0.75, {}, 1.0, 1.0, false));
  }
}

template <size_t MaxPts, size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void test_reconstruct_fd(const std::vector<double>& eps) noexcept {
  CAPTURE(Dim);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);

  // For FD reconstruction we need to set up a coordinate map since we need
  // Jacobians to do the integration.
  const auto coord_map_base_ptr = make_map<Dim>();
  const auto& coord_map = *coord_map_base_ptr;

  const size_t start_pt =
      std::max(static_cast<size_t>(2),
               Spectral::minimum_number_of_points<BasisType, QuadratureType>);
  if (eps.size() != 1 and (eps.size() != MaxPts + 1 - start_pt)) {
    ERROR("eps is the wrong size. Must be size 1 or size "
          << MaxPts + 1 - start_pt << " but is " << eps.size());
  }
  for (size_t num_pts_1d = start_pt; num_pts_1d < MaxPts + 1; ++num_pts_1d) {
    const double local_eps =
        eps.size() == 1 ? eps[0] : eps[num_pts_1d - start_pt];
    Approx local_approx = Approx::custom().epsilon(local_eps).scale(1.);
    CAPTURE(local_eps);
    CAPTURE(num_pts_1d);
    const Mesh<Dim> dg_mesh{num_pts_1d, BasisType, QuadratureType};
    const auto inertial_coords = coord_map(logical_coordinates(dg_mesh));
    const size_t num_subcells_1d = 2 * num_pts_1d - 1;
    const Mesh<Dim> subcell_mesh(num_subcells_1d,
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered);

    const auto subcell_logical_coords = logical_coordinates(subcell_mesh);
    const auto subcell_inertial_coords = coord_map(subcell_logical_coords);

    const auto dg_jac = coord_map.jacobian(logical_coordinates(dg_mesh));
    const DataVector dg_det_jac = get(determinant(dg_jac));
    const DataVector projected_det_jac = evolution::dg::subcell::fd::project(
        dg_det_jac, dg_mesh, subcell_mesh.extents());

    // Our FD reconstruction scheme can integrate polynomials up to degree 5
    // exactly. However, we want to verify that if we have a higher degree
    // polynomial that we can still reconstruct the solution exactly. That is,
    // we want to verify that R(P(uJ))=uJ.
    const DataVector expected_nodal_coeffs =
        TestHelpers::evolution::dg::subcell::cell_values(
            std::min(dg_mesh.extents(0) - 2, 6_st), inertial_coords);

    const DataVector subcell_values =
        TestHelpers::evolution::dg::subcell::cell_values(
            std::min(dg_mesh.extents(0) - 2, 6_st), subcell_inertial_coords);

    const DataVector reconstructed_datavector =
        evolution::dg::subcell::fd::reconstruct(
            subcell_values * projected_det_jac, dg_mesh,
            subcell_mesh.extents()) /
        dg_det_jac;

    // Test reconstruction of a DataVector
    CHECK_ITERABLE_CUSTOM_APPROX(reconstructed_datavector,
                                 expected_nodal_coeffs, local_approx);

    // Test reconstruction of a Variables
    Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>> cell_values_vars(
        subcell_mesh.number_of_grid_points());
    get(get<Tags::Scalar>(cell_values_vars)) =
        TestHelpers::evolution::dg::subcell::cell_values(
            std::min(dg_mesh.extents(0) - 2, 6_st), subcell_inertial_coords) *
        projected_det_jac;
    for (size_t d = 0; d < Dim; ++d) {
      get<Tags::Vector<Dim>>(cell_values_vars).get(d) =
          (d + 2.0) *
          TestHelpers::evolution::dg::subcell::cell_values(
              std::min(dg_mesh.extents(0) - 2, 6_st), subcell_inertial_coords) *
          projected_det_jac;
    }

    const auto check_each_field_in_vars = [&dg_det_jac, &expected_nodal_coeffs,
                                           &local_approx](
                                              const auto& local_dg_vars) {
      REQUIRE(local_dg_vars.number_of_grid_points() ==
              expected_nodal_coeffs.size());
      CHECK_ITERABLE_CUSTOM_APPROX(
          DataVector(get(get<tmpl::front<typename std::decay_t<decltype(
                             local_dg_vars)>::tags_list>>(local_dg_vars)) /
                     dg_det_jac),
          expected_nodal_coeffs, local_approx);
      for (size_t d = 0; d < Dim; ++d) {
        CHECK_ITERABLE_CUSTOM_APPROX(
            DataVector(get<tmpl::back<typename std::decay_t<decltype(
                           local_dg_vars)>::tags_list>>(local_dg_vars)
                           .get(d) /
                       dg_det_jac),
            (d + 2.0) * expected_nodal_coeffs, local_approx);
      }
    };

    Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>>
        reconstructed_vars = evolution::dg::subcell::fd::reconstruct(
            cell_values_vars, dg_mesh, subcell_mesh.extents());
    check_each_field_in_vars(reconstructed_vars);

    // Check not_null with no prefix tags that we correctly resize the buffer.
    reconstructed_vars.initialize(0);
    evolution::dg::subcell::fd::reconstruct(make_not_null(&reconstructed_vars),
                                            cell_values_vars, dg_mesh,
                                            subcell_mesh.extents());
    check_each_field_in_vars(reconstructed_vars);

    // Check with the prefix on the subcell vars
    Variables<
        tmpl::list<Tags::Prefix<Tags::Scalar>, Tags::Prefix<Tags::Vector<Dim>>>>
        prefixed_cell_values_vars(subcell_mesh.number_of_grid_points());
    prefixed_cell_values_vars = cell_values_vars;
    evolution::dg::subcell::fd::reconstruct(make_not_null(&reconstructed_vars),
                                            prefixed_cell_values_vars, dg_mesh,
                                            subcell_mesh.extents());
    check_each_field_in_vars(reconstructed_vars);

    // Check with prefix tag on the DG vars
    Variables<
        tmpl::list<Tags::Prefix<Tags::Scalar>, Tags::Prefix<Tags::Vector<Dim>>>>
        prefixed_reconstructed_vars{dg_mesh.number_of_grid_points()};
    evolution::dg::subcell::fd::reconstruct(
        make_not_null(&prefixed_reconstructed_vars), cell_values_vars, dg_mesh,
        subcell_mesh.extents());
    check_each_field_in_vars(prefixed_reconstructed_vars);

    // Check with the prefix on the DG and subcell vars
    prefixed_reconstructed_vars.initialize(0);
    evolution::dg::subcell::fd::reconstruct(
        make_not_null(&prefixed_reconstructed_vars), prefixed_cell_values_vars,
        dg_mesh, subcell_mesh.extents());
    check_each_field_in_vars(prefixed_reconstructed_vars);
  }
}

// [[TimeOut, 15]]
SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Fd.Reconstruction",
                  "[Evolution][Unit]") {
  test_reconstruct_fd<10, 1, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto>({5.0e-14});
  test_reconstruct_fd<10, 1, Spectral::Basis::Legendre,
                      Spectral::Quadrature::Gauss>({5.0e-14});

  test_reconstruct_fd<8, 2, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto>(
      {1.0e-14, 2.0e-13, 5.0e-7, 3.0e-7, 3.0e-8, 1.0e-9, 1.0e-10});
  test_reconstruct_fd<8, 2, Spectral::Basis::Legendre,
                      Spectral::Quadrature::Gauss>(
      {1.0e-14, 1.0e-14, 5.0e-7, 3.0e-7, 3.0e-8, 1.0e-9, 1.0e-10});

  test_reconstruct_fd<6, 3, Spectral::Basis::Legendre,
                      Spectral::Quadrature::GaussLobatto>(
      {1.0e-14, 1.0e-14, 3.0e-6, 3.0e-7, 3.0e-8});
  test_reconstruct_fd<6, 3, Spectral::Basis::Legendre,
                      Spectral::Quadrature::Gauss>(
      {1.0e-14, 1.0e-14, 1.0e-6, 3.0e-7, 3.0e-8});
}
}  // namespace
