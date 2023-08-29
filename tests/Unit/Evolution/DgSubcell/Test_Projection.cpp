// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <cstddef>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/DgSubcell/Projection.hpp"
#include "Helpers/Evolution/DgSubcell/ProjectionTestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace Tags {
template <typename Tag>
struct Prefix : db::PrefixTag, db::SimpleTag {
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

template <size_t MaxPts, size_t Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void test_project_fd() {
  CAPTURE(Dim);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);

  for (size_t num_pts_1d = std::max(
           static_cast<size_t>(2),
           Spectral::minimum_number_of_points<BasisType, QuadratureType>);
       num_pts_1d < MaxPts + 1; ++num_pts_1d) {
    CAPTURE(num_pts_1d);
    const Mesh<Dim> dg_mesh{num_pts_1d, BasisType, QuadratureType};
    const auto logical_coords = logical_coordinates(dg_mesh);
    const size_t num_subcells_1d = 2 * num_pts_1d - 1;
    const Mesh<Dim> subcell_mesh(num_subcells_1d,
                                 Spectral::Basis::FiniteDifference,
                                 Spectral::Quadrature::CellCentered);
    const DataVector nodal_coeffs =
        TestHelpers::evolution::dg::subcell::cell_values(dg_mesh.extents(0) - 2,
                                                         logical_coords);
    const DataVector expected_subcell_values =
        TestHelpers::evolution::dg::subcell::cell_values(
            dg_mesh.extents(0) - 2, logical_coordinates(subcell_mesh));
    // Test projection of a DataVector
    const DataVector subcell_values = evolution::dg::subcell::fd::project(
        nodal_coeffs, dg_mesh, subcell_mesh.extents());
    CHECK_ITERABLE_APPROX(subcell_values, expected_subcell_values);

    // Test projection of a Variables using the return-by-value function
    Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>> vars(
        dg_mesh.number_of_grid_points());
    get(get<Tags::Scalar>(vars)) = nodal_coeffs;
    for (size_t d = 0; d < Dim; ++d) {
      get<Tags::Vector<Dim>>(vars).get(d) = (d + 2.0) * nodal_coeffs;
    }
    Variables<tmpl::list<Tags::Scalar, Tags::Vector<Dim>>> subcell_vars =
        evolution::dg::subcell::fd::project(vars, dg_mesh,
                                            subcell_mesh.extents());

    const auto check_each_field_in_vars =
        [&expected_subcell_values](const auto& local_subcell_vars) {
          CHECK_ITERABLE_APPROX(
              get(get<tmpl::front<typename std::decay_t<decltype(
                      local_subcell_vars)>::tags_list>>(local_subcell_vars)),
              expected_subcell_values);
          for (size_t d = 0; d < Dim; ++d) {
            CHECK_ITERABLE_APPROX(
                get<tmpl::back<typename std::decay_t<decltype(
                    local_subcell_vars)>::tags_list>>(local_subcell_vars)
                    .get(d),
                (d + 2.0) * expected_subcell_values);
          }
        };
    check_each_field_in_vars(subcell_vars);

    // Check with the prefix on the subcell vars
    Variables<
        tmpl::list<Tags::Prefix<Tags::Scalar>, Tags::Prefix<Tags::Vector<Dim>>>>
        prefixed_subcell_vars{subcell_mesh.number_of_grid_points()};
    evolution::dg::subcell::fd::project(make_not_null(&prefixed_subcell_vars),
                                        vars, dg_mesh, subcell_mesh.extents());
    check_each_field_in_vars(prefixed_subcell_vars);

    // Check with the prefix on the DG vars
    Variables<
        tmpl::list<Tags::Prefix<Tags::Scalar>, Tags::Prefix<Tags::Vector<Dim>>>>
        prefixed_vars(dg_mesh.number_of_grid_points());
    prefixed_vars = vars;
    subcell_vars.initialize(0);
    evolution::dg::subcell::fd::project(make_not_null(&subcell_vars),
                                        prefixed_vars, dg_mesh,
                                        subcell_mesh.extents());
    check_each_field_in_vars(subcell_vars);

    // Check with the prefix on the DG and subcell vars
    prefixed_subcell_vars.initialize(0);
    evolution::dg::subcell::fd::project(make_not_null(&prefixed_subcell_vars),
                                        prefixed_vars, dg_mesh,
                                        subcell_mesh.extents());
    check_each_field_in_vars(prefixed_subcell_vars);
  }
}

template <size_t MaxPts, size_t Dim, size_t Face_Dim, Spectral::Basis BasisType,
          Spectral::Quadrature QuadratureType>
void test_project_on_face_fd() {
  CAPTURE(Dim);
  CAPTURE(Face_Dim);
  CAPTURE(BasisType);
  CAPTURE(QuadratureType);

  for (size_t num_pts_1d = std::max(
           static_cast<size_t>(2),
           Spectral::minimum_number_of_points<BasisType, QuadratureType>);
       num_pts_1d < MaxPts + 1; ++num_pts_1d) {
    CAPTURE(num_pts_1d);
    const Mesh<Dim> dg_mesh{num_pts_1d, BasisType, QuadratureType};
    const auto logical_coords = logical_coordinates(dg_mesh);
    const size_t num_subcells_1d_face = 2 * num_pts_1d;
    const size_t num_subcells_1d_cell = 2 * num_pts_1d - 1;
    CAPTURE(num_subcells_1d_face);
    CAPTURE(num_subcells_1d_cell);

    std::array<size_t, Dim> extents{};
    std::array<Spectral::Basis, Dim> basis{};
    std::array<Spectral::Quadrature, Dim> quadrature{};
    for (size_t d = 0; d < Dim; d++) {
      basis[d] = Spectral::Basis::FiniteDifference;
      if (d == Face_Dim) {
        extents[d] = num_subcells_1d_face;
        quadrature[d] = Spectral::Quadrature::FaceCentered;
      } else {
        extents[d] = num_subcells_1d_cell;
        quadrature[d] = Spectral::Quadrature::CellCentered;
      }
    }

    const Mesh<Dim> subcell_mesh(extents, basis, quadrature);
    const DataVector nodal_coeffs =
        TestHelpers::evolution::dg::subcell::cell_values(dg_mesh.extents(0) - 2,
                                                         logical_coords);
    const DataVector expected_subcell_values =
        TestHelpers::evolution::dg::subcell::cell_values(
            dg_mesh.extents(0) - 2, logical_coordinates(subcell_mesh));
    // Test projection of a DataVector
    const DataVector subcell_values =
        evolution::dg::subcell::fd::project_to_face(
            nodal_coeffs, dg_mesh, subcell_mesh.extents(), Face_Dim);
    CHECK_ITERABLE_APPROX(subcell_values, expected_subcell_values);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Subcell.Fd.Projection", "[Evolution][Unit]") {
  test_project_fd<10, 1, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto>();
  test_project_fd<10, 1, Spectral::Basis::Legendre,
                  Spectral::Quadrature::Gauss>();

  test_project_fd<10, 2, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto>();
  test_project_fd<10, 2, Spectral::Basis::Legendre,
                  Spectral::Quadrature::Gauss>();

  test_project_fd<5, 3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::GaussLobatto>();
  test_project_fd<4, 3, Spectral::Basis::Legendre,
                  Spectral::Quadrature::Gauss>();
  test_project_on_face_fd<10, 1, 0, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto>();
  test_project_on_face_fd<10, 1, 0, Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss>();
  test_project_on_face_fd<5, 3, 0, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto>();
  test_project_on_face_fd<4, 3, 0, Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss>();
  test_project_on_face_fd<5, 3, 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto>();
  test_project_on_face_fd<4, 3, 1, Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss>();
  test_project_on_face_fd<5, 3, 2, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto>();
  test_project_on_face_fd<4, 3, 2, Spectral::Basis::Legendre,
                          Spectral::Quadrature::Gauss>();
}
}  // namespace
