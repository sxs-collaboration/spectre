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
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/BasisFunctionValue.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/MinimumNumberOfPoints.hpp"
#include "NumericalAlgorithms/Spectral/Parity.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
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

    // Verify the DataVector + tmpl::list overload gives the same result as the
    // Variables overload for non-ZernikeB1 (Legendre) meshes, exercising the
    // non-ZernikeB1 fallback path in project_impl_with_tag_list.
    {
      using VarsTagList = tmpl::list<Tags::Scalar, Tags::Vector<Dim>>;
      DataVector packed_dg(vars.size());
      std::copy(vars.data(), vars.data() + vars.size(), packed_dg.data());
      const DataVector subcell_dv_taglist = evolution::dg::subcell::fd::project(
          packed_dg, dg_mesh, subcell_mesh.extents(), VarsTagList{});
      DataVector packed_subcell(subcell_vars.size());
      std::copy(subcell_vars.data(), subcell_vars.data() + subcell_vars.size(),
                packed_subcell.data());
      CHECK_ITERABLE_APPROX(subcell_dv_taglist, packed_subcell);
    }
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
        evolution::dg::subcell::fd::project_to_faces(
            nodal_coeffs, dg_mesh, subcell_mesh.extents(), Face_Dim,
            Spectral::Parity::Uninitialized);
    CHECK_ITERABLE_APPROX(subcell_values, expected_subcell_values);
  }
}

void test_project_zernike_b1() {
  constexpr Spectral::Basis zb1 = Spectral::Basis::ZernikeB1;
  constexpr Spectral::Quadrature zb1_quad =
      Spectral::Quadrature::GaussRadauUpper;
  const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.);

  for (size_t n_r = Spectral::minimum_number_of_points<zb1, zb1_quad>; n_r <= 5;
       ++n_r) {
    CAPTURE(n_r);
    const size_t n_fd = 2 * n_r - 1;
    {
      INFO("Spherical");
      const Mesh<3> dg_mesh{
          {{n_r, 1, 1}},
          {zb1, Spectral::Basis::Cartoon, Spectral::Basis::Cartoon},
          {zb1_quad, Spectral::Quadrature::SphericalSymmetry,
           Spectral::Quadrature::SphericalSymmetry}};
      const Index<3> subcell_extents{{n_fd, 1, 1}};
      const auto xi_dg_r = get<0>(logical_coordinates(dg_mesh));
      const Mesh<1> fd_1d{n_fd, Spectral::Basis::FiniteDifference,
                          Spectral::Quadrature::CellCentered};
      const auto xi_fd_r = get<0>(logical_coordinates(fd_1d));

      for (const Spectral::Parity parity :
           {Spectral::Parity::Even, Spectral::Parity::Odd}) {
        CAPTURE(parity);
        for (size_t k = 0; k < n_r; ++k) {
          CAPTURE(k);
          const size_t n = parity == Spectral::Parity::Even ? 2 * k : 2 * k + 1;
          const size_t m =
              parity == Spectral::Parity::Even ? size_t{0} : size_t{1};
          const DataVector dg_u =
              Spectral::compute_basis_function_value<zb1>(n, m, xi_dg_r);
          const DataVector expected =
              Spectral::compute_basis_function_value<zb1>(n, m, xi_fd_r);
          const DataVector result = evolution::dg::subcell::fd::project(
              dg_u, dg_mesh, subcell_extents, parity);
          CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
        }
      }
    }
    {
      INFO("Axial");
      for (size_t n_phi = 2; n_phi <= 4; ++n_phi) {
        CAPTURE(n_phi);
        const size_t n_fd_phi = 2 * n_phi - 1;
        const Mesh<3> dg_mesh{
            {{n_r, n_phi, 1}},
            {zb1, Spectral::Basis::Legendre, Spectral::Basis::Cartoon},
            {zb1_quad, Spectral::Quadrature::GaussLobatto,
             Spectral::Quadrature::AxialSymmetry}};
        const Index<3> subcell_extents{{n_fd, n_fd_phi, 1}};

        const auto xi_dg = logical_coordinates(dg_mesh);
        const auto& xi_dg_r = get<0>(xi_dg);
        const auto& xi_dg_phi = get<1>(xi_dg);

        const Mesh<3> subcell_mesh{
            {{n_fd, n_fd_phi, 1}},
            {Spectral::Basis::FiniteDifference,
             Spectral::Basis::FiniteDifference, Spectral::Basis::Cartoon},
            {Spectral::Quadrature::CellCentered,
             Spectral::Quadrature::CellCentered,
             Spectral::Quadrature::AxialSymmetry}};
        const auto xi_fd = logical_coordinates(subcell_mesh);
        const auto& xi_fd_r = get<0>(xi_fd);
        const auto& xi_fd_phi = get<1>(xi_fd);

        for (const Spectral::Parity parity :
             {Spectral::Parity::Even, Spectral::Parity::Odd}) {
          CAPTURE(parity);
          // Use the lowest-degree basis function of the given parity times
          // the phi coordinate (degree-1 Legendre, exact for n_phi >= 2).
          const size_t n =
              parity == Spectral::Parity::Even ? size_t{0} : size_t{1};
          const size_t m =
              parity == Spectral::Parity::Even ? size_t{0} : size_t{1};
          const DataVector dg_u =
              Spectral::compute_basis_function_value<zb1>(n, m, xi_dg_r) *
              xi_dg_phi;
          const DataVector expected =
              Spectral::compute_basis_function_value<zb1>(n, m, xi_fd_r) *
              xi_fd_phi;
          const DataVector result = evolution::dg::subcell::fd::project(
              dg_u, dg_mesh, subcell_extents, parity);
          CHECK_ITERABLE_CUSTOM_APPROX(result, expected, custom_approx);
        }
      }
    }
  }
}

// Tests the Variables overload and the DataVector + tmpl::list overload on
// ZernikeB1 DG meshes
void test_project_zernike_b1_variables_and_datavector_taglist() {
  constexpr Spectral::Basis zb1 = Spectral::Basis::ZernikeB1;
  constexpr Spectral::Quadrature zb1_quad =
      Spectral::Quadrature::GaussRadauUpper;
  const Approx custom_approx = Approx::custom().epsilon(1.0e-11).scale(1.);

  for (size_t n_r = Spectral::minimum_number_of_points<zb1, zb1_quad>; n_r <= 5;
       ++n_r) {
    CAPTURE(n_r);
    const size_t n_fd = 2 * n_r - 1;

    const Mesh<3> dg_mesh{
        {{n_r, 1, 1}},
        {zb1, Spectral::Basis::Cartoon, Spectral::Basis::Cartoon},
        {zb1_quad, Spectral::Quadrature::SphericalSymmetry,
         Spectral::Quadrature::SphericalSymmetry}};
    const Index<3> subcell_extents{{n_fd, 1, 1}};
    const auto xi_dg_r = get<0>(logical_coordinates(dg_mesh));
    const Mesh<1> fd_1d{n_fd, Spectral::Basis::FiniteDifference,
                        Spectral::Quadrature::CellCentered};
    const auto xi_fd_r = get<0>(logical_coordinates(fd_1d));

    using VarsTagList = tmpl::list<Tags::Scalar, Tags::Vector<3>>;

    // Lowest-degree ZernikeB1 basis functions of each parity.
    const DataVector even_dg =
        Spectral::compute_basis_function_value<zb1>(0, 0, xi_dg_r);
    const DataVector odd_dg =
        Spectral::compute_basis_function_value<zb1>(1, 1, xi_dg_r);
    const DataVector even_fd =
        Spectral::compute_basis_function_value<zb1>(0, 0, xi_fd_r);
    const DataVector odd_fd =
        Spectral::compute_basis_function_value<zb1>(1, 1, xi_fd_r);

    Variables<VarsTagList> dg_vars(dg_mesh.number_of_grid_points());
    get(get<Tags::Scalar>(dg_vars)) = even_dg;
    get<Tags::Vector<3>>(dg_vars).get(0) = 2.0 * odd_dg;
    get<Tags::Vector<3>>(dg_vars).get(1) = 3.0 * even_dg;
    get<Tags::Vector<3>>(dg_vars).get(2) = 4.0 * even_dg;

    const Variables<VarsTagList> subcell_vars =
        evolution::dg::subcell::fd::project(dg_vars, dg_mesh, subcell_extents);
    CHECK_ITERABLE_CUSTOM_APPROX(get(get<Tags::Scalar>(subcell_vars)), even_fd,
                                 custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::Vector<3>>(subcell_vars).get(0),
                                 2.0 * odd_fd, custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::Vector<3>>(subcell_vars).get(1),
                                 3.0 * even_fd, custom_approx);
    CHECK_ITERABLE_CUSTOM_APPROX(get<Tags::Vector<3>>(subcell_vars).get(2),
                                 4.0 * even_fd, custom_approx);

    DataVector packed_dg(dg_vars.size());
    std::copy(dg_vars.data(), dg_vars.data() + dg_vars.size(),
              packed_dg.data());
    const DataVector subcell_dv = evolution::dg::subcell::fd::project(
        packed_dg, dg_mesh, subcell_extents, VarsTagList{});
    DataVector expected_packed_subcell(subcell_vars.size());
    std::copy(subcell_vars.data(), subcell_vars.data() + subcell_vars.size(),
              expected_packed_subcell.data());
    CHECK_ITERABLE_CUSTOM_APPROX(subcell_dv, expected_packed_subcell,
                                 custom_approx);
  }
}

#ifdef SPECTRE_DEBUG
void test_projection_asserts() {
  // ZernikeB1 mesh with Parity::Uninitialized
  {
    const Mesh<3> dg_mesh{{{4, 1, 1}},
                          {Spectral::Basis::ZernikeB1, Spectral::Basis::Cartoon,
                           Spectral::Basis::Cartoon},
                          {Spectral::Quadrature::GaussRadauUpper,
                           Spectral::Quadrature::SphericalSymmetry,
                           Spectral::Quadrature::SphericalSymmetry}};
    const Index<3> subcell_extents{{7, 1, 1}};
    const DataVector dg_u(4, 1.0);
    CHECK_THROWS_WITH(
        evolution::dg::subcell::fd::project(dg_u, dg_mesh, subcell_extents,
                                            Spectral::Parity::Uninitialized),
        Catch::Matchers::ContainsSubstring(
            "Parity must be set when using ZernikeB1"));
  }
  // Passing a multi-component DataVector (size > num_dg_pts) with a
  // non-Uninitialized parity
  {
    const Mesh<3> dg_mesh{4, Spectral::Basis::Legendre,
                          Spectral::Quadrature::GaussLobatto};
    const Index<3> subcell_extents{7};
    const DataVector dg_u(2 * dg_mesh.number_of_grid_points(), 1.0);
    CHECK_THROWS_WITH(
        evolution::dg::subcell::fd::project(dg_u, dg_mesh, subcell_extents,
                                            Spectral::Parity::Even),
        Catch::Matchers::ContainsSubstring(
            "Must pass the types as a template"));
  }
}
#endif

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
  test_project_zernike_b1();
  test_project_zernike_b1_variables_and_datavector_taglist();
#ifdef SPECTRE_DEBUG
  test_projection_asserts();
#endif
}
}  // namespace
