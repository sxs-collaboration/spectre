// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>  // IWYU pragma: keep
// IWYU pragma: no_include <type_traits>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
template <size_t Dim>
Mesh<Dim> lgl_mesh(const std::array<size_t, Dim>& extents) noexcept {
  return {extents, Spectral::Basis::Legendre,
          Spectral::Quadrature::GaussLobatto};
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.MortarHelpers.mortar_mesh",
                  "[Unit][NumericalAlgorithms]") {
  CHECK(dg::mortar_mesh(lgl_mesh<0>({}), lgl_mesh<0>({})) == lgl_mesh<0>({}));
  CHECK(dg::mortar_mesh(lgl_mesh<1>({{3}}), lgl_mesh<1>({{5}})) ==
        lgl_mesh<1>({{5}}));
  CHECK(dg::mortar_mesh(lgl_mesh<2>({{2, 5}}), lgl_mesh<2>({{3, 4}})) ==
        lgl_mesh<2>({{3, 5}}));
}

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.MortarHelpers.projections",
                  "[Unit][NumericalAlgorithms]") {
  // Check 0D
  {
    Variables<tmpl::list<Var>> vars(1);
    get(get<Var>(vars)) = 4.;
    CHECK(get<Var>(
              dg::project_to_mortar(vars, lgl_mesh<0>({}), lgl_mesh<0>({}))) ==
          Scalar<DataVector>{{{{4.}}}});
    CHECK(get<Var>(dg::project_from_mortar(vars, lgl_mesh<0>({}),
                                           lgl_mesh<0>({}))) ==
          Scalar<DataVector>{{{{4.}}}});
  }
  // Check 1D
  {
    const auto mortar_mesh = lgl_mesh<1>({{7}});
    const auto mortar_coords = logical_coordinates(mortar_mesh);
    const auto func = [](
        const tnsr::I<DataVector, 1, Frame::Logical>& coords) noexcept {
      return pow<3>(get<0>(coords));
    };
    for (const auto& face_mesh : {mortar_mesh, lgl_mesh<1>({{5}})}) {
      CAPTURE(face_mesh);
      const auto face_coords = logical_coordinates(face_mesh);
      Variables<tmpl::list<Var>> vars(face_mesh.number_of_grid_points());
      get(get<Var>(vars)) = func(face_coords);
      CHECK_ITERABLE_APPROX(
          get(get<Var>(dg::project_to_mortar(vars, face_mesh, mortar_mesh))),
          func(mortar_coords));

      vars.initialize(mortar_mesh.number_of_grid_points());
      get(get<Var>(vars)) = func(mortar_coords);
      if (face_mesh.extents(0) < mortar_mesh.extents(0)) {
        // Add some data orthogonal to the function space on the face.
        // It should be projected to zero.

        // This is the face_mesh[0]th basis function (5 here).  There
        // is no nice way to get this from the spectral code (bug
        // #801).
        get(get<Var>(vars)) +=
            1. / 8. * get<0>(mortar_coords) *
            (15. + square(get<0>(mortar_coords)) *
                       (-70. + square(get<0>(mortar_coords)) * 63.));
      }
      CHECK_ITERABLE_APPROX(
          get(get<Var>(dg::project_from_mortar(vars, face_mesh, mortar_mesh))),
          func(face_coords));
    }
  }
  // Check 2D
  {
    const auto mortar_mesh = lgl_mesh<2>({{7, 8}});
    const auto mortar_coords = logical_coordinates(mortar_mesh);
    const auto func = [](
        const tnsr::I<DataVector, 2, Frame::Logical>& coords) noexcept {
      return pow<3>(get<0>(coords)) * pow<5>(get<1>(coords));
    };
    for (const auto& face_mesh :
         {mortar_mesh, lgl_mesh<2>({{5, 8}}), lgl_mesh<2>({{7, 6}}),
          lgl_mesh<2>({{5, 6}})}) {
      CAPTURE(face_mesh);
      const auto face_coords = logical_coordinates(face_mesh);
      Variables<tmpl::list<Var>> vars(face_mesh.number_of_grid_points());
      get(get<Var>(vars)) = func(face_coords);
      CHECK_ITERABLE_APPROX(
          get(get<Var>(dg::project_to_mortar(vars, face_mesh, mortar_mesh))),
          func(mortar_coords));

      vars.initialize(mortar_mesh.number_of_grid_points());
      get(get<Var>(vars)) = func(mortar_coords);
      if (face_mesh.extents(0) < mortar_mesh.extents(0)) {
        // Add some data orthogonal to the function space on the face.
        // It should be projected to zero.

        // This is the face_mesh[0]th basis function (5 here).  There
        // is no nice way to get this from the spectral code (bug
        // #801).
        get(get<Var>(vars)) +=
            1. / 8. * get<0>(mortar_coords) *
            (15. + square(get<0>(mortar_coords)) *
                       (-70. + square(get<0>(mortar_coords)) * 63.));
      }
      if (face_mesh.extents(1) < mortar_mesh.extents(1)) {
        // Add some data orthogonal to the function space on the face.
        // It should be projected to zero.

        // This is the face_mesh[1]th basis function (6 here).  There
        // is no nice way to get this from the spectral code (bug
        // #801).
        get(get<Var>(vars)) +=
            1. / 16. *
            (-5. +
             square(get<1>(mortar_coords)) *
                 (105. + square(get<1>(mortar_coords)) *
                             (-315. + square(get<1>(mortar_coords)) * 231.)));
      }
      CHECK_ITERABLE_APPROX(
          get(get<Var>(dg::project_from_mortar(vars, face_mesh, mortar_mesh))),
          func(face_coords));
    }
  }
}
