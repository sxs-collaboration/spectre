// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>  // IWYU pragma: keep
#include <utility>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Structure/Side.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace {
template <size_t Dim>
Mesh<Dim> lgl_mesh(const std::array<size_t, Dim>& extents) {
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

SPECTRE_TEST_CASE("Unit.DG.MortarHelpers.mortar_size",
                  "[Unit][NumericalAlgorithms]") {
  CHECK(dg::mortar_size(ElementId<1>(0, {{{0, 0}}}),
                        ElementId<1>(0, {{{5, 2}}}), 0, {}) ==
        std::array<Spectral::MortarSize, 0>{});

  // Check the root segment to make sure the code doesn't try to get
  // its parent.
  CHECK(dg::mortar_size(ElementId<2>(0, {{{0, 0}, {0, 0}}}),
                        ElementId<2>(1, {{{0, 0}, {0, 0}}}), 1, {}) ==
        std::array<Spectral::MortarSize, 1>{{Spectral::MortarSize::Full}});
  CHECK(dg::mortar_size(ElementId<2>(0, {{{1, 0}, {0, 0}}}),
                        ElementId<2>(1, {{{0, 0}, {0, 0}}}), 1, {}) ==
        std::array<Spectral::MortarSize, 1>{{Spectral::MortarSize::Full}});
  CHECK(dg::mortar_size(ElementId<2>(0, {{{0, 0}, {0, 0}}}),
                        ElementId<2>(1, {{{1, 0}, {0, 0}}}), 1, {}) ==
        std::array<Spectral::MortarSize, 1>{{Spectral::MortarSize::LowerHalf}});

  // Check all the aligned cases in 3D
  const auto test_segment = [](const SegmentId& base, const size_t test) {
    switch (test) {
      case 0:
        return base;
      case 1:
        return base.id_of_parent();
      case 2:
        return base.id_of_child(Side::Lower);
      case 3:
        return base.id_of_child(Side::Upper);
      default:
        ERROR("Test logic error");
    }
  };
  const auto expected_size = [](const size_t test) {
    switch (test) {
      case 0:
      case 1:
        return Spectral::MortarSize::Full;
      case 2:
        return Spectral::MortarSize::LowerHalf;
      case 3:
        return Spectral::MortarSize::UpperHalf;
      default:
        ERROR("Test logic error");
    }
  };

  const SegmentId segment0(1, 1);
  const SegmentId segment1(2, 0);
  // We do not expect to actually have abutting elements with
  // difference greater than one in perpendicular refinement levels,
  // but this function should work even in that situation, so we test
  // it here.
  const SegmentId perp0(5, 1);
  const SegmentId perp1(7, 20);

  for (size_t dimension = 0; dimension < 3; ++dimension) {
    using SegArray = std::array<SegmentId, 2>;
    const ElementId<3> self(
        0, insert_element(SegArray{{segment0, segment1}}, dimension, perp0));

    for (size_t test0 = 0; test0 < 4; ++test0) {
      for (size_t test1 = 0; test1 < 4; ++test1) {
        const ElementId<3> neighbor(
            0, insert_element(SegArray{{test_segment(segment0, test0),
                                        test_segment(segment1, test1)}},
                              dimension, perp1));
        CAPTURE(neighbor);
        const std::array<Spectral::MortarSize, 2> expected{
            {expected_size(test0), expected_size(test1)}};
        CHECK(dg::mortar_size(self, neighbor, dimension, {}) == expected);
      }
    }
  }

  // Check an orientation case
  CHECK(dg::mortar_size(ElementId<3>(1, {{{0, 0}, {3, 2}, {7, 5}}}),
                        ElementId<3>(4, {{{6, 61}, {3, 0}, {4, 5}}}), 0,
                        OrientationMap<3>{{{Direction<3>::upper_eta(),
                                            Direction<3>::upper_zeta(),
                                            Direction<3>::lower_xi()}}}) ==
        std::array<Spectral::MortarSize, 2>{
            {Spectral::MortarSize::UpperHalf, Spectral::MortarSize::Full}});
}

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

// Scales the coordinates to a mortar
template <size_t Dim>
tnsr::I<DataVector, Dim, Frame::ElementLogical> scaled_coords(
    tnsr::I<DataVector, Dim, Frame::ElementLogical> coords,
    const std::array<Spectral::MortarSize, Dim>& mortar_size) {
  for (size_t d = 0; d < Dim; ++d) {
    switch (gsl::at(mortar_size, d)) {
      case Spectral::MortarSize::LowerHalf:
        coords.get(d) = 0.5 * (coords.get(d) - 1.);
        break;
      case Spectral::MortarSize::UpperHalf:
        coords.get(d) = 0.5 * (coords.get(d) + 1.);
        break;
      default:
        break;
    }
  }
  return coords;
}

// There is no nice way to get the basis functions from the spectral
// code (bug #801).  In each case here we want the first unresolved
// basis function.
DataVector basis5(const DataVector& coords) {
  return 1. / 8. * coords *
         (15. + square(coords) * (-70. + square(coords) * 63.));
}
DataVector basis6(const DataVector& coords) {
  return 1. / 16. *
         (-5. + square(coords) *
                    (105. + square(coords) * (-315. + square(coords) * 231.)));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.MortarHelpers.projections",
                  "[Unit][NumericalAlgorithms]") {
  using Spectral::MortarSize;
  const auto all_mortar_sizes = {MortarSize::Full, MortarSize::LowerHalf,
                                 MortarSize::UpperHalf};
  // Check 0D
  {
    Variables<tmpl::list<Var>> vars(1);
    get(get<Var>(vars)) = 4.;
    CHECK(get<Var>(dg::project_to_mortar(vars, lgl_mesh<0>({}), lgl_mesh<0>({}),
                                         {})) == Scalar<DataVector>{{{{4.}}}});
  }
  // Check 1D
  {
    const auto mortar_mesh = lgl_mesh<1>({{7}});
    const auto mortar_coords = logical_coordinates(mortar_mesh);
    const auto func =
        [](const tnsr::I<DataVector, 1, Frame::ElementLogical>& coords)
        -> DataVector { return pow<3>(get<0>(coords)); };
    for (const auto& face_mesh : {mortar_mesh, lgl_mesh<1>({{5}})}) {
      CAPTURE(face_mesh);
      const auto face_coords = logical_coordinates(face_mesh);
      // face -> mortar
      for (const auto& slice_size : all_mortar_sizes) {
        const std::array<MortarSize, 1> mortar_size{{slice_size}};
        CAPTURE(mortar_size);
        Variables<tmpl::list<Var>> vars(face_mesh.number_of_grid_points());
        get(get<Var>(vars)) = func(face_coords);
        CHECK_ITERABLE_APPROX(get(get<Var>(dg::project_to_mortar(
                                  vars, face_mesh, mortar_mesh, mortar_size))),
                              func(scaled_coords(mortar_coords, mortar_size)));
      }

      const auto make_mortar_data =
          [&face_mesh, &func, &mortar_coords,
           &mortar_mesh](const std::array<MortarSize, 1>& mortar_size) {
            Variables<tmpl::list<Var>> vars(
                mortar_mesh.number_of_grid_points());
            get(get<Var>(vars)) =
                func(scaled_coords(mortar_coords, mortar_size));
            if (face_mesh.extents(0) < mortar_mesh.extents(0)) {
              // Add some data orthogonal to the function space on the
              // face.  It should be projected to zero.
              get(get<Var>(vars)) +=
                  basis5(get<0>(scaled_coords(mortar_coords, mortar_size)));
            }
            return vars;
          };

      // full mortar -> face
      if (face_mesh != mortar_mesh) {
        const std::array<MortarSize, 1> mortar_size{{MortarSize::Full}};
        CAPTURE(mortar_size);
        const auto vars = make_mortar_data(mortar_size);
        CHECK_ITERABLE_APPROX(get(get<Var>(dg::project_from_mortar(
                                  vars, face_mesh, mortar_mesh, mortar_size))),
                              func(face_coords));
      }
      // half mortar -> face
      {
        const auto vars_lo = make_mortar_data({{MortarSize::LowerHalf}});
        const auto vars_hi = make_mortar_data({{MortarSize::UpperHalf}});
        CHECK_ITERABLE_APPROX(
            get(get<Var>(dg::project_from_mortar(
                vars_lo, face_mesh, mortar_mesh, {{MortarSize::LowerHalf}}))) +
                get(get<Var>(
                    dg::project_from_mortar(vars_hi, face_mesh, mortar_mesh,
                                            {{MortarSize::UpperHalf}}))),
            func(face_coords));
      }
    }
  }
  // Check 2D
  {
    const auto mortar_mesh = lgl_mesh<2>({{7, 8}});
    const auto mortar_coords = logical_coordinates(mortar_mesh);
    const auto func =
        [](const tnsr::I<DataVector, 2, Frame::ElementLogical>& coords)
        -> DataVector {
      return pow<3>(get<0>(coords)) * pow<5>(get<1>(coords));
    };
    for (const auto& face_mesh :
         {mortar_mesh, lgl_mesh<2>({{5, 8}}), lgl_mesh<2>({{7, 6}}),
          lgl_mesh<2>({{5, 6}})}) {
      CAPTURE(face_mesh);
      const auto face_coords = logical_coordinates(face_mesh);
      // face -> mortar
      for (const auto& slice_size0 : all_mortar_sizes) {
        for (const auto& slice_size1 : all_mortar_sizes) {
          const std::array<MortarSize, 2> mortar_size{
              {slice_size0, slice_size1}};
          CAPTURE(mortar_size);
          Variables<tmpl::list<Var>> vars(face_mesh.number_of_grid_points());
          get(get<Var>(vars)) = func(face_coords);
          CHECK_ITERABLE_APPROX(
              get(get<Var>(dg::project_to_mortar(vars, face_mesh, mortar_mesh,
                                                 mortar_size))),
              func(scaled_coords(mortar_coords, mortar_size)));
        }
      }

      const auto make_mortar_data =
          [&face_mesh, &func, &mortar_coords,
           &mortar_mesh](const std::array<MortarSize, 2>& mortar_size) {
            Variables<tmpl::list<Var>> vars(
                mortar_mesh.number_of_grid_points());
            get(get<Var>(vars)) =
                func(scaled_coords(mortar_coords, mortar_size));

            // Add some data orthogonal to the function space on the face.
            // It should be projected to zero.
            if (face_mesh.extents(0) < mortar_mesh.extents(0)) {
              get(get<Var>(vars)) += basis5(get<0>(mortar_coords));
            }
            if (face_mesh.extents(1) < mortar_mesh.extents(1)) {
              get(get<Var>(vars)) += basis6(get<1>(mortar_coords));
            }
            return vars;
          };

      // full mortar -> face
      if (face_mesh != mortar_mesh) {
        const std::array<MortarSize, 2> mortar_size{
            {MortarSize::Full, MortarSize::Full}};
        const auto vars = make_mortar_data(mortar_size);
        CHECK_ITERABLE_APPROX(get(get<Var>(dg::project_from_mortar(
                                  vars, face_mesh, mortar_mesh, mortar_size))),
                              func(face_coords));
      }

      // mortar -> face from a mortar long in one direction and short
      // in the other
      {
        const auto vars_lo =
            make_mortar_data({{MortarSize::Full, MortarSize::LowerHalf}});
        const auto vars_hi =
            make_mortar_data({{MortarSize::Full, MortarSize::UpperHalf}});
        CHECK_ITERABLE_APPROX(
            get(get<Var>(dg::project_from_mortar(
                vars_lo, face_mesh, mortar_mesh,
                {{MortarSize::Full, MortarSize::LowerHalf}}))) +
                get(get<Var>(dg::project_from_mortar(
                    vars_hi, face_mesh, mortar_mesh,
                    {{MortarSize::Full, MortarSize::UpperHalf}}))),
            func(face_coords));
      }
    }
  }
}
