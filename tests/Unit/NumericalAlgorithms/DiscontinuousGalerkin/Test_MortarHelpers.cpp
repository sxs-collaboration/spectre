// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <initializer_list>  // IWYU pragma: keep
#include <utility>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFlux.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/MortarHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdArrayHelpers.hpp"
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
  const auto test_segment = [](const SegmentId& base,
                               const size_t test) noexcept {
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
  const auto expected_size = [](const size_t test) noexcept {
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
tnsr::I<DataVector, Dim, Frame::Logical> scaled_coords(
    tnsr::I<DataVector, Dim, Frame::Logical> coords,
    const std::array<Spectral::MortarSize, Dim>& mortar_size) noexcept {
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
DataVector basis5(const DataVector& coords) noexcept {
  return 1. / 8. * coords *
         (15. + square(coords) * (-70. + square(coords) * 63.));
}
DataVector basis6(const DataVector& coords) noexcept {
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
    const auto func = [](const tnsr::I<DataVector, 1, Frame::Logical>&
                             coords) noexcept -> DataVector {
      return pow<3>(get<0>(coords));
    };
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

      const auto make_mortar_data = [
        &face_mesh, &func, &mortar_coords, &mortar_mesh
      ](const std::array<MortarSize, 1>& mortar_size) noexcept {
        Variables<tmpl::list<Var>> vars(mortar_mesh.number_of_grid_points());
        get(get<Var>(vars)) = func(scaled_coords(mortar_coords, mortar_size));
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
    const auto func = [](const tnsr::I<DataVector, 2, Frame::Logical>&
                             coords) noexcept -> DataVector {
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

      const auto make_mortar_data = [
        &face_mesh, &func, &mortar_coords, &mortar_mesh
      ](const std::array<MortarSize, 2>& mortar_size) noexcept {
        Variables<tmpl::list<Var>> vars(mortar_mesh.number_of_grid_points());
        get(get<Var>(vars)) = func(scaled_coords(mortar_coords, mortar_size));

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

namespace {
struct ComputeBoundaryFluxContributionMetavars {
  using temporal_id = size_t;

  struct system {
    static constexpr size_t volume_dim = 1;
    using variables_tag = Tags::Variables<tmpl::list<Var>>;
  };

  struct normal_dot_numerical_flux {
    struct type {
      DataVector answer;

      using package_tags = tmpl::list<Var>;
      void operator()(const gsl::not_null<Scalar<DataVector>*> numerical_flux,
                      const Scalar<DataVector>& local_var,
                      const Scalar<DataVector>& remote_var) const noexcept {
        CHECK(get(local_var) == DataVector{1., 2., 3.});
        CHECK(get(remote_var) == DataVector{6., 5., 4.});
        get(*numerical_flux) = answer;
      }
    };
  };
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.MortarHelpers.compute_boundary_flux_contribution",
                  "[Unit][NumericalAlgorithms]") {
  using flux_comm_types =
      dg::FluxCommunicationTypes<ComputeBoundaryFluxContributionMetavars>;
  using flux_computer = typename ComputeBoundaryFluxContributionMetavars::
      normal_dot_numerical_flux::type;

  // p-refinement
  {
    const Mesh<1> face_mesh(2, Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
    const Mesh<1> mortar_mesh(3, Spectral::Basis::Legendre,
                              Spectral::Quadrature::GaussLobatto);
    const size_t extent_perpendicular_to_boundary = 4;
    const std::array<Spectral::MortarSize, 1> mortar_size{
        {Spectral::MortarSize::Full}};

    flux_comm_types::LocalData local_data{};
    local_data.mortar_data.initialize(mortar_mesh.number_of_grid_points());
    get(get<Var>(local_data.mortar_data)) = DataVector{1., 2., 3.};
    get(get<Tags::NormalDotFlux<Var>>(local_data.mortar_data)) =
        DataVector{-3., 0., 3.};
    get(local_data.magnitude_of_face_normal) = DataVector{2., 5.};

    flux_comm_types::PackagedData remote_data(
        mortar_mesh.number_of_grid_points());
    get(get<Var>(remote_data)) = DataVector{6., 5., 4.};

    // projected F* - F = {5., -1.}
    Variables<tmpl::list<Tags::NormalDotNumericalFlux<Var>>> fstar_minus_f(2);
    get(get<Tags::NormalDotNumericalFlux<Var>>(fstar_minus_f)) =
        DataVector{5., -1.};
    const auto expected =
        dg::lift_flux(fstar_minus_f, extent_perpendicular_to_boundary,
                      local_data.magnitude_of_face_normal);

    const auto result = dg::compute_boundary_flux_contribution<flux_comm_types>(
        flux_computer{{0., 3., 0.}}, local_data, remote_data, face_mesh,
        mortar_mesh, extent_perpendicular_to_boundary, mortar_size);
    CHECK_ITERABLE_APPROX(get<Var>(result), get<Var>(expected));
    CHECK(dg::compute_boundary_flux_contribution<flux_comm_types>(
              flux_computer{{0., 3., 0.}}, std::move(local_data), remote_data,
              face_mesh, mortar_mesh, extent_perpendicular_to_boundary,
              mortar_size) ==
          result);
  }

  // h-refinement
  {
    const auto compute_contribution = [](
        const std::array<Spectral::MortarSize, 1>& mortar_size,
        const DataVector& numerical_flux) noexcept {
      const Mesh<1> mesh(3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto);

      // These are all arbitrary
      const size_t extent_perpendicular_to_boundary = 4;
      const DataVector local_flux{-1., 5., 7.};
      const DataVector magnitude_of_face_normal{2., 5., 7.};

      flux_comm_types::LocalData local_data{};
      local_data.mortar_data.initialize(mesh.number_of_grid_points(), 0.);
      get(get<Tags::NormalDotFlux<Var>>(local_data.mortar_data)) = local_flux;
      local_data.mortar_data = dg::project_to_mortar(local_data.mortar_data,
                                                     mesh, mesh, mortar_size);
      get(get<Var>(local_data.mortar_data)) = DataVector{1., 2., 3.};
      get(local_data.magnitude_of_face_normal) = magnitude_of_face_normal;

      flux_comm_types::PackagedData remote_data(mesh.number_of_grid_points());
      get(get<Var>(remote_data)) = DataVector{6., 5., 4.};

      return dg::compute_boundary_flux_contribution<flux_comm_types>(
          flux_computer{numerical_flux}, local_data, remote_data, mesh, mesh,
          extent_perpendicular_to_boundary, mortar_size);
    };

    const auto unrefined_result =
        compute_contribution({{Spectral::MortarSize::Full}}, {1., 4., 9.});
    const decltype(unrefined_result) refined_result =
        compute_contribution({{Spectral::MortarSize::LowerHalf}},
                             {1., 9. / 4., 4.}) +
        compute_contribution({{Spectral::MortarSize::UpperHalf}},
                             {4., 25. / 4., 9.});

    CHECK_ITERABLE_APPROX(get<Var>(unrefined_result), get<Var>(refined_result));
  }
}
