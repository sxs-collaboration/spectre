// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/DiscreteRotation.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Side.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/TMPL.hpp"

namespace {

using Affine = domain::CoordinateMaps::Affine;
using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
using Affine3D = domain::CoordinateMaps::ProductOf3Maps<Affine, Affine, Affine>;

struct ScalarTensor {
  using type = Scalar<DataVector>;
};

template <size_t SpatialDim>
struct Coords {
  using type = tnsr::I<DataVector, SpatialDim, Frame::Inertial>;
};

void test_1d_orient_variables() noexcept {
  const Index<1> extents{4};
  Variables<tmpl::list<ScalarTensor, Coords<1>>> vars(extents.product());
  get(get<ScalarTensor>(vars)) = DataVector{{1.0, 2.0, 3.0, 4.0}};
  get<0>(get<Coords<1>>(vars)) = DataVector{{0.5, 0.6, 0.7, 0.8}};

  // Check aligned case
  {
    const auto oriented_vars = orient_variables(vars, extents, {});
    CHECK(oriented_vars == vars);
  }

  // Check anti-aligned case
  {
    const OrientationMap<1> orientation_map(
        std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
    const auto oriented_vars = orient_variables(vars, extents, orientation_map);
    Variables<tmpl::list<ScalarTensor, Coords<1>>> expected_vars(
        extents.product());
    get(get<ScalarTensor>(expected_vars)) = DataVector{{4.0, 3.0, 2.0, 1.0}};
    get<0>(get<Coords<1>>(expected_vars)) = DataVector{{0.8, 0.7, 0.6, 0.5}};
    CHECK(oriented_vars == expected_vars);
  }
}

// Test one case by hand. This test case is redundant with (though not
// identical to) one of the cases hit by `test_2d_orient_variables`. However, by
// writing it out by hand, we provide a sanity check and a clearer example of
// how to use `orient_variables`.
void test_2d_simple_case_by_hand() noexcept {
  const OrientationMap<2> orientation_map(std::array<Direction<2>, 2>{
      {Direction<2>::lower_eta(), Direction<2>::upper_xi()}});
  const auto extents = Index<2>{2, 3};

  Variables<tmpl::list<ScalarTensor>> vars(6);
  get(get<ScalarTensor>(vars)) = DataVector{{1.0, 2.0, 3.0, 4.0, 5.0, 6.0}};
  const auto oriented_vars = orient_variables(vars, extents, orientation_map);

  Variables<tmpl::list<ScalarTensor>> expected_vars(6);
  get(get<ScalarTensor>(expected_vars)) =
      DataVector{{2.0, 4.0, 6.0, 1.0, 3.0, 5.0}};
  CHECK(oriented_vars == expected_vars);
}

// Test orient_variables using a general orientation.
// The challenge for the general test is to easily create the expected oriented
// tensor, ideally without using the same algorithm that is used in
// `orient_variables`. We do this by computing the coordinates of a rectangular
// grid using two different coordinate maps and orientations of the grid,
// this provides both the input tensor and the expected output tensor.
void test_2d_with_orientation(
    const OrientationMap<2>& orientation_map) noexcept {
  const auto extents = Index<2>{3, 4};
  const auto mesh = Mesh<2>(extents.indices(), Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto affine =
      Affine2D{Affine(-1.0, 1.0, 2.3, 4.5), Affine(-1.0, 1.0, 0.8, 3.1)};
  const auto map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(affine);
  const auto map_oriented =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::DiscreteRotation<2>{orientation_map}, affine);

  Variables<tmpl::list<ScalarTensor, Coords<2>>> vars(extents.product());
  // Fill ScalarTensor with x-coordinate values
  get(get<ScalarTensor>(vars)) = get<0>(map(logical_coordinates(mesh)));
  get<Coords<2>>(vars) = map(logical_coordinates(mesh));
  const auto oriented_vars = orient_variables(vars, extents, orientation_map);

  Variables<tmpl::list<ScalarTensor, Coords<2>>> expected_vars(
      extents.product());
  get(get<ScalarTensor>(expected_vars)) = get<0>(
      map_oriented(logical_coordinates(orientation_map.inverse_map()(mesh))));
  get<Coords<2>>(expected_vars) =
      map_oriented(logical_coordinates(orientation_map.inverse_map()(mesh)));
  CHECK(oriented_vars == expected_vars);
}

void test_2d_orient_variables() noexcept {
  size_t number_of_orientations_checked = 0;
  auto dimensions = make_array(0_st, 1_st);
  do {
    CAPTURE(dimensions);
    for (const auto& side_1 : {Side::Lower, Side::Upper}) {
      const auto dir_1 = Direction<2>(dimensions[0], side_1);
      for (const auto& side_2 : {Side::Lower, Side::Upper}) {
        const auto dir_2 = Direction<2>(dimensions[1], side_2);
        const OrientationMap<2> orientation_map(
            std::array<Direction<2>, 2>{{dir_1, dir_2}});
        CAPTURE(orientation_map);
        test_2d_with_orientation(orientation_map);
        number_of_orientations_checked++;
      }
    }
  } while (std::next_permutation(dimensions.begin(), dimensions.end()));
  CHECK(number_of_orientations_checked == 8);
}

// Test one case by hand. This test case is redundant with (though not
// identical to) one of the cases hit by `test_3d_orient_variables`. However, by
// writing it out by hand, we provide a sanity check and a clearer example of
// how to use `orient_variables`.
void test_3d_simple_case_by_hand() noexcept {
  const OrientationMap<3> orientation_map(std::array<Direction<3>, 3>{
      {Direction<3>::upper_zeta(), Direction<3>::upper_eta(),
       Direction<3>::upper_xi()}});
  const auto extents = Index<3>{2, 3, 4};

  Variables<tmpl::list<ScalarTensor>> vars(24);
  get(get<ScalarTensor>(vars)) = DataVector{
      {0.0,  1.0,  2.0,  3.0,  4.0,  5.0,  6.0,  7.0,  8.0,  9.0,  10.0, 11.0,
       12.0, 13.0, 14.0, 15.0, 16.0, 17.0, 18.0, 19.0, 20.0, 21.0, 22.0, 23.0}};
  const auto oriented_vars = orient_variables(vars, extents, orientation_map);

  Variables<tmpl::list<ScalarTensor>> expected_vars(24);
  get(get<ScalarTensor>(expected_vars)) = DataVector{
      {0.0, 6.0, 12.0, 18.0, 2.0, 8.0, 14.0, 20.0, 4.0, 10.0, 16.0, 22.0,
       1.0, 7.0, 13.0, 19.0, 3.0, 9.0, 15.0, 21.0, 5.0, 11.0, 17.0, 23.0}};
  CHECK(oriented_vars == expected_vars);
}

// Test orient_variables using a general orientation.
// The challenge for the general test is to easily create the expected oriented
// tensor, ideally without using the same algorithm that is used in
// `orient_variables`. We do this by computing the coordinates of a rectangular
// grid using two different coordinate maps and orientations of the grid,
// this provides both the input tensor and the expected output tensor.
void test_3d_with_orientation(
    const OrientationMap<3>& orientation_map) noexcept {
  const auto extents = Index<3>{2, 3, 4};
  const auto mesh = Mesh<3>(extents.indices(), Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto);
  const auto affine =
      Affine3D{Affine(-1.0, 1.0, 2.3, 4.5), Affine(-1.0, 1.0, 0.8, 3.1),
               Affine(-1.0, 1.0, -4.8, -3.9)};
  const auto map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(affine);
  const auto map_oriented =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::DiscreteRotation<3>{orientation_map}, affine);

  Variables<tmpl::list<ScalarTensor, Coords<3>>> vars(extents.product());
  // Fill ScalarTensor with x-coordinate values
  get(get<ScalarTensor>(vars)) = get<0>(map(logical_coordinates(mesh)));
  get<Coords<3>>(vars) = map(logical_coordinates(mesh));
  const auto oriented_vars = orient_variables(vars, extents, orientation_map);

  Variables<tmpl::list<ScalarTensor, Coords<3>>> expected_vars(
      extents.product());
  get(get<ScalarTensor>(expected_vars)) = get<0>(
      map_oriented(logical_coordinates(orientation_map.inverse_map()(mesh))));
  get<Coords<3>>(expected_vars) =
      map_oriented(logical_coordinates(orientation_map.inverse_map()(mesh)));
  CHECK(oriented_vars == expected_vars);
}

void test_3d_orient_variables() noexcept {
  size_t number_of_orientations_checked = 0;
  auto dimensions = make_array(0_st, 1_st, 2_st);
  do {
    CAPTURE(dimensions);
    for (const auto& side_1 : {Side::Lower, Side::Upper}) {
      const auto dir_1 = Direction<3>(dimensions[0], side_1);
      for (const auto& side_2 : {Side::Lower, Side::Upper}) {
        const auto dir_2 = Direction<3>(dimensions[1], side_2);
        for (const auto& side_3 : {Side::Lower, Side::Upper}) {
          const auto dir_3 = Direction<3>(dimensions[2], side_3);
          const OrientationMap<3> orientation_map(
              std::array<Direction<3>, 3>{{dir_1, dir_2, dir_3}});
          CAPTURE(orientation_map);
          test_3d_with_orientation(orientation_map);
          number_of_orientations_checked++;
        }
      }
    }
  } while (std::next_permutation(dimensions.begin(), dimensions.end()));
  CHECK(number_of_orientations_checked == 48);
}

// Test 0D slice of a 1D element
void test_0d_orient_variables_on_slice() noexcept {
  const Index<0> slice_extents{1};
  Variables<tmpl::list<ScalarTensor, Coords<1>>> vars(slice_extents.product());
  get(get<ScalarTensor>(vars)) = DataVector{{-0.5}};
  get<0>(get<Coords<1>>(vars)) = DataVector{{1.0}};
  for (const auto& side : {Side::Lower, Side::Upper}) {
    CAPTURE(side);
    const OrientationMap<1> orientation_map(
        std::array<Direction<1>, 1>{{Direction<1>(0, side)}});
    const auto oriented_vars =
        orient_variables_on_slice(vars, slice_extents, 0, orientation_map);
    // 1D boundary is a point, so no change expected
    CHECK(oriented_vars == vars);
  }
}

void test_1d_slice_with_orientation(
    const OrientationMap<2>& orientation_map) noexcept {
  const auto slice_extents = Index<1>{4};
  const auto slice_mesh =
      Mesh<1>(slice_extents.indices(), Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
  const auto affine = Affine(-1.0, 1.0, 2.3, 4.5);
  const auto map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(affine);

  for (const size_t sliced_dim : {0_st, 1_st}) {
    CAPTURE(sliced_dim);
    // Because `orientation_map` transforms between directions in the volume, we
    // make a new OrientationMap that transforms between directions on the
    // slices. In the case of a 1D slice, this is a 1D OrientationMap.
    const size_t remaining_dim = 1 - sliced_dim;
    const auto slice_orientation_map =
        OrientationMap<1>(std::array<Direction<1>, 1>({{Direction<1>(
            0, orientation_map(Direction<2>(remaining_dim, Side::Upper))
                   .side())}}));
    const auto map_oriented =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
            domain::CoordinateMaps::DiscreteRotation<1>{slice_orientation_map},
            affine);

    Variables<tmpl::list<ScalarTensor, Coords<1>>> vars(
        slice_extents.product());
    // Fill ScalarTensor with x-coordinate values
    get(get<ScalarTensor>(vars)) = get<0>(map(logical_coordinates(slice_mesh)));
    get<Coords<1>>(vars) = map(logical_coordinates(slice_mesh));
    const auto oriented_vars = orient_variables_on_slice(
        vars, slice_extents, sliced_dim, orientation_map);

    Variables<tmpl::list<ScalarTensor, Coords<1>>> expected_vars(
        slice_extents.product());
    get(get<ScalarTensor>(expected_vars)) = get<0>(map_oriented(
        logical_coordinates(slice_orientation_map.inverse_map()(slice_mesh))));
    get<Coords<1>>(expected_vars) = map_oriented(
        logical_coordinates(slice_orientation_map.inverse_map()(slice_mesh)));
    CHECK(oriented_vars == expected_vars);
  }
}

// Test 1D slice of a 2D element
void test_1d_orient_variables_on_slice() noexcept {
  size_t number_of_orientations_checked = 0;
  auto dimensions = make_array(0_st, 1_st);
  do {
    CAPTURE(dimensions);
    for (const auto& side_1 : {Side::Lower, Side::Upper}) {
      const auto dir_1 = Direction<2>(dimensions[0], side_1);
      for (const auto& side_2 : {Side::Lower, Side::Upper}) {
        const auto dir_2 = Direction<2>(dimensions[1], side_2);
        const OrientationMap<2> orientation_map(
            std::array<Direction<2>, 2>{{dir_1, dir_2}});
        CAPTURE(orientation_map);
        test_1d_slice_with_orientation(orientation_map);
        number_of_orientations_checked++;
      }
    }
  } while (std::next_permutation(dimensions.begin(), dimensions.end()));
  CHECK(number_of_orientations_checked == 8);
}

void test_2d_slice_with_orientation(
    const OrientationMap<3>& orientation_map) noexcept {
  const auto slice_extents = Index<2>{3, 4};
  const auto slice_mesh =
      Mesh<2>(slice_extents.indices(), Spectral::Basis::Legendre,
              Spectral::Quadrature::GaussLobatto);
  const auto affine =
      Affine2D{Affine(-1.0, 1.0, 2.3, 4.5), Affine(-1.0, 1.0, 0.8, 3.1)};
  const auto map =
      domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(affine);

  for (const size_t sliced_dim : {0_st, 1_st, 2_st}) {
    CAPTURE(sliced_dim);

    // Because `orientation_map` transforms between directions in the volume, we
    // make a new OrientationMap that transforms between directions on the
    // slices. In the case of a 2D slice, this is a 2D OrientationMap.
    const auto slice_orientation_map =
        [&orientation_map, &sliced_dim ]() noexcept {
      const auto dims_of_slice =
          (sliced_dim == 0 ? make_array(1_st, 2_st)
                           : (sliced_dim == 1 ? make_array(0_st, 2_st)
                                              : make_array(0_st, 1_st)));
      const auto neighbor_dims_of_slice = make_array(
          orientation_map(dims_of_slice[0]), orientation_map(dims_of_slice[1]));
      const bool neighbor_axes_transposed =
          (neighbor_dims_of_slice[1] < neighbor_dims_of_slice[0]);
      const size_t neighbor_first_slice_dim = neighbor_axes_transposed ? 1 : 0;
      const size_t neighbor_second_slice_dim = neighbor_axes_transposed ? 0 : 1;

      return OrientationMap<2>(std::array<Direction<2>, 2>(
          {{Direction<2>(
                neighbor_first_slice_dim,
                orientation_map(Direction<3>(dims_of_slice[0], Side::Upper))
                    .side()),
            Direction<2>(
                neighbor_second_slice_dim,
                orientation_map(Direction<3>(dims_of_slice[1], Side::Upper))
                    .side())}}));
    }
    ();

    const auto map_oriented =
        domain::make_coordinate_map<Frame::Logical, Frame::Inertial>(
            domain::CoordinateMaps::DiscreteRotation<2>{slice_orientation_map},
            affine);

    Variables<tmpl::list<ScalarTensor, Coords<2>>> vars(
        slice_extents.product());
    // Fill ScalarTensor with x-coordinate values
    get(get<ScalarTensor>(vars)) = get<0>(map(logical_coordinates(slice_mesh)));
    get<Coords<2>>(vars) = map(logical_coordinates(slice_mesh));
    const auto oriented_vars = orient_variables_on_slice(
        vars, slice_extents, sliced_dim, orientation_map);

    Variables<tmpl::list<ScalarTensor, Coords<2>>> expected_vars(
        slice_extents.product());
    get(get<ScalarTensor>(expected_vars)) = get<0>(map_oriented(
        logical_coordinates(slice_orientation_map.inverse_map()(slice_mesh))));
    get<Coords<2>>(expected_vars) = map_oriented(
        logical_coordinates(slice_orientation_map.inverse_map()(slice_mesh)));
    CHECK(oriented_vars == expected_vars);
  }
}

// Test 2D slice of a 3D element
void test_2d_orient_variables_on_slice() noexcept {
  size_t number_of_orientations_checked = 0;
  auto dimensions = make_array(0_st, 1_st, 2_st);
  do {
    CAPTURE(dimensions);
    for (const auto& side_1 : {Side::Lower, Side::Upper}) {
      const auto dir_1 = Direction<3>(dimensions[0], side_1);
      for (const auto& side_2 : {Side::Lower, Side::Upper}) {
        const auto dir_2 = Direction<3>(dimensions[1], side_2);
        for (const auto& side_3 : {Side::Lower, Side::Upper}) {
          const auto dir_3 = Direction<3>(dimensions[2], side_3);
          const OrientationMap<3> orientation_map(
              std::array<Direction<3>, 3>{{dir_1, dir_2, dir_3}});
          CAPTURE(orientation_map);
          test_2d_slice_with_orientation(orientation_map);
          number_of_orientations_checked++;
        }
      }
    }
  } while (std::next_permutation(dimensions.begin(), dimensions.end()));
  CHECK(number_of_orientations_checked == 48);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Variables.OrientVariables",
                  "[DataStructures][Unit]") {
  SECTION("Testing orient_variables") {
    test_1d_orient_variables();
    test_2d_orient_variables();
    test_3d_orient_variables();
    test_2d_simple_case_by_hand();
    test_3d_simple_case_by_hand();
  }

  SECTION("Testing orient_variables_on_slice") {
    // Each of these tests closely follows the corresponding volume test
    // e.g., test_1d_orient_variables_on_slice -> test_2d_orient_variables
    test_0d_orient_variables_on_slice();
    test_1d_orient_variables_on_slice();
    test_2d_orient_variables_on_slice();
  }
}
