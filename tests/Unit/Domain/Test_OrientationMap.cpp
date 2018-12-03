// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <functional>
#include <string>
#include <type_traits>

#include "DataStructures/DataVector.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/DereferenceWrapper.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"  // IWYU pragma: keep
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

void test_1d() {
  // Test constructors:
  OrientationMap<1> default_orientation{};
  CHECK(default_orientation.is_aligned());
  CHECK(get_output(default_orientation) == "(+0)");
  OrientationMap<1> custom1(
      std::array<Direction<1>, 1>{{Direction<1>::upper_xi()}});
  CHECK(custom1.is_aligned());
  OrientationMap<1> custom2(
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
  CHECK_FALSE(custom2.is_aligned());

  // Test if OrientationMap can encode a 1D parallel/antiparallel.
  std::array<Direction<1>, 1> block1_directions{{Direction<1>::upper_xi()}};
  std::array<Direction<1>, 1> block2_directions{{Direction<1>::lower_xi()}};
  OrientationMap<1> parallel_orientation(block1_directions, block1_directions);
  OrientationMap<1> antiparallel_orientation(block1_directions,
                                             block2_directions);
  std::array<SegmentId, 1> segment_ids{{SegmentId(2, 1)}};
  std::array<SegmentId, 1> expected_antiparallel_segment_ids{{SegmentId(2, 2)}};
  CHECK(parallel_orientation(segment_ids) == segment_ids);
  CHECK(antiparallel_orientation(segment_ids) ==
        expected_antiparallel_segment_ids);
  const Mesh<1> mesh(4, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);
  CHECK(parallel_orientation(mesh) == mesh);
  CHECK(antiparallel_orientation(mesh) == mesh);
  CHECK(std::array<int, 1>{{1}} ==
        parallel_orientation.permute_from_neighbor(std::array<int, 1>{{1}}));
  CHECK(
      std::array<int, 1>{{1}} ==
      antiparallel_orientation.permute_from_neighbor(std::array<int, 1>{{1}}));

  CHECK(get_output(parallel_orientation) == "(+0)");
  CHECK(get_output(antiparallel_orientation) == "(-0)");

  // Test comparison:
  CHECK(custom1 != custom2);
  CHECK(custom1 == parallel_orientation);

  // Test semantics:
  const auto custom_copy = custom1;
  test_copy_semantics(custom2);
  // clang-tidy: std::move of trivially-copyable type has no effect.
  test_move_semantics(std::move(custom1), custom_copy);  // NOLINT

  // Test serialization:
  test_serialization(custom2);

  // Test inverse:
  CHECK(default_orientation.inverse_map() == default_orientation);
  CHECK(custom2.inverse_map() == custom2);
}

void test_2d() {
  // Test constructors:
  OrientationMap<2> default_orientation{};
  CHECK(default_orientation.is_aligned());
  CHECK(get_output(default_orientation) == "(+0, +1)");
  OrientationMap<2> custom1(std::array<Direction<2>, 2>{
      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}});
  CHECK(custom1.is_aligned());
  OrientationMap<2> custom2(std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}});
  CHECK_FALSE(custom2.is_aligned());

  // Test if OrientationMap can encode a 2D rotated.
  const auto& upper_xi = Direction<2>::upper_xi();
  const auto& upper_eta = Direction<2>::upper_eta();
  const auto& lower_xi = Direction<2>::lower_xi();
  const auto& lower_eta = Direction<2>::lower_eta();

  // Note: the naming convention used here gives the directions in
  // the host block which map to the pos_xi and pos_eta directions
  // in the neighbor block, respectively.
  // For example, "rotated2d_neg_eta_neg_xi"would indicate that the neg_eta
  // direction in the host block maps to the pos_xi direction in the
  // neighbor block, and neg_xi direction in the host block maps to the
  // pos_eta direction in the neighbor block.
  std::array<Direction<2>, 2> block_directions1{{upper_xi, upper_eta}};
  std::array<Direction<2>, 2> block_directions2{{lower_xi, lower_eta}};
  OrientationMap<2> rotated2d_neg_xi_neg_eta(block_directions2,
                                             block_directions1);

  std::array<Direction<2>, 2> block_directions3{{upper_xi, upper_eta}};
  std::array<Direction<2>, 2> block_directions4{{lower_eta, upper_xi}};
  OrientationMap<2> rotated2d_neg_eta_pos_xi(block_directions4,
                                             block_directions3);

  std::array<Direction<2>, 2> block_directions5{{upper_xi, upper_eta}};
  std::array<Direction<2>, 2> block_directions6{{upper_eta, lower_xi}};
  OrientationMap<2> rotated2d_pos_eta_neg_xi(block_directions6,
                                             block_directions5);

  std::array<Direction<2>, 2> block_directions7{{upper_xi, upper_eta}};
  std::array<Direction<2>, 2> block_directions8{{upper_xi, upper_eta}};
  OrientationMap<2> rotated2d_pos_xi_pos_eta(block_directions7,
                                             block_directions8);

  std::array<SegmentId, 2> segment_ids{{SegmentId(2, 1), SegmentId(3, 5)}};

  std::array<SegmentId, 2> expected_neg_xi_neg_eta_segment_ids{
      {SegmentId(2, 2), SegmentId(3, 2)}};
  std::array<SegmentId, 2> expected_neg_eta_pos_xi_segment_ids{
      {SegmentId(3, 2), SegmentId(2, 1)}};
  std::array<SegmentId, 2> expected_pos_eta_neg_xi_segment_ids{
      {SegmentId(3, 5), SegmentId(2, 2)}};

  CHECK_FALSE(rotated2d_neg_eta_pos_xi.is_aligned());
  CHECK_FALSE(rotated2d_neg_xi_neg_eta.is_aligned());
  CHECK_FALSE(rotated2d_pos_eta_neg_xi.is_aligned());
  CHECK(rotated2d_pos_xi_pos_eta.is_aligned());

  // Check mapped(size_t dimension) function
  CHECK(rotated2d_neg_xi_neg_eta(0) == 0);
  CHECK(rotated2d_neg_xi_neg_eta(1) == 1);
  CHECK(rotated2d_neg_eta_pos_xi(0) == 1);
  CHECK(rotated2d_neg_eta_pos_xi(1) == 0);
  CHECK(rotated2d_pos_eta_neg_xi(0) == 1);
  CHECK(rotated2d_pos_eta_neg_xi(1) == 0);

  // Check mapped(Direction<2> direction) function
  CHECK(rotated2d_neg_xi_neg_eta(upper_xi) == lower_xi);
  CHECK(rotated2d_neg_xi_neg_eta(upper_eta) == lower_eta);
  CHECK(rotated2d_neg_xi_neg_eta(lower_xi) == upper_xi);
  CHECK(rotated2d_neg_xi_neg_eta(lower_eta) == upper_eta);

  CHECK(rotated2d_neg_eta_pos_xi(upper_xi) == upper_eta);
  CHECK(rotated2d_neg_eta_pos_xi(upper_eta) == lower_xi);
  CHECK(rotated2d_neg_eta_pos_xi(lower_xi) == lower_eta);
  CHECK(rotated2d_neg_eta_pos_xi(lower_eta) == upper_xi);
  CHECK(rotated2d_pos_eta_neg_xi(upper_xi) == lower_eta);
  CHECK(rotated2d_pos_eta_neg_xi(upper_eta) == upper_xi);
  CHECK(rotated2d_pos_eta_neg_xi(lower_xi) == upper_eta);
  CHECK(rotated2d_pos_eta_neg_xi(lower_eta) == lower_xi);

  // Check mapped(std::array<SegmentIds, VolumeDim> segment_ids)
  CHECK(rotated2d_neg_eta_pos_xi(segment_ids) ==
        expected_neg_eta_pos_xi_segment_ids);
  CHECK(rotated2d_neg_xi_neg_eta(segment_ids) ==
        expected_neg_xi_neg_eta_segment_ids);
  CHECK(rotated2d_pos_eta_neg_xi(segment_ids) ==
        expected_pos_eta_neg_xi_segment_ids);

  // Check mapped(Mesh<2> mesh)
  const Mesh<2> input_mesh(
      {{3, 4}}, {{Spectral::Basis::Legendre, Spectral::Basis::Chebyshev}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}});
  const Mesh<2> flipped_mesh(
      {{4, 3}}, {{Spectral::Basis::Chebyshev, Spectral::Basis::Legendre}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}});
  CHECK(rotated2d_pos_xi_pos_eta(input_mesh) == input_mesh);
  CHECK(rotated2d_neg_xi_neg_eta(input_mesh) == input_mesh);
  CHECK(rotated2d_neg_eta_pos_xi(input_mesh) == flipped_mesh);
  CHECK(rotated2d_pos_eta_neg_xi(input_mesh) == flipped_mesh);

  // Check permute_from_neighbor(std::array<T, 2> array)
  const std::array<int, 2> input_array{{1, -3}};
  const std::array<int, 2> flipped_array{{-3, 1}};
  CHECK(rotated2d_neg_xi_neg_eta.permute_from_neighbor(input_array) ==
        input_array);
  CHECK(rotated2d_neg_eta_pos_xi.permute_from_neighbor(input_array) ==
        flipped_array);
  CHECK(rotated2d_pos_eta_neg_xi.permute_from_neighbor(input_array) ==
        flipped_array);

  // The naming convention used in this test:
  // "neg_eta_pos_xi" means that -1 in the host maps to +0,
  // and that +0 in the host maps to +1, in the neighbor.
  // For the output operator, the directions that correspond
  // to the +0 and +1 directions in the host are outputted.
  // This means we expect neg_eta_pos_xi to output (+1, -0).
  CHECK(get_output(rotated2d_neg_eta_pos_xi) == "(+1, -0)");
  CHECK(get_output(rotated2d_pos_xi_pos_eta) == "(+0, +1)");

  // Test comparison operators:
  CHECK(rotated2d_neg_eta_pos_xi != rotated2d_pos_eta_neg_xi);
  CHECK(rotated2d_neg_eta_pos_xi == rotated2d_neg_eta_pos_xi);

  // Test semantics:
  const auto rotated_copy = rotated2d_neg_eta_pos_xi;
  test_copy_semantics(rotated2d_pos_eta_neg_xi);
  // clang-tidy: std::move of trivially-copyable type has no effect.
  test_move_semantics(std::move(rotated2d_neg_eta_pos_xi),  // NOLINT
                      rotated_copy);

  // Test serialization:
  test_serialization(rotated_copy);

  // Test inverse:
  CHECK(default_orientation.inverse_map() == default_orientation);
  CHECK(
      OrientationMap<2>(std::array<Direction<2>, 2>{{Direction<2>::lower_eta(),
                                                     Direction<2>::upper_xi()}})
          .inverse_map() ==
      OrientationMap<2>(std::array<Direction<2>, 2>{
          {Direction<2>::upper_eta(), Direction<2>::lower_xi()}}));
  CHECK(custom1.inverse_map().inverse_map() == custom1);
  CHECK(custom2.inverse_map().inverse_map() == custom2);
}

void test_3d() {
  // Test constructors:
  OrientationMap<3> default_orientation{};
  CHECK(default_orientation.is_aligned());
  CHECK(get_output(default_orientation) == "(+0, +1, +2)");
  OrientationMap<3> custom1(std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::upper_eta(),
       Direction<3>::upper_zeta()}});
  CHECK(custom1.is_aligned());
  OrientationMap<3> custom2(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::lower_zeta()}});
  CHECK_FALSE(custom2.is_aligned());

  // Test if OrientationMap can encode a 3D Flipped.
  const auto& upper_xi = Direction<3>::upper_xi();
  const auto& upper_eta = Direction<3>::upper_eta();
  const auto& upper_zeta = Direction<3>::upper_zeta();
  const auto& lower_xi = Direction<3>::lower_xi();
  const auto& lower_eta = Direction<3>::lower_eta();
  const auto& lower_zeta = Direction<3>::lower_zeta();

  std::array<Direction<3>, 3> block_directions1{
      {upper_xi, upper_eta, upper_zeta}};
  std::array<Direction<3>, 3> block_directions2{
      {lower_xi, lower_eta, lower_zeta}};
  OrientationMap<3> custom_orientation(block_directions1, block_directions2);
  CHECK(custom_orientation(upper_xi) == lower_xi);
  CHECK(custom_orientation(upper_eta) == lower_eta);
  CHECK(custom_orientation(upper_zeta) == lower_zeta);

  Direction<3> direction(Direction<3>::Axis::Zeta, Side::Upper);
  std::array<SegmentId, 3> segment_ids{
      {SegmentId(2, 1), SegmentId(3, 1), SegmentId(3, 3)}};

  std::array<SegmentId, 3> flipped_ids{
      {SegmentId(2, 2), SegmentId(3, 6), SegmentId(3, 4)}};
  CHECK(custom_orientation(2) == 2);
  CHECK(custom_orientation(direction) == direction.opposite());
  CHECK(custom_orientation(segment_ids) == flipped_ids);
  CHECK_FALSE(custom_orientation.is_aligned());
  OrientationMap<3> aligned_orientation(block_directions1, block_directions1);
  CHECK(aligned_orientation.is_aligned());
  CHECK(get_output(custom_orientation) == "(-0, -1, -2)");
  CHECK(get_output(aligned_orientation) == "(+0, +1, +2)");

  // Test comparison operators:
  CHECK(custom_orientation != aligned_orientation);
  CHECK(custom_orientation == custom_orientation);

  // Test semantics:
  const auto custom_copy = custom_orientation;
  test_copy_semantics(aligned_orientation);
  // clang-tidy: std::move of trivially-copyable type has no effect.
  test_move_semantics(std::move(custom_orientation), custom_copy);  // NOLINT

  // Test serialzation:
  test_serialization(custom2);

  // Test inverse:
  CHECK(default_orientation.inverse_map() == default_orientation);
  OrientationMap<3> custom3{std::array<Direction<3>, 3>{
      {Direction<3>::lower_eta(), Direction<3>::lower_zeta(),
       Direction<3>::upper_xi()}}};
  OrientationMap<3> custom4{std::array<Direction<3>, 3>{
      {Direction<3>::upper_zeta(), Direction<3>::lower_xi(),
       Direction<3>::lower_eta()}}};
  CHECK(custom3.inverse_map() == custom4);
  CHECK(Mesh<3>({{4, 5, 3}}, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto) ==
        custom3(Mesh<3>({{3, 4, 5}}, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto)));
  CHECK(Mesh<3>({{5, 3, 4}}, Spectral::Basis::Legendre,
                Spectral::Quadrature::GaussLobatto) ==
        custom4(Mesh<3>({{3, 4, 5}}, Spectral::Basis::Legendre,
                        Spectral::Quadrature::GaussLobatto)));
  CHECK(std::array<int, 3>{{-8, 12, 4}} ==
        custom3.permute_from_neighbor(std::array<int, 3>{{4, -8, 12}}));
  CHECK(std::array<int, 3>{{12, 4, -8}} ==
        custom4.permute_from_neighbor(std::array<int, 3>{{4, -8, 12}}));
  CHECK(custom1.inverse_map().inverse_map() == custom1);
  CHECK(custom2.inverse_map().inverse_map() == custom2);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.OrientationMap", "[Domain][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}

// [[OutputRegex, This OrientationMap fails to map Directions one-to-one.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.OrientationMap.Bijective",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_orientationmap = OrientationMap<2>{std::array<Direction<2>, 2>{
      {Direction<2>::upper_xi(), Direction<2>::lower_xi()}}};
  static_cast<void>(failed_orientationmap);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, This OrientationMap fails to map Directions one-to-one.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.OrientationMap.BijectiveHost",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_orientationmap = OrientationMap<2>{
      std::array<Direction<2>, 2>{
          {Direction<2>::upper_xi(), Direction<2>::lower_xi()}},
      std::array<Direction<2>, 2>{
          {Direction<2>::upper_xi(), Direction<2>::upper_eta()}},
  };
  static_cast<void>(failed_orientationmap);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, This OrientationMap fails to map Directions one-to-one.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.OrientationMap.BijectiveNeighbor",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_orientationmap = OrientationMap<3>{
      std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                   Direction<3>::lower_eta(),
                                   Direction<3>::lower_zeta()}},
      std::array<Direction<3>, 3>{{Direction<3>::upper_xi(),
                                   Direction<3>::upper_eta(),
                                   Direction<3>::lower_eta()}},
  };
  static_cast<void>(failed_orientationmap);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

SPECTRE_TEST_CASE("Unit.Domain.DiscreteRotation.AllOrientations",
                  "[Domain][Unit]") {
  for (OrientationMapIterator<2> map_i{}; map_i; ++map_i) {
    const std::array<double, 2> original_point{{0.5, -2.0}};
    const std::array<double, 2> new_point =
        discrete_rotation(map_i(), original_point);
    for (size_t d = 0; d < 2; d++) {
      CHECK(gsl::at(new_point, d) ==
            (map_i()(Direction<2>{d, Side::Upper}).side() == Side::Upper
                 ? gsl::at(original_point, map_i()(d))
                 : -1.0 * gsl::at(original_point, map_i()(d))));
    }
  }
  for (OrientationMapIterator<3> map_i{}; map_i; ++map_i) {
    const std::array<double, 3> original_point{{0.5, -2.0, 1.5}};
    const std::array<double, 3> new_point =
        discrete_rotation(map_i(), original_point);
    for (size_t d = 0; d < 3; d++) {
      CHECK(gsl::at(new_point, d) ==
            (map_i()(Direction<3>{d, Side::Upper}).side() == Side::Upper
                 ? gsl::at(original_point, map_i()(d))
                 : -1.0 * gsl::at(original_point, map_i()(d))));
    }
  }
}

SPECTRE_TEST_CASE("Unit.Domain.DiscreteRotation.Rotation", "[Domain][Unit]") {
  const OrientationMap<1> rotation1(
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
  const std::array<DataVector, 1> test_points1{
      {DataVector{-1.0, 1.0, 0.7, 0.0}}};
  const std::array<DataVector, 1> expected_rotated_points1{
      {DataVector{1.0, -1.0, -0.7, 0.0}}};
  CHECK(discrete_rotation(rotation1, test_points1) == expected_rotated_points1);
  CHECK(discrete_rotation(rotation1, std::array<double, 1>{{-0.2}}) ==
        std::array<double, 1>{{0.2}});

  const OrientationMap<2> rotation2(std::array<Direction<2>, 2>{
      {Direction<2>::upper_eta(), Direction<2>::upper_xi()}});
  const std::array<DataVector, 2> test_points2{
      {DataVector{-1.0, 1.0, 0.7, 0.0}, DataVector{0.25, 1.0, -0.2, 0.0}}};
  const std::array<DataVector, 2> expected_rotated_points2{
      {DataVector{0.25, 1.0, -0.2, 0.0}, DataVector{-1.0, 1.0, 0.7, 0.0}}};
  CHECK(discrete_rotation(rotation2, test_points2) == expected_rotated_points2);
  CHECK(discrete_rotation(rotation2, std::array<double, 2>{{-1.0, 0.5}}) ==
        std::array<double, 2>{{0.5, -1.0}});

  const OrientationMap<3> rotation3(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
       Direction<3>::lower_xi()}});
  const std::array<DataVector, 3> test_points3{
      {DataVector{-1.0, 1.0, 0.7, 0.0}, DataVector{0.25, 1.0, -0.2, 0.0},
       DataVector{0.0, -0.5, 0.4, 0.0}}};
  const std::array<DataVector, 3> expected_rotated_points3{
      {DataVector{0.25, 1.0, -0.2, 0.0}, DataVector{0.0, 0.5, -0.4, 0.0},
       DataVector{1.0, -1.0, -0.7, 0.0}}};
  CHECK(discrete_rotation(rotation3, test_points3) == expected_rotated_points3);
  CHECK(discrete_rotation(rotation3, std::array<double, 3>{{-1.0, 0.5, 1.0}}) ==
        std::array<double, 3>{{0.5, -1.0, 1.0}});
}

SPECTRE_TEST_CASE("Unit.Domain.DiscreteRotation.ReferenceWrapper",
                  "[Domain][Unit]") {
  const OrientationMap<3> rotation(std::array<Direction<3>, 3>{
      {Direction<3>::upper_eta(), Direction<3>::lower_zeta(),
       Direction<3>::lower_xi()}});

  // This test will check that these points are not modified.
  DataVector x_points{-1.0, 1.0, 0.7, 0.0};
  DataVector y_points{0.25, 1.0, -0.2, 0.0};
  DataVector z_points{0.0, -0.5, 0.4, 0.0};

  // These variables are not passed to any functions;
  // they will not be modified by construction.
  // clang-tidy: local copy is never modified
  const DataVector x_points_proof = x_points;  // NOLINT
  const DataVector y_points_proof = y_points;  // NOLINT
  const DataVector z_points_proof = z_points;  // NOLINT

  // References to the points to be tested:
  const auto ref_x_points = std::cref(x_points);
  const auto ref_y_points = std::cref(y_points);
  const auto ref_z_points = std::cref(z_points);

  // Array of references to the points to be tested.
  const std::array<const std::reference_wrapper<const DataVector>, 3>
      test_points{{ref_x_points, ref_y_points, ref_z_points}};

  // The value of new_points is irrelevant to this test.
  auto new_points = discrete_rotation(rotation, test_points);
  CHECK(test_points[0] == x_points_proof);
  CHECK(test_points[1] == y_points_proof);
  CHECK(test_points[2] == z_points_proof);

  const DataVector new_pt{0.0, 0.5, -0.4, 0.0};
  new_points[0] = new_pt;
  new_points[1] = new_pt;
  new_points[2] = new_pt;

  // Check that modifying new_points does not modify the test points.
  CHECK(test_points[0] == x_points_proof);
  CHECK(test_points[1] == y_points_proof);
  CHECK(test_points[2] == z_points_proof);
}
