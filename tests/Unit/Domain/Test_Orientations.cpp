// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>

#include "Domain/Direction.hpp"
#include "Domain/Orientation.hpp"
#include "Utilities/StdHelpers.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

void test_1d() {
  // Test constructors:
  Orientation<1> custom1(
      std::array<Direction<1>, 1>{{Direction<1>::upper_xi()}});
  CHECK(custom1.is_aligned() == true);
  Orientation<1> custom2(
      std::array<Direction<1>, 1>{{Direction<1>::lower_xi()}});
  CHECK(custom2.is_aligned() == false);

  // Test if Orientation can encode a 1D parallel/ antiparallel.
  std::array<Direction<1>, 1> block1_directions{{Direction<1>::upper_xi()}};
  std::array<Direction<1>, 1> block2_directions{{Direction<1>::lower_xi()}};
  Orientation<1> parallel_orientation(block1_directions, block1_directions);
  Orientation<1> antiparallel_orientation(block1_directions, block2_directions);
  std::array<SegmentId, 1> segment_ids{{SegmentId(2, 1)}};
  std::array<SegmentId, 1> expected_antiparallel_segment_ids{{SegmentId(2, 2)}};
  CHECK(parallel_orientation.mapped(segment_ids) == segment_ids);
  CHECK(antiparallel_orientation.mapped(segment_ids) ==
        expected_antiparallel_segment_ids);
  CHECK(get_output(parallel_orientation) == "(+0)");
  CHECK(get_output(antiparallel_orientation) == "(-0)");

  // Test comparison:
  CHECK(custom1 != custom2);
  CHECK(custom1 == parallel_orientation);

  // Test semantics:
  const auto custom_copy = custom1;
  test_copy_semantics(custom2);
  test_move_semantics(std::move(custom1), custom_copy);

  // Test serialization:
  serialize_and_deserialize(custom2);
}

void test_2d() {
  // Test constructors:
  Orientation<2> custom1(std::array<Direction<2>, 2>{
      {Direction<2>::upper_xi(), Direction<2>::upper_eta()}});
  CHECK(custom1.is_aligned() == true);
  Orientation<2> custom2(std::array<Direction<2>, 2>{
      {Direction<2>::lower_xi(), Direction<2>::lower_eta()}});
  CHECK(custom2.is_aligned() == false);

  // Test if Orientation can encode a 2D rotated.
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
  Orientation<2> rotated2d_neg_xi_neg_eta(block_directions2, block_directions1);

  std::array<Direction<2>, 2> block_directions3{{upper_xi, upper_eta}};
  std::array<Direction<2>, 2> block_directions4{{lower_eta, upper_xi}};
  Orientation<2> rotated2d_neg_eta_pos_xi(block_directions4, block_directions3);

  std::array<Direction<2>, 2> block_directions5{{upper_xi, upper_eta}};
  std::array<Direction<2>, 2> block_directions6{{upper_eta, lower_xi}};
  Orientation<2> rotated2d_pos_eta_neg_xi(block_directions6, block_directions5);

  std::array<Direction<2>, 2> block_directions7{{upper_xi, upper_eta}};
  std::array<Direction<2>, 2> block_directions8{{upper_xi, upper_eta}};
  Orientation<2> rotated2d_pos_xi_pos_eta(block_directions7, block_directions8);

  std::array<SegmentId, 2> segment_ids{{SegmentId(2, 1), SegmentId(3, 5)}};

  std::array<SegmentId, 2> expected_neg_xi_neg_eta_segment_ids{
      {SegmentId(2, 2), SegmentId(3, 2)}};
  std::array<SegmentId, 2> expected_neg_eta_pos_xi_segment_ids{
      {SegmentId(3, 2), SegmentId(2, 1)}};
  std::array<SegmentId, 2> expected_pos_eta_neg_xi_segment_ids{
      {SegmentId(3, 5), SegmentId(2, 2)}};

  // Check mapped(size_t dimension) function
  CHECK(rotated2d_neg_xi_neg_eta.mapped(0) == 0);
  CHECK(rotated2d_neg_xi_neg_eta.mapped(1) == 1);
  CHECK(rotated2d_neg_eta_pos_xi.mapped(0) == 1);
  CHECK(rotated2d_neg_eta_pos_xi.mapped(1) == 0);
  CHECK(rotated2d_pos_eta_neg_xi.mapped(0) == 1);
  CHECK(rotated2d_pos_eta_neg_xi.mapped(1) == 0);

  // Check mapped(Direction<2> direction function)
  CHECK(rotated2d_neg_xi_neg_eta.mapped(upper_xi) == lower_xi);
  CHECK(rotated2d_neg_xi_neg_eta.mapped(upper_eta) == lower_eta);
  CHECK(rotated2d_neg_xi_neg_eta.mapped(lower_xi) == upper_xi);
  CHECK(rotated2d_neg_xi_neg_eta.mapped(lower_eta) == upper_eta);

  CHECK(rotated2d_neg_eta_pos_xi.mapped(upper_xi) == upper_eta);
  CHECK(rotated2d_neg_eta_pos_xi.mapped(upper_eta) == lower_xi);
  CHECK(rotated2d_neg_eta_pos_xi.mapped(lower_xi) == lower_eta);
  CHECK(rotated2d_neg_eta_pos_xi.mapped(lower_eta) == upper_xi);
  CHECK(rotated2d_pos_eta_neg_xi.mapped(upper_xi) == lower_eta);
  CHECK(rotated2d_pos_eta_neg_xi.mapped(upper_eta) == upper_xi);
  CHECK(rotated2d_pos_eta_neg_xi.mapped(lower_xi) == upper_eta);
  CHECK(rotated2d_pos_eta_neg_xi.mapped(lower_eta) == lower_xi);

  // Check mapped(std::array<SegmentIds, VolumeDim> segment_ids)
  CHECK(rotated2d_neg_eta_pos_xi.mapped(segment_ids) ==
        expected_neg_eta_pos_xi_segment_ids);
  CHECK(rotated2d_neg_xi_neg_eta.mapped(segment_ids) ==
        expected_neg_xi_neg_eta_segment_ids);
  CHECK(rotated2d_pos_eta_neg_xi.mapped(segment_ids) ==
        expected_pos_eta_neg_xi_segment_ids);
  CHECK(rotated2d_neg_eta_pos_xi.is_aligned() == false);
  CHECK(rotated2d_neg_xi_neg_eta.is_aligned() == false);
  CHECK(rotated2d_pos_eta_neg_xi.is_aligned() == false);
  CHECK(rotated2d_pos_xi_pos_eta.is_aligned() == true);

  // The naming convention used in this test:
  //"neg_eta_pos_xi" means that -1 in the host maps to +0,
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
  test_move_semantics(std::move(rotated2d_neg_eta_pos_xi), rotated_copy);

  // Test serialization:
  serialize_and_deserialize(rotated_copy);
}

void test_3d() {
  // Test constructors:
  Orientation<3> custom1(std::array<Direction<3>, 3>{
      {Direction<3>::upper_xi(), Direction<3>::upper_eta(),
       Direction<3>::upper_zeta()}});
  CHECK(custom1.is_aligned() == true);
  Orientation<3> custom2(std::array<Direction<3>, 3>{
      {Direction<3>::lower_xi(), Direction<3>::lower_eta(),
       Direction<3>::lower_zeta()}});
  CHECK(custom2.is_aligned() == false);

  // Test if Orientation can encode a 3D Flipped.
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
  Orientation<3> custom_orientation(block_directions1, block_directions2);
  CHECK(custom_orientation.mapped(upper_xi) == lower_xi);
  CHECK(custom_orientation.mapped(upper_eta) == lower_eta);
  CHECK(custom_orientation.mapped(upper_zeta) == lower_zeta);

  Direction<3> direction(Direction<3>::Axis::Zeta, Side::Upper);
  std::array<SegmentId, 3> segment_ids{
      {SegmentId(2, 1), SegmentId(3, 1), SegmentId(3, 3)}};

  std::array<SegmentId, 3> flipped_ids{
      {SegmentId(2, 2), SegmentId(3, 6), SegmentId(3, 4)}};
  CHECK(custom_orientation.mapped(2) == 2);
  CHECK(custom_orientation.mapped(direction) == direction.opposite());
  CHECK(custom_orientation.mapped(segment_ids) == flipped_ids);
  CHECK(custom_orientation.is_aligned() == false);
  Orientation<3> aligned_orientation(block_directions1, block_directions1);
  CHECK(aligned_orientation.is_aligned() == true);
  CHECK(get_output(custom_orientation) == "(-0, -1, -2)");
  CHECK(get_output(aligned_orientation) == "(+0, +1, +2)");

  // Test comparison operators:
  CHECK(custom_orientation != aligned_orientation);
  CHECK(custom_orientation == custom_orientation);

  // Test semantics:
  const auto custom_copy = custom_orientation;
  test_copy_semantics(aligned_orientation);
  test_move_semantics(std::move(custom_orientation), custom_copy);

  // Test serialzation:
  serialize_and_deserialize(custom2);
}

}  // namespace

TEST_CASE("Unit.Domain.Orientations", "[Domain][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}
