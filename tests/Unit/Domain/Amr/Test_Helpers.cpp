// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/rational.hpp>
#include <cstddef>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/OrientationMap.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"

namespace {
void test_desired_refinement_levels() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(amr::domain::desired_refinement_levels(element_id_1d,
                                               {{amr::domain::Flag::Split}}) ==
        std::array<size_t, 1>{{3}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_1d, {{amr::domain::Flag::DoNothing}}) ==
        std::array<size_t, 1>{{2}});
  CHECK(amr::domain::desired_refinement_levels(element_id_1d,
                                               {{amr::domain::Flag::Join}}) ==
        std::array<size_t, 1>{{1}});

  const ElementId<2> element_id_2d{1, {{SegmentId(3, 5), SegmentId(1, 1)}}};
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::Split, amr::domain::Flag::Join}}) ==
        std::array<size_t, 2>{{4, 0}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::Join, amr::domain::Flag::DoNothing}}) ==
        std::array<size_t, 2>{{2, 1}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::Join, amr::domain::Flag::Join}}) ==
        std::array<size_t, 2>{{2, 0}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::DoNothing, amr::domain::Flag::Split}}) ==
        std::array<size_t, 2>{{3, 2}});
  CHECK(amr::domain::desired_refinement_levels(
            element_id_2d,
            {{amr::domain::Flag::DoNothing, amr::domain::Flag::DoNothing}}) ==
        std::array<size_t, 2>{{3, 1}});

  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 15), SegmentId(2, 0), SegmentId(4, 6)}}};
  CHECK(amr::domain::desired_refinement_levels(
            element_id_3d, {{amr::domain::Flag::Split, amr::domain::Flag::Join,
                             amr::domain::Flag::DoNothing}}) ==
        std::array<size_t, 3>{{6, 1, 4}});
}

template <size_t VolumeDim>
void check_desired_refinement_levels_of_neighbor(
    const ElementId<VolumeDim>& neighbor_id,
    const std::array<amr::domain::Flag, VolumeDim>& neighbor_flags) {
  for (OrientationMapIterator<VolumeDim> orientation{}; orientation;
       ++orientation) {
    const auto desired_levels_my_frame = desired_refinement_levels_of_neighbor(
        neighbor_id, neighbor_flags, *orientation);
    const auto desired_levels_neighbor_frame =
        desired_refinement_levels(neighbor_id, neighbor_flags);
    for (size_t d = 0; d < VolumeDim; ++d) {
      CHECK(gsl::at(desired_levels_my_frame, d) ==
            gsl::at(desired_levels_neighbor_frame, (*orientation)(d)));
    }
  }
}

void test_desired_refinement_levels_of_neighbor() {
  const ElementId<1> neighbor_id_1d{0, {{SegmentId(2, 3)}}};
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{amr::domain::Flag::Split}});
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{amr::domain::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{amr::domain::Flag::Join}});

  const ElementId<2> neighbor_id_2d{1, {{SegmentId(3, 0), SegmentId(1, 1)}}};
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d, {{amr::domain::Flag::Split, amr::domain::Flag::Join}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{amr::domain::Flag::Join, amr::domain::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d, {{amr::domain::Flag::Join, amr::domain::Flag::Join}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{amr::domain::Flag::DoNothing, amr::domain::Flag::Split}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{amr::domain::Flag::DoNothing, amr::domain::Flag::DoNothing}});

  const ElementId<3> neighbor_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_3d, {{amr::domain::Flag::Split, amr::domain::Flag::Join,
                        amr::domain::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_3d, {{amr::domain::Flag::Join, amr::domain::Flag::DoNothing,
                        amr::domain::Flag::Split}});
}

void test_fraction_of_block_volume() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(fraction_of_block_volume(element_id_1d) ==
        boost::rational<size_t>(1, 4));
  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  CHECK(fraction_of_block_volume(element_id_2d) ==
        boost::rational<size_t>(1, 16));
  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  CHECK(fraction_of_block_volume(element_id_3d) ==
        boost::rational<size_t>(1, 2048));
}

void test_has_potential_sibling() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(amr::domain::has_potential_sibling(element_id_1d,
                                           Direction<1>::lower_xi()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_1d,
                                                 Direction<1>::upper_xi()));

  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  CHECK(amr::domain::has_potential_sibling(element_id_2d,
                                           Direction<2>::upper_xi()));
  CHECK(amr::domain::has_potential_sibling(element_id_2d,
                                           Direction<2>::lower_eta()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_2d,
                                                 Direction<2>::lower_xi()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_2d,
                                                 Direction<2>::upper_eta()));

  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  CHECK(amr::domain::has_potential_sibling(element_id_3d,
                                           Direction<3>::lower_xi()));
  CHECK(amr::domain::has_potential_sibling(element_id_3d,
                                           Direction<3>::upper_eta()));
  CHECK(amr::domain::has_potential_sibling(element_id_3d,
                                           Direction<3>::lower_zeta()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_3d,
                                                 Direction<3>::upper_xi()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_3d,
                                                 Direction<3>::lower_eta()));
  CHECK_FALSE(amr::domain::has_potential_sibling(element_id_3d,
                                                 Direction<3>::upper_zeta()));
}

void test_id_of_parent() {
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  CHECK(amr::domain::id_of_parent(element_id_1d,
                                  std::array{amr::domain::Flag::Join}) ==
        ElementId<1>{0, {{SegmentId(1, 1)}}});
  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  CHECK(amr::domain::id_of_parent(
            element_id_2d,
            std::array{amr::domain::Flag::Join, amr::domain::Flag::Join}) ==
        ElementId<2>{0, {{SegmentId(2, 0), SegmentId(0, 0)}}});
  CHECK(amr::domain::id_of_parent(element_id_2d,
                                  std::array{amr::domain::Flag::DoNothing,
                                             amr::domain::Flag::Join}) ==
        ElementId<2>{0, {{SegmentId(3, 0), SegmentId(0, 0)}}});
  CHECK(amr::domain::id_of_parent(element_id_2d,
                                  std::array{amr::domain::Flag::Join,
                                             amr::domain::Flag::DoNothing}) ==
        ElementId<2>{0, {{SegmentId(2, 0), SegmentId(1, 1)}}});
  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::Join, amr::domain::Flag::Join,
                     amr::domain::Flag::Join}) ==
      ElementId<3>{7, {{SegmentId(4, 15), SegmentId(1, 0), SegmentId(3, 7)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::Join, amr::domain::Flag::Join,
                     amr::domain::Flag::DoNothing}) ==
      ElementId<3>{7, {{SegmentId(4, 15), SegmentId(1, 0), SegmentId(4, 15)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::Join, amr::domain::Flag::DoNothing,
                     amr::domain::Flag::Join}) ==
      ElementId<3>{7, {{SegmentId(4, 15), SegmentId(2, 0), SegmentId(3, 7)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::DoNothing, amr::domain::Flag::Join,
                     amr::domain::Flag::Join}) ==
      ElementId<3>{7, {{SegmentId(5, 31), SegmentId(1, 0), SegmentId(3, 7)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::Join, amr::domain::Flag::DoNothing,
                     amr::domain::Flag::DoNothing}) ==
      ElementId<3>{7, {{SegmentId(4, 15), SegmentId(2, 0), SegmentId(4, 15)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::DoNothing, amr::domain::Flag::Join,
                     amr::domain::Flag::DoNothing}) ==
      ElementId<3>{7, {{SegmentId(5, 31), SegmentId(1, 0), SegmentId(4, 15)}}});
  CHECK(
      amr::domain::id_of_parent(
          element_id_3d,
          std::array{amr::domain::Flag::DoNothing, amr::domain::Flag::DoNothing,
                     amr::domain::Flag::Join}) ==
      ElementId<3>{7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(3, 7)}}});
}

void test_assertions() {
#ifdef SPECTRE_DEBUG
  const ElementId<1> element_id_1d{0, {{SegmentId(2, 3)}}};
  const ElementId<2> element_id_2d{0, {{SegmentId(3, 0), SegmentId{1, 1}}}};
  const ElementId<3> element_id_3d{
      7, {{SegmentId(5, 31), SegmentId(2, 0), SegmentId(4, 15)}}};
  const std::array flags_1d_undefined{amr::domain::Flag::Undefined};
  const std::array flags_2d_no_join{amr::domain::Flag::DoNothing,
                                    amr::domain::Flag::Split};
  const std::array flags_3d_split_join{amr::domain::Flag::DoNothing,
                                       amr::domain::Flag::Split,
                                       amr::domain::Flag::Join};
  CHECK_THROWS_WITH(
      amr::domain::desired_refinement_levels(element_id_1d, flags_1d_undefined),
      Catch::Contains("Undefined Flag in dimension"));
  CHECK_THROWS_WITH(amr::domain::desired_refinement_levels_of_neighbor(
                        element_id_1d, flags_1d_undefined,
                        OrientationMap<1>{{{Direction<1>::lower_xi()}}}),
                    Catch::Contains("Undefined Flag in dimension"));
  CHECK_THROWS_WITH(amr::domain::id_of_parent(element_id_2d, flags_2d_no_join),
                    Catch::Contains("is not joining given flags"));
  CHECK_THROWS_WITH(
      amr::domain::id_of_parent(element_id_3d, flags_3d_split_join),
      Catch::Contains("Splitting and joining an Element is not supported"));
#endif
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Amr.Helpers", "[Domain][Unit]") {
  test_desired_refinement_levels();
  test_desired_refinement_levels_of_neighbor();
  test_fraction_of_block_volume();
  test_has_potential_sibling();
  test_id_of_parent();
  test_assertions();
}
