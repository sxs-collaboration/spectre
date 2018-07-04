// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Amr/Helpers.hpp"
#include "Domain/Direction.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/SegmentId.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/Domain/CoordinateMaps/TestMapHelpers.hpp"

namespace {
void test_desired_refinement_levels() noexcept {
  domain::ElementId<1> element_id_1d{0, {{domain::SegmentId(2, 3)}}};
  CHECK(domain::amr::desired_refinement_levels(element_id_1d,
                                               {{domain::amr::Flag::Split}}) ==
        std::array<size_t, 1>{{3}});
  CHECK(domain::amr::desired_refinement_levels(
            element_id_1d, {{domain::amr::Flag::DoNothing}}) ==
        std::array<size_t, 1>{{2}});
  CHECK(domain::amr::desired_refinement_levels(element_id_1d,
                                               {{domain::amr::Flag::Join}}) ==
        std::array<size_t, 1>{{1}});

  domain::ElementId<2> element_id_2d{
      1, {{domain::SegmentId(3, 5), domain::SegmentId(1, 1)}}};
  CHECK(domain::amr::desired_refinement_levels(
            element_id_2d,
            {{domain::amr::Flag::Split, domain::amr::Flag::Join}}) ==
        std::array<size_t, 2>{{4, 0}});
  CHECK(domain::amr::desired_refinement_levels(
            element_id_2d,
            {{domain::amr::Flag::Join, domain::amr::Flag::DoNothing}}) ==
        std::array<size_t, 2>{{2, 1}});
  CHECK(domain::amr::desired_refinement_levels(
            element_id_2d,
            {{domain::amr::Flag::Join, domain::amr::Flag::Join}}) ==
        std::array<size_t, 2>{{2, 0}});
  CHECK(domain::amr::desired_refinement_levels(
            element_id_2d,
            {{domain::amr::Flag::DoNothing, domain::amr::Flag::Split}}) ==
        std::array<size_t, 2>{{3, 2}});
  CHECK(domain::amr::desired_refinement_levels(
            element_id_2d,
            {{domain::amr::Flag::DoNothing, domain::amr::Flag::DoNothing}}) ==
        std::array<size_t, 2>{{3, 1}});

  domain::ElementId<3> element_id_3d{
      7,
      {{domain::SegmentId(5, 15), domain::SegmentId(2, 0),
        domain::SegmentId(4, 6)}}};
  CHECK(domain::amr::desired_refinement_levels(
            element_id_3d, {{domain::amr::Flag::Split, domain::amr::Flag::Join,
                             domain::amr::Flag::DoNothing}}) ==
        std::array<size_t, 3>{{6, 1, 4}});
}

template <size_t VolumeDim>
void check_desired_refinement_levels_of_neighbor(
    const domain::ElementId<VolumeDim>& neighbor_id,
    const std::array<domain::amr::Flag, VolumeDim>& neighbor_flags) noexcept {
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

void test_desired_refinement_levels_of_neighbor() noexcept {
  domain::ElementId<1> neighbor_id_1d{0, {{domain::SegmentId(2, 3)}}};
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{domain::amr::Flag::Split}});
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{domain::amr::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(neighbor_id_1d,
                                              {{domain::amr::Flag::Join}});

  domain::ElementId<2> neighbor_id_2d{
      1, {{domain::SegmentId(3, 0), domain::SegmentId(1, 1)}}};
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d, {{domain::amr::Flag::Split, domain::amr::Flag::Join}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{domain::amr::Flag::Join, domain::amr::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d, {{domain::amr::Flag::Join, domain::amr::Flag::Join}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{domain::amr::Flag::DoNothing, domain::amr::Flag::Split}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_2d,
      {{domain::amr::Flag::DoNothing, domain::amr::Flag::DoNothing}});

  domain::ElementId<3> neighbor_id_3d{
      7,
      {{domain::SegmentId(5, 31), domain::SegmentId(2, 0),
        domain::SegmentId(4, 15)}}};
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_3d, {{domain::amr::Flag::Split, domain::amr::Flag::Join,
                        domain::amr::Flag::DoNothing}});
  check_desired_refinement_levels_of_neighbor(
      neighbor_id_3d, {{domain::amr::Flag::Join, domain::amr::Flag::DoNothing,
                        domain::amr::Flag::Split}});
}

void test_has_potential_sibling() {
  domain::ElementId<1> element_id_1d{0, {{domain::SegmentId(2, 3)}}};
  CHECK(domain::amr::has_potential_sibling(element_id_1d,
                                           domain::Direction<1>::lower_xi()));
  CHECK_FALSE(domain::amr::has_potential_sibling(
      element_id_1d, domain::Direction<1>::upper_xi()));

  domain::ElementId<2> element_id_2d{
      0, {{domain::SegmentId(3, 0), domain::SegmentId{1, 1}}}};
  CHECK(domain::amr::has_potential_sibling(element_id_2d,
                                           domain::Direction<2>::upper_xi()));
  CHECK(domain::amr::has_potential_sibling(element_id_2d,
                                           domain::Direction<2>::lower_eta()));
  CHECK_FALSE(domain::amr::has_potential_sibling(
      element_id_2d, domain::Direction<2>::lower_xi()));
  CHECK_FALSE(domain::amr::has_potential_sibling(
      element_id_2d, domain::Direction<2>::upper_eta()));

  domain::ElementId<3> element_id_3d{
      7,
      {{domain::SegmentId(5, 31), domain::SegmentId(2, 0),
        domain::SegmentId(4, 15)}}};
  CHECK(domain::amr::has_potential_sibling(element_id_3d,
                                           domain::Direction<3>::lower_xi()));
  CHECK(domain::amr::has_potential_sibling(element_id_3d,
                                           domain::Direction<3>::upper_eta()));
  CHECK(domain::amr::has_potential_sibling(element_id_3d,
                                           domain::Direction<3>::lower_zeta()));
  CHECK_FALSE(domain::amr::has_potential_sibling(
      element_id_3d, domain::Direction<3>::upper_xi()));
  CHECK_FALSE(domain::amr::has_potential_sibling(
      element_id_3d, domain::Direction<3>::lower_eta()));
  CHECK_FALSE(domain::amr::has_potential_sibling(
      element_id_3d, domain::Direction<3>::upper_zeta()));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Amr.Helpers", "[Domain][Unit]") {
  test_desired_refinement_levels();
  test_desired_refinement_levels_of_neighbor();
  test_has_potential_sibling();
}

// [[OutputRegex, Undefined amr::Flag in dimension]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Amr.Helpers.BadFlag",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto bad_flag = domain::amr::desired_refinement_levels(
      domain::ElementId<1>{0}, {{domain::amr::Flag::Undefined}});
  static_cast<void>(bad_flag);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, Undefined amr::Flag in dimension]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Amr.Helpers.BadFlag2",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto bad_flag = domain::amr::desired_refinement_levels_of_neighbor(
      domain::ElementId<1>{0}, {{domain::amr::Flag::Undefined}},
      domain::OrientationMap<1>{{{domain::Direction<1>::lower_xi()}}});
  static_cast<void>(bad_flag);
  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
