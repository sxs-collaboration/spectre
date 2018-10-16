// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <string>

#include "Domain/Direction.hpp"
#include "Domain/Side.hpp"
#include "ErrorHandling/Error.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/Gsl.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {
void test_construction_1d() {
  auto upper_xi_1 = Direction<1>::upper_xi();
  CHECK(upper_xi_1.opposite() == Direction<1>::lower_xi());
  CHECK(upper_xi_1.opposite().sign() == -1.0);
  CHECK(upper_xi_1.axis() == Direction<1>::Axis::Xi);
  test_serialization(upper_xi_1);

  auto lower_xi_1 = Direction<1>(0, Side::Lower);
  CHECK(lower_xi_1 == Direction<1>::lower_xi());
  CHECK(lower_xi_1.axis() == Direction<1>::Axis::Xi);
  CHECK(lower_xi_1.side() == Side::Lower);
  test_serialization(lower_xi_1);
}

void test_construction_2d() {
  auto upper_xi_2 = Direction<2>(0, Side::Upper);
  CHECK(upper_xi_2 == Direction<2>::upper_xi());
  CHECK(upper_xi_2.axis() == Direction<2>::Axis::Xi);
  CHECK(upper_xi_2.side() == Side::Upper);
  test_serialization(upper_xi_2);

  auto upper_eta_2 = Direction<2>(1, Side::Upper);
  CHECK(upper_eta_2 == Direction<2>::upper_eta());
  CHECK(upper_eta_2.axis() == Direction<2>::Axis::Eta);
  CHECK(upper_eta_2.side() == Side::Upper);
  test_serialization(upper_eta_2);

  auto lower_xi_2 = Direction<2>(0, Side::Lower);
  CHECK(lower_xi_2 == Direction<2>::lower_xi());
  CHECK(lower_xi_2.axis() == Direction<2>::Axis::Xi);
  CHECK(lower_xi_2.side() == Side::Lower);
  test_serialization(lower_xi_2);

  auto lower_eta_2 = Direction<2>(1, Side::Lower);
  CHECK(lower_eta_2 == Direction<2>::lower_eta());
  CHECK(lower_eta_2.axis() == Direction<2>::Axis::Eta);
  CHECK(lower_eta_2.side() == Side::Lower);
  test_serialization(lower_eta_2);
}

void test_construction_3d() {
  auto upper_xi = Direction<3>(0, Side::Upper);
  CHECK(upper_xi == Direction<3>::upper_xi());
  CHECK(upper_xi.axis() == Direction<3>::Axis::Xi);
  CHECK(upper_xi.side() == Side::Upper);
  test_serialization(upper_xi);

  auto upper_eta = Direction<3>(1, Side::Upper);
  CHECK(upper_eta == Direction<3>::upper_eta());
  CHECK(upper_eta.axis() == Direction<3>::Axis::Eta);
  CHECK(upper_eta.side() == Side::Upper);
  test_serialization(upper_eta);

  auto lower_xi = Direction<3>(0, Side::Lower);
  CHECK(lower_xi == Direction<3>::lower_xi());
  CHECK(lower_xi.axis() == Direction<3>::Axis::Xi);
  CHECK(lower_xi.side() == Side::Lower);
  test_serialization(lower_xi);

  auto lower_eta = Direction<3>(1, Side::Lower);
  CHECK(lower_eta == Direction<3>::lower_eta());
  CHECK(lower_eta.axis() == Direction<3>::Axis::Eta);
  CHECK(lower_eta.side() == Side::Lower);
  test_serialization(lower_eta);

  auto upper_zeta = Direction<3>(2, Side::Upper);
  CHECK(upper_zeta == Direction<3>::upper_zeta());
  CHECK(upper_zeta.axis() == Direction<3>::Axis::Zeta);
  CHECK(upper_zeta.side() == Side::Upper);
  test_serialization(upper_zeta);

  auto lower_zeta = Direction<3>(2, Side::Lower);
  CHECK(lower_zeta == Direction<3>::lower_zeta());
  CHECK(lower_zeta.axis() == Direction<3>::Axis::Zeta);
  CHECK(lower_zeta.side() == Side::Lower);
  test_serialization(lower_zeta);
}

void test_helper_functions() {
  auto upper_xi_3 = Direction<3>::upper_xi();
  CHECK(upper_xi_3.axis() == Direction<3>::Axis::Xi);
  CHECK(upper_xi_3.side() == Side::Upper);

  auto lower_xi_3 = Direction<3>::lower_xi();
  CHECK(lower_xi_3.axis() == Direction<3>::Axis::Xi);
  CHECK(lower_xi_3.side() == Side::Lower);

  auto upper_eta_3 = Direction<3>::upper_eta();
  CHECK(upper_eta_3.axis() == Direction<3>::Axis::Eta);
  CHECK(upper_eta_3.side() == Side::Upper);

  auto lower_eta_3 = Direction<3>::lower_eta();
  CHECK(lower_eta_3.axis() == Direction<3>::Axis::Eta);
  CHECK(lower_eta_3.side() == Side::Lower);

  auto upper_zeta_3 = Direction<3>::upper_zeta();
  CHECK(upper_zeta_3.axis() == Direction<3>::Axis::Zeta);
  CHECK(upper_zeta_3.side() == Side::Upper);

  auto lower_zeta_3 = Direction<3>::lower_zeta();
  CHECK(lower_zeta_3.axis() == Direction<3>::Axis::Zeta);
  CHECK(lower_zeta_3.side() == Side::Lower);
}

void test_std_hash() {
  size_t upper_xi_1_hash = std::hash<Direction<1>>{}(Direction<1>::upper_xi());
  size_t lower_xi_1_hash = std::hash<Direction<1>>{}(Direction<1>::lower_xi());

  size_t upper_xi_2_hash = std::hash<Direction<2>>{}(Direction<2>::upper_xi());
  size_t lower_xi_2_hash = std::hash<Direction<2>>{}(Direction<2>::lower_xi());
  size_t upper_eta_2_hash =
      std::hash<Direction<2>>{}(Direction<2>::upper_eta());
  size_t lower_eta_2_hash =
      std::hash<Direction<2>>{}(Direction<2>::lower_eta());

  size_t upper_xi_3_hash = std::hash<Direction<3>>{}(Direction<3>::upper_xi());
  size_t lower_xi_3_hash = std::hash<Direction<3>>{}(Direction<3>::lower_xi());
  size_t upper_eta_3_hash =
      std::hash<Direction<3>>{}(Direction<3>::upper_eta());
  size_t lower_eta_3_hash =
      std::hash<Direction<3>>{}(Direction<3>::lower_eta());
  size_t upper_zeta_3_hash =
      std::hash<Direction<3>>{}(Direction<3>::upper_zeta());
  size_t lower_zeta_3_hash =
      std::hash<Direction<3>>{}(Direction<3>::lower_zeta());

  CHECK(upper_xi_1_hash != lower_xi_1_hash);
  std::array<size_t, 4> direction_2 = {{upper_xi_2_hash, lower_xi_2_hash,
                                       upper_eta_2_hash, lower_eta_2_hash}};
  for (size_t i = 0; i < direction_2.size(); i++) {
    for (size_t j = i + 1; j < direction_2.size(); j++) {
      CHECK(gsl::at(direction_2, i) != gsl::at(direction_2, j));
    }
  }
  std::array<size_t, 6> direction_3 = {{upper_xi_3_hash,   lower_xi_3_hash,
                                       upper_eta_3_hash,  lower_eta_3_hash,
                                       upper_zeta_3_hash, lower_zeta_3_hash}};
  for (size_t i = 0; i < direction_3.size(); i++) {
    for (size_t j = i + 1; j < direction_3.size(); j++) {
      CHECK(gsl::at(direction_3, i) != gsl::at(direction_3, j));
    }
  }
}

void test_direction_hash() {
  CHECK(DirectionHash<1>{}(Direction<1>::lower_xi()) == 0);
  CHECK(DirectionHash<1>{}(Direction<1>::upper_xi()) == 1);

  CHECK(DirectionHash<2>{}(Direction<2>::lower_xi()) == 0);
  CHECK(DirectionHash<2>{}(Direction<2>::upper_xi()) == 1);
  CHECK(DirectionHash<2>{}(Direction<2>::lower_eta()) == 2);
  CHECK(DirectionHash<2>{}(Direction<2>::upper_eta()) == 3);

  CHECK(DirectionHash<3>{}(Direction<3>::lower_xi()) == 0);
  CHECK(DirectionHash<3>{}(Direction<3>::upper_xi()) == 1);
  CHECK(DirectionHash<3>{}(Direction<3>::lower_eta()) == 2);
  CHECK(DirectionHash<3>{}(Direction<3>::upper_eta()) == 3);
  CHECK(DirectionHash<3>{}(Direction<3>::lower_zeta()) == 4);
  CHECK(DirectionHash<3>{}(Direction<3>::upper_zeta()) == 5);
}

void test_output() {
  auto upper_xi = Direction<1>(0, Side::Upper);
  CHECK(get_output(upper_xi) == "+0");

  auto lower_xi = Direction<2>(0, Side::Lower);
  CHECK(get_output(lower_xi) == "-0");

  auto upper_eta = Direction<2>(1, Side::Upper);
  CHECK(get_output(upper_eta) == "+1");

  auto lower_eta = Direction<2>(1, Side::Lower);
  CHECK(get_output(lower_eta) == "-1");

  auto upper_xi_3 = Direction<3>(0, Side::Upper);
  CHECK(get_output(upper_xi_3) == "+0");

  auto lower_xi_3 = Direction<3>(0, Side::Lower);
  CHECK(get_output(lower_xi_3) == "-0");

  auto upper_eta_3 = Direction<3>(1, Side::Upper);
  CHECK(get_output(upper_eta_3) == "+1");

  auto lower_eta_3 = Direction<3>(1, Side::Lower);
  CHECK(get_output(lower_eta_3) == "-1");

  auto upper_zeta_3 = Direction<3>(2, Side::Upper);
  CHECK(get_output(upper_zeta_3) == "+2");

  auto lower_zeta_3 = Direction<3>(2, Side::Lower);
  CHECK(get_output(lower_zeta_3) == "-2");
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.Direction", "[Domain][Unit]") {
  test_construction_1d();
  test_construction_2d();
  test_construction_3d();
  test_helper_functions();
  test_std_hash();
  test_direction_hash();
  test_output();
}

// [[OutputRegex, dim = 1, for Direction<1> only dim = 0 is allowed.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Direction.FirstDimension",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_direction = Direction<1>(1, Side::Upper);
  static_cast<void>(failed_direction);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, dim = 2, for Direction<2> only dim = 0 or dim = 1 are
// allowed.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Direction.SecondDimension",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_direction = Direction<2>(2, Side::Upper);
  static_cast<void>(failed_direction);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}

// [[OutputRegex, dim = 3, for Direction<3> only dim = 0, dim = 1, or dim = 2
// are allowed.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Direction.ThirdDimension",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_direction = Direction<3>(3, Side::Upper);
  static_cast<void>(failed_direction);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
