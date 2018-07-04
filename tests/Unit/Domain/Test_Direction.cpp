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

SPECTRE_TEST_CASE("Unit.Domain.Direction.Construction1D", "[Domain][Unit]") {
  auto upper_xi_1 = domain::Direction<1>::upper_xi();
  CHECK(upper_xi_1.opposite() == domain::Direction<1>::lower_xi());
  CHECK(upper_xi_1.opposite().sign() == -1.0);
  CHECK(upper_xi_1.axis() == domain::Direction<1>::Axis::Xi);
  test_serialization(upper_xi_1);

  auto lower_xi_1 = domain::Direction<1>(0, domain::Side::Lower);
  CHECK(lower_xi_1 == domain::Direction<1>::lower_xi());
  CHECK(lower_xi_1.axis() == domain::Direction<1>::Axis::Xi);
  CHECK(lower_xi_1.side() == domain::Side::Lower);
  test_serialization(lower_xi_1);
}

SPECTRE_TEST_CASE("Unit.Domain.Direction.Construction2D", "[Domain][Unit]") {
  auto upper_xi_2 = domain::Direction<2>(0, domain::Side::Upper);
  CHECK(upper_xi_2 == domain::Direction<2>::upper_xi());
  CHECK(upper_xi_2.axis() == domain::Direction<2>::Axis::Xi);
  CHECK(upper_xi_2.side() == domain::Side::Upper);
  test_serialization(upper_xi_2);

  auto upper_eta_2 = domain::Direction<2>(1, domain::Side::Upper);
  CHECK(upper_eta_2 == domain::Direction<2>::upper_eta());
  CHECK(upper_eta_2.axis() == domain::Direction<2>::Axis::Eta);
  CHECK(upper_eta_2.side() == domain::Side::Upper);
  test_serialization(upper_eta_2);

  auto lower_xi_2 = domain::Direction<2>(0, domain::Side::Lower);
  CHECK(lower_xi_2 == domain::Direction<2>::lower_xi());
  CHECK(lower_xi_2.axis() == domain::Direction<2>::Axis::Xi);
  CHECK(lower_xi_2.side() == domain::Side::Lower);
  test_serialization(lower_xi_2);

  auto lower_eta_2 = domain::Direction<2>(1, domain::Side::Lower);
  CHECK(lower_eta_2 == domain::Direction<2>::lower_eta());
  CHECK(lower_eta_2.axis() == domain::Direction<2>::Axis::Eta);
  CHECK(lower_eta_2.side() == domain::Side::Lower);
  test_serialization(lower_eta_2);
}

SPECTRE_TEST_CASE("Unit.Domain.Direction.Construction3D", "[Domain][Unit]") {
  auto upper_xi = domain::Direction<3>(0, domain::Side::Upper);
  CHECK(upper_xi == domain::Direction<3>::upper_xi());
  CHECK(upper_xi.axis() == domain::Direction<3>::Axis::Xi);
  CHECK(upper_xi.side() == domain::Side::Upper);
  test_serialization(upper_xi);

  auto upper_eta = domain::Direction<3>(1, domain::Side::Upper);
  CHECK(upper_eta == domain::Direction<3>::upper_eta());
  CHECK(upper_eta.axis() == domain::Direction<3>::Axis::Eta);
  CHECK(upper_eta.side() == domain::Side::Upper);
  test_serialization(upper_eta);

  auto lower_xi = domain::Direction<3>(0, domain::Side::Lower);
  CHECK(lower_xi == domain::Direction<3>::lower_xi());
  CHECK(lower_xi.axis() == domain::Direction<3>::Axis::Xi);
  CHECK(lower_xi.side() == domain::Side::Lower);
  test_serialization(lower_xi);

  auto lower_eta = domain::Direction<3>(1, domain::Side::Lower);
  CHECK(lower_eta == domain::Direction<3>::lower_eta());
  CHECK(lower_eta.axis() == domain::Direction<3>::Axis::Eta);
  CHECK(lower_eta.side() == domain::Side::Lower);
  test_serialization(lower_eta);

  auto upper_zeta = domain::Direction<3>(2, domain::Side::Upper);
  CHECK(upper_zeta == domain::Direction<3>::upper_zeta());
  CHECK(upper_zeta.axis() == domain::Direction<3>::Axis::Zeta);
  CHECK(upper_zeta.side() == domain::Side::Upper);
  test_serialization(upper_zeta);

  auto lower_zeta = domain::Direction<3>(2, domain::Side::Lower);
  CHECK(lower_zeta == domain::Direction<3>::lower_zeta());
  CHECK(lower_zeta.axis() == domain::Direction<3>::Axis::Zeta);
  CHECK(lower_zeta.side() == domain::Side::Lower);
  test_serialization(lower_zeta);
}

SPECTRE_TEST_CASE("Unit.Domain.Direction.HelperFunctions", "[Domain][Unit]") {
  auto upper_xi_3 = domain::Direction<3>::upper_xi();
  CHECK(upper_xi_3.axis() == domain::Direction<3>::Axis::Xi);
  CHECK(upper_xi_3.side() == domain::Side::Upper);

  auto lower_xi_3 = domain::Direction<3>::lower_xi();
  CHECK(lower_xi_3.axis() == domain::Direction<3>::Axis::Xi);
  CHECK(lower_xi_3.side() == domain::Side::Lower);

  auto upper_eta_3 = domain::Direction<3>::upper_eta();
  CHECK(upper_eta_3.axis() == domain::Direction<3>::Axis::Eta);
  CHECK(upper_eta_3.side() == domain::Side::Upper);

  auto lower_eta_3 = domain::Direction<3>::lower_eta();
  CHECK(lower_eta_3.axis() == domain::Direction<3>::Axis::Eta);
  CHECK(lower_eta_3.side() == domain::Side::Lower);

  auto upper_zeta_3 = domain::Direction<3>::upper_zeta();
  CHECK(upper_zeta_3.axis() == domain::Direction<3>::Axis::Zeta);
  CHECK(upper_zeta_3.side() == domain::Side::Upper);

  auto lower_zeta_3 = domain::Direction<3>::lower_zeta();
  CHECK(lower_zeta_3.axis() == domain::Direction<3>::Axis::Zeta);
  CHECK(lower_zeta_3.side() == domain::Side::Lower);
}

SPECTRE_TEST_CASE("Unit.Domain.Direction.Hash", "[Domain][Unit]") {
  size_t upper_xi_1_hash =
      std::hash<domain::Direction<1>>{}(domain::Direction<1>::upper_xi());
  size_t lower_xi_1_hash =
      std::hash<domain::Direction<1>>{}(domain::Direction<1>::lower_xi());

  size_t upper_xi_2_hash =
      std::hash<domain::Direction<2>>{}(domain::Direction<2>::upper_xi());
  size_t lower_xi_2_hash =
      std::hash<domain::Direction<2>>{}(domain::Direction<2>::lower_xi());
  size_t upper_eta_2_hash =
      std::hash<domain::Direction<2>>{}(domain::Direction<2>::upper_eta());
  size_t lower_eta_2_hash =
      std::hash<domain::Direction<2>>{}(domain::Direction<2>::lower_eta());

  size_t upper_xi_3_hash =
      std::hash<domain::Direction<3>>{}(domain::Direction<3>::upper_xi());
  size_t lower_xi_3_hash =
      std::hash<domain::Direction<3>>{}(domain::Direction<3>::lower_xi());
  size_t upper_eta_3_hash =
      std::hash<domain::Direction<3>>{}(domain::Direction<3>::upper_eta());
  size_t lower_eta_3_hash =
      std::hash<domain::Direction<3>>{}(domain::Direction<3>::lower_eta());
  size_t upper_zeta_3_hash =
      std::hash<domain::Direction<3>>{}(domain::Direction<3>::upper_zeta());
  size_t lower_zeta_3_hash =
      std::hash<domain::Direction<3>>{}(domain::Direction<3>::lower_zeta());

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

SPECTRE_TEST_CASE("Unit.Domain.Direction.Output", "[Domain][Unit]") {
  auto upper_xi = domain::Direction<1>(0, domain::Side::Upper);
  CHECK(get_output(upper_xi) == "+0");

  auto lower_xi = domain::Direction<2>(0, domain::Side::Lower);
  CHECK(get_output(lower_xi) == "-0");

  auto upper_eta = domain::Direction<2>(1, domain::Side::Upper);
  CHECK(get_output(upper_eta) == "+1");

  auto lower_eta = domain::Direction<2>(1, domain::Side::Lower);
  CHECK(get_output(lower_eta) == "-1");

  auto upper_xi_3 = domain::Direction<3>(0, domain::Side::Upper);
  CHECK(get_output(upper_xi_3) == "+0");

  auto lower_xi_3 = domain::Direction<3>(0, domain::Side::Lower);
  CHECK(get_output(lower_xi_3) == "-0");

  auto upper_eta_3 = domain::Direction<3>(1, domain::Side::Upper);
  CHECK(get_output(upper_eta_3) == "+1");

  auto lower_eta_3 = domain::Direction<3>(1, domain::Side::Lower);
  CHECK(get_output(lower_eta_3) == "-1");

  auto upper_zeta_3 = domain::Direction<3>(2, domain::Side::Upper);
  CHECK(get_output(upper_zeta_3) == "+2");

  auto lower_zeta_3 = domain::Direction<3>(2, domain::Side::Lower);
  CHECK(get_output(lower_zeta_3) == "-2");
}

// [[OutputRegex, dim = 1, for Direction<1> only dim = 0 is allowed.]]
[[noreturn]] SPECTRE_TEST_CASE("Unit.Domain.Direction.FirstDimension",
                               "[Domain][Unit]") {
  ASSERTION_TEST();
#ifdef SPECTRE_DEBUG
  auto failed_direction = domain::Direction<1>(1, domain::Side::Upper);
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
  auto failed_direction = domain::Direction<2>(2, domain::Side::Upper);
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
  auto failed_direction = domain::Direction<3>(3, domain::Side::Upper);
  static_cast<void>(failed_direction);

  ERROR("Failed to trigger ASSERT in an assertion test");
#endif
}
