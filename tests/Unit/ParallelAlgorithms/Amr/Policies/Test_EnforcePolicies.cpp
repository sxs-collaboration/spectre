// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <catch2/matchers/catch_matchers.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <cstddef>
#include <vector>

#include "Domain/Amr/Flag.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Amr/Policies/EnforcePolicies.hpp"
#include "ParallelAlgorithms/Amr/Policies/Isotropy.hpp"
#include "ParallelAlgorithms/Amr/Policies/Limits.hpp"
#include "ParallelAlgorithms/Amr/Policies/Policies.hpp"
#include "Utilities/Gsl.hpp"

namespace {
template <size_t Dim>
void test_decision(const std::array<amr::Flag, Dim>& original_decision,
                   const amr::Policies& amr_policies,
                   const ElementId<Dim>& element_id, const Mesh<Dim>& mesh,
                   const std::array<amr::Flag, Dim>& expected_decision,
                   const bool error_beyond_limits = false) {
  auto decision = original_decision;
  if (error_beyond_limits) {
    CHECK_THROWS_WITH((enforce_policies(make_not_null(&decision), amr_policies,
                                        element_id, mesh)),
                      Catch::Matchers::ContainsSubstring(
                          "Tried refining beyond the AMR limits in element"));
  } else {
    enforce_policies(make_not_null(&decision), amr_policies, element_id, mesh);
    CHECK(decision == expected_decision);
  }
}

void test_1d() {
  const auto split = std::array{amr::Flag::Split};
  const auto increase = std::array{amr::Flag::IncreaseResolution};
  const auto stay = std::array{amr::Flag::DoNothing};
  const auto decrease = std::array{amr::Flag::DecreaseResolution};
  const auto join = std::array{amr::Flag::Join};
  const auto all_flags = std::vector{split, increase, stay, decrease, join};

  amr::Policies isotropic{amr::Isotropy::Isotropic, amr::Limits{}, true};
  amr::Policies anisotropic{amr::Isotropy::Anisotropic, amr::Limits{}, true};
  const auto policies = std::array{anisotropic, isotropic};
  const ElementId<1> element_id{0, {{{1, 0}}}};
  const Mesh<1> mesh{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
  for (const auto flags : all_flags) {
    for (const auto policy : policies) {
      test_decision(flags, policy, element_id, mesh, flags);
    }
  }

  amr::Policies policy{amr::Isotropy::Anisotropic, amr::Limits{0, 5, 1, 12},
                       true};
  test_decision(join, policy, ElementId<1>{0}, mesh, stay);
  test_decision(split, policy, ElementId<1>{0, {{{5, 2}}}}, mesh, stay);
  test_decision(
      decrease, policy, element_id,
      Mesh<1>{1, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss}, stay);
  test_decision(
      decrease, policy, element_id,
      Mesh<1>{2, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto},
      stay);
  test_decision(
      increase, policy, element_id,
      Mesh<1>{12, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss},
      stay);
}

void test_2d() {
  const auto split_split = std::array{amr::Flag::Split, amr::Flag::Split};
  const auto increase_split =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::Split};
  const auto stay_split = std::array{amr::Flag::DoNothing, amr::Flag::Split};
  const auto decrease_split =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::Split};
  const auto join_split = std::array{amr::Flag::Join, amr::Flag::Split};
  const auto split_increase =
      std::array{amr::Flag::Split, amr::Flag::IncreaseResolution};
  const auto increase_increase =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::IncreaseResolution};
  const auto stay_increase =
      std::array{amr::Flag::DoNothing, amr::Flag::IncreaseResolution};
  const auto decrease_increase =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::IncreaseResolution};
  const auto join_increase =
      std::array{amr::Flag::Join, amr::Flag::IncreaseResolution};
  const auto split_stay = std::array{amr::Flag::Split, amr::Flag::DoNothing};
  const auto increase_stay =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DoNothing};
  const auto stay_stay = std::array{amr::Flag::DoNothing, amr::Flag::DoNothing};
  const auto decrease_stay =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DoNothing};
  const auto join_stay = std::array{amr::Flag::Join, amr::Flag::DoNothing};
  const auto split_decrease =
      std::array{amr::Flag::Split, amr::Flag::DecreaseResolution};
  const auto increase_decrease =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::DecreaseResolution};
  const auto stay_decrease =
      std::array{amr::Flag::DoNothing, amr::Flag::DecreaseResolution};
  const auto decrease_decrease =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::DecreaseResolution};
  const auto join_decrease =
      std::array{amr::Flag::Join, amr::Flag::DecreaseResolution};
  const auto split_join = std::array{amr::Flag::Split, amr::Flag::Join};
  const auto increase_join =
      std::array{amr::Flag::IncreaseResolution, amr::Flag::Join};
  const auto stay_join = std::array{amr::Flag::DoNothing, amr::Flag::Join};
  const auto decrease_join =
      std::array{amr::Flag::DecreaseResolution, amr::Flag::Join};
  const auto join_join = std::array{amr::Flag::Join, amr::Flag::Join};

  const auto all_flags = std::vector{
      split_split,       increase_split, stay_split,        decrease_split,
      join_split,        split_increase, increase_increase, stay_increase,
      decrease_increase, join_increase,  split_stay,        increase_stay,
      stay_stay,         decrease_stay,  join_stay,         split_decrease,
      increase_decrease, stay_decrease,  decrease_decrease, join_decrease,
      split_join,        increase_join,  stay_join,         decrease_join,
      join_join};

  amr::Policies isotropic{amr::Isotropy::Isotropic, amr::Limits{}, true};
  amr::Policies anisotropic{amr::Isotropy::Anisotropic, amr::Limits{}, true};

  const ElementId<2> element_id{0, {{{1, 0}, {1, 0}}}};
  const Mesh<2> mesh{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};

  for (const auto flags : all_flags) {
    test_decision(flags, anisotropic, element_id, mesh, flags);
  }

  for (const auto flags : std::vector{
           split_split, increase_split, stay_split, decrease_split, join_split,
           split_increase, split_stay, split_decrease, split_join}) {
    test_decision(flags, isotropic, element_id, mesh, split_split);
  }
  for (const auto flags : std::vector{
           increase_increase, stay_increase, decrease_increase, join_increase,
           increase_stay, increase_decrease, increase_join}) {
    test_decision(flags, isotropic, element_id, mesh, increase_increase);
  }
  for (const auto flags : std::vector{stay_stay, stay_decrease, stay_join,
                                      decrease_stay, join_stay}) {
    test_decision(flags, isotropic, element_id, mesh, stay_stay);
  }
  for (const auto flags :
       std::vector{decrease_decrease, decrease_join, join_decrease}) {
    test_decision(flags, isotropic, element_id, mesh, decrease_decrease);
  }
  test_decision(join_join, isotropic, element_id, mesh, join_join);
}

void test_3d() {
  const auto stay_split_split =
      std::array{amr::Flag::DoNothing, amr::Flag::Split, amr::Flag::Split};
  const auto split_split_split =
      std::array{amr::Flag::Split, amr::Flag::Split, amr::Flag::Split};
  amr::Policies isotropic{amr::Isotropy::Isotropic, amr::Limits{}, true};
  amr::Policies anisotropic{amr::Isotropy::Anisotropic, amr::Limits{}, true};
  ElementId<3> element_id{0, {{{1, 0}, {1, 0}, {1, 0}}}};
  const Mesh<3> mesh{3, Spectral::Basis::Legendre, Spectral::Quadrature::Gauss};
  test_decision(stay_split_split, anisotropic, element_id, mesh,
                stay_split_split);
  test_decision(stay_split_split, isotropic, element_id, mesh,
                split_split_split);

  // Test error beyond limits
  isotropic = amr::Policies{amr::Isotropy::Isotropic,
                            amr::Limits{{{0, 0}}, {{2, 8}}, true}, true};
  anisotropic = amr::Policies{amr::Isotropy::Anisotropic,
                              amr::Limits{{{0, 0}}, {{2, 8}}, true}, true};
  element_id = ElementId<3>{0};
  test_decision(stay_split_split, anisotropic, element_id, mesh,
                stay_split_split, true);
  test_decision(stay_split_split, isotropic, element_id, mesh,
                split_split_split, true);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelAlgorithms.Amr.EnforcePolicies",
                  "[ParallelAlgorithms][Unit]") {
  test_1d();
  test_2d();
  test_3d();
}
