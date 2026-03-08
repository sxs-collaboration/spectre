// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <string>
#include <vector>

#include "Domain/Creators/AlignedLattice.hpp"
#include "Elliptic/Systems/SelfForce/Scalar/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace ScalarSelfForce {
namespace {

void test_null_slicing_tag_logic() {
  INFO("Testing NullSlicingBlocks structure");
  const std::array<std::vector<double>, 2> coords{
      {{0.0, 0.25, 0.5}, {0.0, 0.5}}};
  using Region = domain::creators::RefinementRegion<2>;
  const std::unique_ptr<DomainCreator<2>> lattice =
      std::make_unique<domain::creators::AlignedLattice<2>>(
          std::array<std::vector<double>, 2>{{{0., 0.25, 0.5}, {0., 0.5}}},
          std::array<size_t, 2>{{0, 0}}, std::array<size_t, 2>{{3, 2}},
          std::vector<Region>{Region{{0, 0}, {1, 1}, {0, 1}}},
          std::vector<Region>{}, std::vector<std::array<size_t, 2>>{},
          std::array<bool, 2>{{false, false}}, Options::Context{});

  const std::vector<std::string> blocks{"Block(0,0)", "Block(1,0)"};
  const std::vector<size_t> result =
      Tags::NullSlicingBlocks::create_from_options(blocks, lattice);
  const std::vector<size_t> expected{0, 1};
  CHECK(result == expected);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Systems.ScalarSelfForce.Tags",
                  "[Unit][Elliptic]") {
  TestHelpers::db::test_simple_tag<Tags::MMode>("MMode");
  TestHelpers::db::test_simple_tag<Tags::Alpha>("Alpha");
  TestHelpers::db::test_simple_tag<Tags::Beta>("Beta");
  TestHelpers::db::test_simple_tag<Tags::Gamma>("Gamma");
  TestHelpers::db::test_simple_tag<Tags::FieldIsRegularized>(
      "FieldIsRegularized");
  TestHelpers::db::test_simple_tag<Tags::SingularField>("SingularField");
  TestHelpers::db::test_simple_tag<Tags::BoyerLindquistRadius>(
      "BoyerLindquistRadius");
  TestHelpers::db::test_simple_tag<Tags::BoostFunction>("BoostFunction");
  TestHelpers::db::test_simple_tag<Tags::BoostFunctionDeriv>(
      "BoostFunctionDeriv");
  TestHelpers::db::test_simple_tag<Tags::NullSlicingBlocks>(
      "NullSlicingBlocks");

  test_null_slicing_tag_logic();
}

}  // namespace ScalarSelfForce
