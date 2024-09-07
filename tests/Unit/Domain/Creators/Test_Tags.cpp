// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Creators/BinaryCompactObject.hpp"
#include "Domain/Creators/OptionTags.hpp"
#include "Domain/Creators/Rectilinear.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/ExternalBoundaryConditions.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Creators/Tags/ObjectCenter.hpp"
#include "Domain/FunctionsOfTime/OptionTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"

namespace domain {
namespace {
template <size_t Dim>
void test_simple_tags() {
  TestHelpers::db::test_simple_tag<Tags::Domain<Dim>>("Domain");
  TestHelpers::db::test_simple_tag<Tags::InitialExtents<Dim>>("InitialExtents");
  TestHelpers::db::test_simple_tag<Tags::InitialRefinementLevels<Dim>>(
      "InitialRefinementLevels");
  TestHelpers::db::test_simple_tag<Tags::ExternalBoundaryConditions<Dim>>(
      "ExternalBoundaryConditions");
}

void test_center_tags() {
  TestHelpers::db::test_simple_tag<Tags::ObjectCenter<ObjectLabel::A>>(
      "ObjectCenterA");
  TestHelpers::db::test_simple_tag<Tags::ObjectCenter<ObjectLabel::B>>(
      "ObjectCenterB");

  using Object = domain::creators::BinaryCompactObject<false>::Object;

  const std::unique_ptr<DomainCreator<3>> domain_creator =
      std::make_unique<domain::creators::BinaryCompactObject<false>>(
          Object{0.2, 5.0, 8.0, true, true}, Object{0.6, 4.0, -5.5, true, true},
          std::array<double, 2>{{0.1, 0.2}}, 100.0, 500.0, 1.0, 1_st, 5_st);

  const auto grid_center_A =
      Tags::ObjectCenter<ObjectLabel::A>::create_from_options(domain_creator);
  const auto grid_center_B =
      Tags::ObjectCenter<ObjectLabel::B>::create_from_options(domain_creator);

  CHECK(grid_center_A == tnsr::I<double, 3, Frame::Grid>{{8.0, 0.1, 0.2}});
  CHECK(grid_center_B == tnsr::I<double, 3, Frame::Grid>{{-5.5, 0.1, 0.2}});

  const std::unique_ptr<DomainCreator<3>> creator_no_excision =
      std::make_unique<domain::creators::Brick>(
          std::array{0.0, 0.0, 0.0}, std::array{1.0, 1.0, 1.0},
          std::array{0_st, 0_st, 0_st}, std::array{2_st, 2_st, 2_st},
          std::array{false, false, false});

  CHECK_THROWS_WITH(
      Tags::ObjectCenter<ObjectLabel::B>::create_from_options(
          creator_no_excision),
      Catch::Matchers::ContainsSubstring(
          " is not in the domain creators grid anchors but is needed "
          "to generate the ObjectCenter"));
}

struct Metavariables {
  static constexpr size_t volume_dim = 3;
};

void test_functions_of_time() {
  TestHelpers::db::test_simple_tag<Tags::FunctionsOfTimeInitialize>(
      "FunctionsOfTime");

  CHECK(std::is_same_v<
        Tags::FunctionsOfTimeInitialize::option_tags<Metavariables>,
        tmpl::list<
            domain::OptionTags::DomainCreator<Metavariables::volume_dim>>>);
}

SPECTRE_TEST_CASE("Unit.Domain.Creators.Tags", "[Unit][Domain]") {
  test_simple_tags<1>();
  test_simple_tags<2>();
  test_simple_tags<3>();

  test_center_tags();

  test_functions_of_time();
}
}  // namespace
}  // namespace domain
