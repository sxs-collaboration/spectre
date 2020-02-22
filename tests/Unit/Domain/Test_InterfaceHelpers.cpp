// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/Tags.hpp"

namespace domain {
namespace {
struct SomeNumber : db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "SomeNumber"; }
};
struct VolumeArgumentBase : db::BaseTag {};
struct SomeVolumeArgument : VolumeArgumentBase, db::SimpleTag {
  using type = double;
  static std::string name() noexcept { return "SomeVolumeArgument"; }
};

template <size_t Dim, typename DirectionsTag>
void test_interface_apply(
    const Element<Dim>& element,
    const std::unordered_map<Direction<Dim>, double>& number_on_interfaces,
    const std::unordered_map<Direction<Dim>, double>&
        expected_result_on_interfaces) {
  // Construct DataBox that holds the test data
  const auto box =
      db::create<db::AddSimpleTags<Tags::Element<Dim>,
                                   Tags::Interface<DirectionsTag, SomeNumber>,
                                   SomeVolumeArgument>,
                 db::AddComputeTags<DirectionsTag>>(element,
                                                    number_on_interfaces, 1.);
  // Test applying a function to the interface and give an example
  /// [interface_apply_example]
  const auto computed_number_on_interfaces =
      interface_apply<DirectionsTag, tmpl::list<SomeNumber, SomeVolumeArgument>,
                      tmpl::list<SomeVolumeArgument>>(
          [](const double& some_number_on_interface,
             const double& volume_argument, const double factor) noexcept {
            return factor * some_number_on_interface + volume_argument;
          },
          box, 2.);
  CHECK(computed_number_on_interfaces.size() ==
        expected_result_on_interfaces.size());
  for (const auto& direction_and_expected_result :
       expected_result_on_interfaces) {
    CHECK(
        computed_number_on_interfaces.at(direction_and_expected_result.first) ==
        direction_and_expected_result.second);
  }
  /// [interface_apply_example]

  // Test volume base tag
  const auto computed_numbers_with_base_tag =
      interface_apply<DirectionsTag, tmpl::list<SomeNumber, VolumeArgumentBase>,
                      tmpl::list<VolumeArgumentBase>>(
          [](const double& some_number_on_interface,
             const double& volume_argument, const double factor) noexcept {
            return factor * some_number_on_interface + volume_argument;
          },
          box, 2.);
  CHECK(computed_numbers_with_base_tag == computed_number_on_interfaces);
}

SPECTRE_TEST_CASE("Unit.Domain.InterfaceHelpers", "[Unit][Domain]") {
  test_interface_apply<1, Tags::InternalDirections<1>>(
      // Reference element has one internal direction:
      // [ X | ]-> xi
      {{0, {{{1, 0}}}}, {{Direction<1>::upper_xi(), {{{0, {{{1, 1}}}}}, {}}}}},
      {{Direction<1>::upper_xi(), 2.}}, {{Direction<1>::upper_xi(), 5.}});
  test_interface_apply<1, Tags::InternalDirections<1>>(
      // Reference element has no internal directions:
      // [ X ]-> xi
      {{0, {{{0, 0}}}}, {}}, {}, {});
  test_interface_apply<1, Tags::BoundaryDirectionsInterior<1>>(
      // Reference element has two boundary directions:
      // [ X ]-> xi
      {{0, {{{0, 0}}}}, {}},
      {{Direction<1>::lower_xi(), 2.}, {Direction<1>::upper_xi(), 3.}},
      {{Direction<1>::lower_xi(), 5.}, {Direction<1>::upper_xi(), 7.}});
  test_interface_apply<2, Tags::InternalDirections<2>>(
      // Reference element has one internal directions:
      // ^ eta
      // +-+-+
      // |X| |
      // +-+-+> xi
      {{0, {{{1, 0}, {0, 0}}}},
       {{Direction<2>::upper_xi(), {{{0, {{{1, 1}, {0, 0}}}}}, {}}}}},
      {{Direction<2>::upper_xi(), 2.}}, {{Direction<2>::upper_xi(), 5.}});
}
}  // namespace
}  // namespace domain
