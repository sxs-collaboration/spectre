// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "Domain/InterfaceHelpers.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"

namespace domain {
namespace {
struct SomeNumber : db::SimpleTag {
  using type = double;
  static std::string name() { return "SomeNumber"; }
};
struct VolumeArgumentBase : db::BaseTag {};
struct SomeVolumeArgument : VolumeArgumentBase, db::SimpleTag {
  using type = double;
  static std::string name() { return "SomeVolumeArgument"; }
};

// [interface_invokable_example]
struct ComputeSomethingOnInterface {
  using argument_tags = tmpl::list<SomeNumber, SomeVolumeArgument>;
  using volume_tags = tmpl::list<SomeVolumeArgument>;
  static double apply(const double& some_number_on_interface,
                      const double& volume_argument, const double factor) {
    return factor * some_number_on_interface + volume_argument;
  }
};
// [interface_invokable_example]

struct ComputeWithTemplateParameters {
  using argument_tags = tmpl::list<SomeNumber>;
  template <typename Arg>
  static double apply(const Arg& arg) {
    return arg;
  }
};

struct ComputeWithVoidReturnType {
  using argument_tags = tmpl::list<SomeNumber>;
  template <typename Arg>
  static void apply(const Arg& arg, const gsl::not_null<double*> result) {
    *result += arg;
  }
};

template <size_t Dim, typename DirectionsTag>
void test_interface_apply(
    const Element<Dim>& element,
    const std::unordered_set<Direction<Dim>>& directions,
    const std::unordered_map<Direction<Dim>, double>& number_on_interfaces,
    const std::unordered_map<Direction<Dim>, double>&
        expected_result_on_interfaces) {
  // Construct DataBox that holds the test data
  const auto box =
      db::create<db::AddSimpleTags<Tags::Element<Dim>, DirectionsTag,
                                   Tags::Interface<DirectionsTag, SomeNumber>,
                                   SomeVolumeArgument>>(
          element, directions, number_on_interfaces, 1.);
  // Test applying a function to the interface and give an example
  // [interface_apply_example]
  const auto computed_number_on_interfaces =
      interface_apply<DirectionsTag, tmpl::list<SomeNumber, SomeVolumeArgument>,
                      tmpl::list<SomeVolumeArgument>>(
          [](const double some_number_on_interface,
             const double volume_argument, const double factor) {
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
  // [interface_apply_example]

  // Test volume base tag
  const auto computed_numbers_with_base_tag =
      interface_apply<DirectionsTag, tmpl::list<SomeNumber, VolumeArgumentBase>,
                      tmpl::list<VolumeArgumentBase>>(
          [](const double some_number_on_interface,
             const double volume_argument, const double factor) {
            return factor * some_number_on_interface + volume_argument;
          },
          box, 2.);
  CHECK(computed_numbers_with_base_tag == computed_number_on_interfaces);

  // Test overload that takes a stateless invokable
  // [interface_apply_example_stateless]
  const auto computed_numbers_with_struct =
      interface_apply<DirectionsTag, ComputeSomethingOnInterface>(box, 2.);
  // [interface_apply_example_stateless]
  CHECK(computed_numbers_with_struct == computed_number_on_interfaces);

  {
    // Test an invokable with template parameters
    const auto computed_numbers =
        interface_apply<DirectionsTag, ComputeWithTemplateParameters>(box);
    for (const auto& direction_and_expected_number : number_on_interfaces) {
      CHECK(computed_numbers.at(direction_and_expected_number.first) ==
            direction_and_expected_number.second);
    }
  }
  {
    // Test an invokable with `void` return value
    size_t face_count = 0;
    interface_apply<DirectionsTag, tmpl::list<>, tmpl::list<>>(
        [&face_count]() { face_count += 1; }, box);
    CHECK(face_count == expected_result_on_interfaces.size());
    double sum = 0;
    interface_apply<DirectionsTag, ComputeWithVoidReturnType>(
        box, make_not_null(&sum));
    double expected_sum = 0;
    for (const auto& direction_and_number : number_on_interfaces) {
      expected_sum += direction_and_number.second;
    }
    CHECK(sum == expected_sum);
  }
}

SPECTRE_TEST_CASE("Unit.Domain.InterfaceHelpers", "[Unit][Domain]") {
  test_interface_apply<1, Tags::InternalDirections<1>>(
      // Reference element has one internal direction:
      // [ X | ]-> xi
      {{0, {{{1, 0}}}}, {{Direction<1>::upper_xi(), {{{0, {{{1, 1}}}}}, {}}}}},
      {Direction<1>::upper_xi()}, {{Direction<1>::upper_xi(), 2.}},
      {{Direction<1>::upper_xi(), 5.}});
  test_interface_apply<1, Tags::InternalDirections<1>>(
      // Reference element has no internal directions:
      // [ X ]-> xi
      {{0, {{{0, 0}}}}, {}}, {}, {}, {});
  test_interface_apply<1, Tags::BoundaryDirectionsInterior<1>>(
      // Reference element has two boundary directions:
      // [ X ]-> xi
      {{0, {{{0, 0}}}}, {}},
      {Direction<1>::lower_xi(), Direction<1>::upper_xi()},
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
      {Direction<2>::upper_xi()}, {{Direction<2>::upper_xi(), 2.}},
      {{Direction<2>::upper_xi(), 5.}});
}
}  // namespace
}  // namespace domain
