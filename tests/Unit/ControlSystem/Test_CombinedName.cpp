// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "ControlSystem/CombinedName.hpp"
#include "Helpers/ControlSystem/TestStructs.hpp"
#include "Utilities/GetOutput.hpp"
#include "Utilities/TMPL.hpp"

namespace {
template <size_t Index>
struct Name {
  static std::string name() { return "Name" + get_output(Index); }
};

template <typename Label>
using measurement = control_system::TestHelpers::Measurement<Label>;

template <typename ControlSystemLabel, typename MeasurementLabel>
using system =
    control_system::TestHelpers::System<2, ControlSystemLabel,
                                        measurement<MeasurementLabel>, 1>;

struct LabelA {};
struct LabelB {};
struct LabelC {};
struct LabelD {};

void test_system_to_combined_names() {
  using control_systems =
      tmpl::list<system<LabelA, LabelA>, system<LabelB, LabelA>,
                 system<LabelC, LabelA>, system<LabelD, LabelD>>;

  const std::unordered_map<std::string, std::string> system_to_combined_names =
      control_system::system_to_combined_names<control_systems>();
  std::unordered_map<std::string, std::string>
      expected_system_to_combined_names{};
  const std::string long_combined_name{"LabelALabelBLabelC"};
  expected_system_to_combined_names["LabelA"] = long_combined_name;
  expected_system_to_combined_names["LabelB"] = long_combined_name;
  expected_system_to_combined_names["LabelC"] = long_combined_name;
  expected_system_to_combined_names["LabelD"] = "LabelD";

  CHECK(expected_system_to_combined_names == system_to_combined_names);
}

void test_combined_name() {
  using list_of_names = tmpl::list<Name<3>, Name<0>, Name<2>, Name<1>>;

  const std::string combined = control_system::combined_name<list_of_names>();
  CHECK(combined == "Name0Name1Name2Name3");

  const std::string empty = control_system::combined_name<tmpl::list<>>();
  CHECK(empty.empty());
}

SPECTRE_TEST_CASE("Unit.ControlSystem.CombinedName", "[Unit][ControlSystem]") {
  test_system_to_combined_names();
  test_combined_name();
}
}  // namespace
