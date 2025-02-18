// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "Framework/TestCreation.hpp"
#include "Time/OptionTags/VariableOrderAlgorithm.hpp"
#include "Time/VariableOrderAlgorithm.hpp"
#include "Utilities/Literals.hpp"

SPECTRE_TEST_CASE("Unit.Time.OptionTags.VariableOrderAlgorithm",
                  "[Unit][Time]") {
  CHECK(TestHelpers::test_option_tag<OptionTags::VariableOrderAlgorithm>(
            "GoalOrder: 4") == VariableOrderAlgorithm(4_st));
}
