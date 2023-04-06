// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <memory>
#include <vector>

#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Evolution/Systems/Burgers/Tags.hpp"
#include "Framework/TestCreation.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/PointwiseFunctions/AnalyticSolutions/Burgers/CheckBurgersSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Step.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Burgers.Step",
                  "[PointwiseFunctions][Unit]") {
  const double left_value = 2.3;
  const double right_value = 1.2;
  const double initial_position = -0.5;
  const Burgers::Solutions::Step solution(left_value, right_value,
                                          initial_position);
  const DataVector positions{-1., 0., 1., 2.5};

  // Check that the initial profile is correct.
  const DataVector expected_initial_profile =
      right_value +
      (left_value - right_value) *
          step_function(DataVector(-positions + initial_position));
  CHECK_ITERABLE_APPROX(
      get(solution.u(tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
      expected_initial_profile);

  // Check serialization
  CHECK_ITERABLE_APPROX(get(serialize_and_deserialize(solution).u(
                            tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
                        expected_initial_profile);

  // Check consistency
  // Note that we avoid checking at any times where x_i == x0 + speed * t,
  // because at these times the shock is near a grid point and the jump in the
  // solution breaks the tests of check_burgers_solution
  check_burgers_solution(solution, positions, {0., 0.7, 1.2, 10.0});

  register_classes_with_charm<Burgers::Solutions::Step>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          Burgers::Solutions::Step>(
          "Step:\n"
          "  LeftValue: 2.3\n"
          "  RightValue: 1.2\n"
          "  InitialPosition: -0.5\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& created_solution = dynamic_cast<const Burgers::Solutions::Step&>(
      *deserialized_option_solution);

  const auto x = tnsr::I<DataVector, 1>{{{positions}}};
  const double t = 0.0;
  CHECK(created_solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}) ==
        solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}));
}
