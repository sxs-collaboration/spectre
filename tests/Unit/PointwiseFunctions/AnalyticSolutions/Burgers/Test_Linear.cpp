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
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Linear.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Burgers.Linear",
                  "[PointwiseFunctions][Unit]") {
  const double shock_time = 1.5;
  const Burgers::Solutions::Linear solution(shock_time);
  const DataVector positions{-1., 0., 1., 2.5};
  check_burgers_solution(solution, positions, {-1., 0., 1., 2.5});

  // Check that the initial profile is correct.
  CHECK_ITERABLE_APPROX(
      get(solution.u(tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
      positions / -shock_time);
  CHECK_ITERABLE_APPROX(get(serialize_and_deserialize(solution).u(
                            tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
                        positions / -shock_time);

  register_classes_with_charm<Burgers::Solutions::Linear>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          Burgers::Solutions::Linear>(
          "Linear:\n"
          "  ShockTime: 1.5\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& created_solution =
      dynamic_cast<const Burgers::Solutions::Linear&>(
          *deserialized_option_solution);

  const auto x = tnsr::I<DataVector, 1>{{{positions}}};
  const double t = 0.0;
  CHECK(created_solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}) ==
        solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}));
}
