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
#include "PointwiseFunctions/AnalyticSolutions/Burgers/Bump.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "PointwiseFunctions/InitialDataUtilities/Tags/InitialData.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tensor

SPECTRE_TEST_CASE("Unit.PointwiseFunctions.AnalyticSolutions.Burgers.Bump",
                  "[PointwiseFunctions][Unit]") {
  const double half_width = 5.;
  const double height = 3.;
  const double center = -8.;
  const Burgers::Solutions::Bump solution(half_width, height, center);
  const DataVector positions{-11., -9., -8., -6.};
  check_burgers_solution(solution, positions, {-0.5, 0., 0.1, 0.5});

  // Check that the initial profile is correct.
  CHECK_ITERABLE_APPROX(
      get(solution.u(tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
      height * (1. - square((positions - center) / half_width)));
  CHECK_ITERABLE_APPROX(
      get(serialize_and_deserialize(solution).u(
          tnsr::I<DataVector, 1>{{{positions}}}, 0.)),
      height * (1. - square((positions - center) / half_width)));

  register_classes_with_charm<Burgers::Solutions::Bump>();
  const std::unique_ptr<evolution::initial_data::InitialData> option_solution =
      TestHelpers::test_option_tag_factory_creation<
          evolution::initial_data::OptionTags::InitialData,
          Burgers::Solutions::Bump>(
          "Bump:\n"
          "  HalfWidth: 5.\n"
          "  Height: 3.\n"
          "  Center: -8.\n")
          ->get_clone();
  const auto deserialized_option_solution =
      serialize_and_deserialize(option_solution);
  const auto& created_solution = dynamic_cast<const Burgers::Solutions::Bump&>(
      *deserialized_option_solution);

  const auto x = tnsr::I<DataVector, 1>{{{positions}}};
  const double t = 0.0;
  CHECK(created_solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}) ==
        solution.variables(x, t, tmpl::list<Burgers::Tags::U>{}));
}
