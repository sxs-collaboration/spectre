// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct DummyType {};
struct DummyTag : db::SimpleTag {
  using type = DummyType;
};

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.AnalyticSolutions.Tags", "[Unit][PointwiseFunctions]") {
  TestHelpers::db::test_base_tag<Tags::AnalyticSolutionBase>(
      "AnalyticSolutionBase");
  TestHelpers::db::test_simple_tag<Tags::AnalyticSolution<DummyType>>(
      "AnalyticSolution");
  // [analytic_name]
  TestHelpers::db::test_prefix_tag<Tags::Analytic<DummyTag>>(
      "Analytic(DummyTag)");
  // [analytic_name]
  TestHelpers::db::test_prefix_tag<Tags::Error<DummyTag>>("Error(DummyTag)");
  TestHelpers::db::test_base_tag<Tags::AnalyticSolutionsBase>(
      "AnalyticSolutionsBase");
  TestHelpers::db::test_simple_tag<
      Tags::AnalyticSolutions<tmpl::list<DummyTag>>>("AnalyticSolutions");
  TestHelpers::db::test_compute_tag<Tags::ErrorsCompute<tmpl::list<FieldTag>>>(
      "Errors");

  tnsr::I<DataVector, 1, Frame::Inertial> inertial_coords{{{{1., 2., 3., 4.}}}};
  const double current_time = 2.;
  const Variables<tmpl::list<FieldTag>> vars{4, 3.};
  using solutions_tag = ::Tags::AnalyticSolutions<tmpl::list<FieldTag>>;
  {
    INFO("Test analytic solution");
    auto solution =
        std::make_optional(typename solutions_tag::type::value_type{4});
    get<Tags::detail::AnalyticImpl<FieldTag>>(solution.value()) =
        Scalar<DataVector>{current_time * get<0>(inertial_coords)};

    auto box = db::create<
        db::AddSimpleTags<domain::Tags::Coordinates<1, Frame::Inertial>,
                          Tags::Time, ::Tags::Variables<tmpl::list<FieldTag>>,
                          solutions_tag>,
        db::AddComputeTags<Tags::ErrorsCompute<tmpl::list<FieldTag>>>>(
        inertial_coords, current_time, vars, solution);
    const DataVector expected{2., 4., 6., 8.};
    const DataVector expected_error{1., -1., -3., -5.};
    CHECK_ITERABLE_APPROX(get(get<Tags::Analytic<FieldTag>>(box).value()),
                          expected);
    CHECK_ITERABLE_APPROX(get(get<Tags::Error<FieldTag>>(box).value()),
                          expected_error);
    db::mutate<::Tags::Variables<tmpl::list<FieldTag>>>(
        [](const auto vars_ptr) {
          *vars_ptr = Variables<tmpl::list<FieldTag>>{4, 4.};
        },
        make_not_null(&box));
    const DataVector new_expected_error{2., 0., -2., -4.};
    CHECK_ITERABLE_APPROX(get(get<Tags::Analytic<FieldTag>>(box).value()),
                          expected);
    CHECK_ITERABLE_APPROX(get(get<Tags::Error<FieldTag>>(box).value()),
                          new_expected_error);
  }
  {
    INFO("Test analytic data");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::Coordinates<1, Frame::Inertial>,
                          Tags::Time, ::Tags::Variables<tmpl::list<FieldTag>>,
                          solutions_tag>,
        db::AddComputeTags<Tags::ErrorsCompute<tmpl::list<FieldTag>>>>(
        inertial_coords, current_time, vars,
        typename solutions_tag::type{std::nullopt});
    const DataVector expected{2., 4., 6., 8.};
    CHECK_FALSE(get<Tags::Analytic<FieldTag>>(box).has_value());
    CHECK_FALSE(get<Tags::Error<FieldTag>>(box).has_value());
  }
}
