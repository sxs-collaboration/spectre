// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/DgSubcell/Tags/ActiveGrid.hpp"
#include "Evolution/DgSubcell/Tags/Coordinates.hpp"
#include "Evolution/DgSubcell/Tags/ObserverCoordinates.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticSolutions/AnalyticSolution.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "PointwiseFunctions/InitialDataUtilities/InitialData.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct TestAnalyticSolution : public MarkAsAnalyticSolution,
                              public evolution::initial_data::InitialData {
  TestAnalyticSolution() = default;
  ~TestAnalyticSolution() override = default;

  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override {
    return std::make_unique<TestAnalyticSolution>(*this);
  }

  explicit TestAnalyticSolution(CkMigrateMessage* msg) : InitialData(msg) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestAnalyticSolution);
#pragma GCC diagnostic pop

  static tuples::TaggedTuple<FieldTag> variables(
      const tnsr::I<DataVector, 1>& x, const double t,
      const tmpl::list<FieldTag> /*meta*/) {
    return {Scalar<DataVector>{t * get<0>(x)}};
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { InitialData::pup(p); }
};

PUP::able::PUP_ID TestAnalyticSolution::my_PUP_ID = 0;

struct TestAnalyticData : public MarkAsAnalyticData,
                          public evolution::initial_data::InitialData {
  TestAnalyticData() = default;
  ~TestAnalyticData() override = default;


  auto get_clone() const
      -> std::unique_ptr<evolution::initial_data::InitialData> override {
    return std::make_unique<TestAnalyticData>(*this);
  }

  explicit TestAnalyticData(CkMigrateMessage* msg) : InitialData(msg) {}
  using PUP::able::register_constructor;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  WRAPPED_PUPable_decl_template(TestAnalyticData);
#pragma GCC diagnostic pop

  static tuples::TaggedTuple<FieldTag> variables(
      const tnsr::I<DataVector, 1>& x, const tmpl::list<FieldTag> /*meta*/) {
    return {Scalar<DataVector>{get<0>(x)}};
  }
  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override { InitialData::pup(p); }
};

PUP::able::PUP_ID TestAnalyticData::my_PUP_ID = 0;
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.ComputeTags", "[Unit][Evolution]") {
  tnsr::I<DataVector, 1, Frame::Inertial> inertial_coords{{{{1., 2., 3., 4.}}}};
  const double current_time = 2.;
  const Variables<tmpl::list<FieldTag>> vars{4, 3.};
  {
    INFO("Test analytic solution with analytic solution tag");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::Coordinates<1, Frame::Inertial>,
                          ::Tags::AnalyticSolution<TestAnalyticSolution>,
                          Tags::Time, ::Tags::Variables<tmpl::list<FieldTag>>>,
        db::AddComputeTags<evolution::Tags::AnalyticSolutionsCompute<
                               1, tmpl::list<FieldTag>, false>,
                           Tags::ErrorsCompute<tmpl::list<FieldTag>>>>(
        inertial_coords, TestAnalyticSolution{}, current_time, vars);
    const DataVector expected{2., 4., 6., 8.};
    const DataVector expected_error{1., -1., -3., -5.};
    CHECK_ITERABLE_APPROX(get(get<Tags::Analytic<FieldTag>>(box).value()),
                          expected);
    CHECK_ITERABLE_APPROX(get(get<Tags::Error<FieldTag>>(box).value()),
                          expected_error);
  }
  {
    INFO("Test analytic solution with analytic solution tag for dg-subcell");
    tnsr::I<DataVector, 1, Frame::Inertial> subcell_inertial_coords{
        {{{1., 1.5, 2., 2.5, 3., 3.5, 4.}}}};
    auto box = db::create<
        db::AddSimpleTags<
            domain::Tags::Coordinates<1, Frame::Inertial>,
            evolution::dg::subcell::Tags::Coordinates<1, Frame::Inertial>,
            evolution::dg::subcell::Tags::ActiveGrid,
            ::Tags::AnalyticSolution<TestAnalyticSolution>, Tags::Time,
            ::Tags::Variables<tmpl::list<FieldTag>>>,
        db::AddComputeTags<evolution::dg::subcell::Tags::
                               ObserverCoordinatesCompute<1, Frame::Inertial>,
                           evolution::Tags::AnalyticSolutionsCompute<
                               1, tmpl::list<FieldTag>, true>,
                           Tags::ErrorsCompute<tmpl::list<FieldTag>>>>(
        inertial_coords, subcell_inertial_coords,
        evolution::dg::subcell::ActiveGrid::Dg, TestAnalyticSolution{},
        current_time, vars);
    const DataVector expected{2., 4., 6., 8.};
    const DataVector expected_error{1., -1., -3., -5.};
    CHECK_ITERABLE_APPROX(get(get<Tags::Analytic<FieldTag>>(box).value()),
                          expected);
    CHECK_ITERABLE_APPROX(get(get<Tags::Error<FieldTag>>(box).value()),
                          expected_error);
    db::mutate<evolution::dg::subcell::Tags::ActiveGrid,
               ::Tags::Variables<tmpl::list<FieldTag>>>(
        [](const auto active_grid_ptr, const auto vars_ptr) {
          *active_grid_ptr = evolution::dg::subcell::ActiveGrid::Subcell;
          vars_ptr->initialize(7);
          *vars_ptr = Variables<tmpl::list<FieldTag>>{7, 3.};
        },
        make_not_null(&box));
    const DataVector subcell_expected{2., 3., 4., 5., 6., 7., 8.};
    const DataVector subcell_expected_error{1., 0., -1., -2., -3., -4., -5.};
    CHECK_ITERABLE_APPROX(get(get<Tags::Analytic<FieldTag>>(box).value()),
                          subcell_expected);
    CHECK_ITERABLE_APPROX(get(get<Tags::Error<FieldTag>>(box).value()),
                          subcell_expected_error);
  }
  {
    INFO("Test analytic solution with base class");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::Coordinates<1, Frame::Inertial>,
                          evolution::initial_data::Tags::InitialData,
                          Tags::Time, ::Tags::Variables<tmpl::list<FieldTag>>>,
        db::AddComputeTags<evolution::Tags::AnalyticSolutionsCompute<
                               1, tmpl::list<FieldTag>, false,
                               tmpl::list<TestAnalyticSolution>>,
                           Tags::ErrorsCompute<tmpl::list<FieldTag>>>>(
        inertial_coords,
        std::unique_ptr<evolution::initial_data::InitialData>{
            std::make_unique<TestAnalyticSolution>()},
        current_time, vars);
    const DataVector expected{2., 4., 6., 8.};
    const DataVector expected_error{1., -1., -3., -5.};
    CHECK_ITERABLE_APPROX(get(get<Tags::Analytic<FieldTag>>(box).value()),
                          expected);
    CHECK_ITERABLE_APPROX(get(get<Tags::Error<FieldTag>>(box).value()),
                          expected_error);
  }
  {
    INFO("Test analytic data with analytic data tag");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::Coordinates<1, Frame::Inertial>,
                          ::Tags::AnalyticData<TestAnalyticData>, Tags::Time,
                          ::Tags::Variables<tmpl::list<FieldTag>>>,
        db::AddComputeTags<evolution::Tags::AnalyticSolutionsCompute<
                               1, tmpl::list<FieldTag>, false>,
                           Tags::ErrorsCompute<tmpl::list<FieldTag>>>>(
        inertial_coords, TestAnalyticData{}, current_time, vars);
    CHECK_FALSE(get<Tags::Analytic<FieldTag>>(box).has_value());
    CHECK_FALSE(get<Tags::Error<FieldTag>>(box).has_value());
  }
  {
    INFO("Test analytic data with base class");
    const auto box = db::create<
        db::AddSimpleTags<domain::Tags::Coordinates<1, Frame::Inertial>,
                          evolution::initial_data::Tags::InitialData,
                          Tags::Time, ::Tags::Variables<tmpl::list<FieldTag>>>,
        db::AddComputeTags<
            evolution::Tags::AnalyticSolutionsCompute<
                1, tmpl::list<FieldTag>, false, tmpl::list<TestAnalyticData>>,
            Tags::ErrorsCompute<tmpl::list<FieldTag>>>>(
        inertial_coords,
        std::unique_ptr<evolution::initial_data::InitialData>{
            std::make_unique<TestAnalyticData>()},
        current_time, vars);
    CHECK_FALSE(get<Tags::Analytic<FieldTag>>(box).has_value());
    CHECK_FALSE(get<Tags::Error<FieldTag>>(box).has_value());
  }

  TestHelpers::db::test_compute_tag<evolution::Tags::AnalyticSolutionsCompute<
      1, tmpl::list<FieldTag>, false>>("AnalyticSolutions");
}
