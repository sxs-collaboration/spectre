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
#include "Domain/Tags.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Evolution/Systems/GeneralizedHarmonic/System.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/GaugeWave.hpp"
#include "PointwiseFunctions/AnalyticSolutions/GeneralRelativity/WrappedGr.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct AnalyticSolution {
  static tuples::TaggedTuple<FieldTag> variables(
      const tnsr::I<DataVector, 1>& x, const double t,
      const tmpl::list<FieldTag> /*meta*/) noexcept {
    return {Scalar<DataVector>{t * get<0>(x)}};
  }
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

struct AnalyticSolutionTag : db::SimpleTag {
  using type = AnalyticSolution;
};

}  // namespace
SPECTRE_TEST_CASE("Unit.Evolution.ComputeTags", "[Unit][Evolution]") {
  tnsr::I<DataVector, 1, Frame::Inertial> inertial_coords{{{{1., 2., 3., 4.}}}};
  const double current_time = 2.;
  const auto box = db::create<
      db::AddSimpleTags<domain::Tags::Coordinates<1, Frame::Inertial>,
                        AnalyticSolutionTag, Tags::Time>,
      db::AddComputeTags<evolution::Tags::AnalyticCompute<
          1, AnalyticSolutionTag, tmpl::list<FieldTag>>>>(
      std::move(inertial_coords), AnalyticSolution{}, current_time);
  const DataVector expected{2., 4., 6., 8.};
  CHECK_ITERABLE_APPROX(get(get<::Tags::Analytic<FieldTag>>(box)), expected);

  TestHelpers::db::test_compute_tag<evolution::Tags::AnalyticCompute<
      1, AnalyticSolutionTag, tmpl::list<FieldTag>>>(
      "Variables(Analytic(FieldTag))");
}

SPECTRE_TEST_CASE("Unit.Evolution.ComputeTags.Errors",
                  "[Unit][PointwiseFunctions]") {
  static constexpr size_t volume_dim = 1;
  using system = GeneralizedHarmonic::System<volume_dim>;
  using solution = GeneralizedHarmonic::Solutions::WrappedGr<
      gr::Solutions::GaugeWave<volume_dim>>;
  using solution_tag = Tags::AnalyticSolution<solution>;

  using simple_tags =
      tmpl::list<domain::Tags::Coordinates<volume_dim, Frame::Inertial>,
                 ::Tags::Time, system::variables_tag, solution_tag>;
  using compute_tags = tmpl::list<evolution::Tags::ErrorsCompute<
      volume_dim, solution_tag, system::variables_tag::tags_list>>;

  constexpr double amplitude = 0.1;
  constexpr double wavelength = 4.0;
  const solution& analytic_solution_computer{amplitude, wavelength};

  tnsr::I<DataVector, volume_dim, Frame::Inertial> inertial_coords{
      {{{1., 2., 3., 4.}}}};
  constexpr double time = 2.;
  const auto& analytic =
      variables_from_tagged_tuple(analytic_solution_computer.variables(
          inertial_coords, time, system::variables_tag::tags_list{}));

  constexpr double constant_numeric_value = 5.0;
  db::item_type<system::variables_tag> numeric{4, constant_numeric_value};

  const auto box = db::create<simple_tags, compute_tags>(
      std::move(inertial_coords), time, numeric,
      solution(amplitude, wavelength));

  tmpl::for_each<system::variables_tag::tags_list>([&box,
                                                    &analytic](auto tag_v) {
    using tensor_tag = typename decltype(tag_v)::type;
    const auto& diff = db::get<Tags::Error<tensor_tag>>(box);
    for (size_t i = 0; i < db::item_type<tensor_tag>::size(); ++i) {
      CHECK_ITERABLE_APPROX(diff[i], -1.0 * get<tensor_tag>(analytic)[i] + 5.0);
    }
  });
}
