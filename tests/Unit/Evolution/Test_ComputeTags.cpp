// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Evolution/ComputeTags.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Time/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {

struct FieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct AnalyticSolution {
  tuples::TaggedTuple<FieldTag> variables(
      const tnsr::I<DataVector, 1>& x, const double t,
      const tmpl::list<FieldTag> /*meta*/) const noexcept {
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
      "Analytic(Variables(Analytic(FieldTag)))");
}
