// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/DataStructures/DataBox/TestHelpers.hpp"

class DataVector;
// IWYU pragma: no_forward_declare Tags::Flux

namespace {
struct Tag : db::SimpleTag {
  using type = double;
};

struct TensorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, 2>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Prefixes",
                  "[Unit][DataStructures]") {
  /// [dt_name]
  TestHelpers::db::test_prefix_tag<Tags::dt<Tag>>("dt(Tag)");
  /// [dt_name]
  /// [analytic_name]
  TestHelpers::db::test_prefix_tag<Tags::Analytic<Tag>>("Analytic(Tag)");
  /// [analytic_name]
  using Dim = tmpl::size_t<2>;
  using Frame = Frame::Inertial;
  using VariablesTag = Tags::Variables<tmpl::list<TensorTag>>;
  /// [flux_name]
  TestHelpers::db::test_prefix_tag<Tags::Flux<TensorTag, Dim, Frame>>(
      "Flux(TensorTag)");
  TestHelpers::db::test_prefix_tag<Tags::Flux<VariablesTag, Dim, Frame>>(
      "Flux(Variables(TensorTag))");
  /// [flux_name]
  /// [source_name]
  TestHelpers::db::test_prefix_tag<Tags::Source<Tag>>("Source(Tag)");
  /// [source_name]
  TestHelpers::db::test_prefix_tag<Tags::FixedSource<Tag>>("FixedSource(Tag)");
  /// [initial_name]
  TestHelpers::db::test_prefix_tag<Tags::Initial<Tag>>("Initial(Tag)");
  /// [initial_name]
  /// [normal_dot_flux_name]
  TestHelpers::db::test_prefix_tag<Tags::NormalDotFlux<Tag>>(
      "NormalDotFlux(Tag)");
  /// [normal_dot_flux_name]
  /// [normal_dot_numerical_flux_name]
  TestHelpers::db::test_prefix_tag<Tags::NormalDotNumericalFlux<Tag>>(
      "NormalDotNumericalFlux(Tag)");
  /// [normal_dot_numerical_flux_name]
  /// [next_name]
  TestHelpers::db::test_prefix_tag<Tags::Next<Tag>>("Next(Tag)");
  /// [next_name]
}
