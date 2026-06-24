// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Utilities/TMPL.hpp"

class DataVector;

namespace {
struct Tag : db::SimpleTag {
  using type = double;
};

struct TensorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, 2>;
};

struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.DataBox.Prefixes",
                  "[Unit][DataStructures]") {
  // [dt_name]
  TestHelpers::db::test_prefix_tag<Tags::dt<Tag>>("dt(Tag)");
  // [dt_name]
  using Dim = tmpl::size_t<2>;
  using Frame = Frame::Inertial;
  using VariablesTag = Tags::Variables<tmpl::list<TensorTag>>;
  // [covariant_deriv_name]
  TestHelpers::db::test_prefix_tag<
      Tags::covariant_deriv<TensorTag, Dim, Frame>>(
      "covariant_deriv(TensorTag)");
  // [covariant_deriv_name]
  // [second_covariant_deriv_name]
  TestHelpers::db::test_prefix_tag<
      Tags::second_covariant_deriv<ScalarTag, Dim, Frame>>(
      "second_covariant_deriv(ScalarTag)");
  TestHelpers::db::test_prefix_tag<
      Tags::second_covariant_deriv<TensorTag, Dim, Frame>>(
      "second_covariant_deriv(TensorTag)");
  // [second_covariant_deriv_name]

  // [flux_name]
  TestHelpers::db::test_prefix_tag<Tags::Flux<TensorTag, Dim, Frame>>(
      "Flux(TensorTag)");
  TestHelpers::db::test_prefix_tag<Tags::Flux<VariablesTag, Dim, Frame>>(
      "Flux(Variables(TensorTag))");
  // [flux_name]
  // [source_name]
  TestHelpers::db::test_prefix_tag<Tags::Source<Tag>>("Source(Tag)");
  // [source_name]
  TestHelpers::db::test_prefix_tag<Tags::FixedSource<Tag>>("FixedSource(Tag)");
  // [initial_name]
  TestHelpers::db::test_prefix_tag<Tags::Initial<Tag>>("Initial(Tag)");
  // [initial_name]
  // [normal_dot_flux_name]
  TestHelpers::db::test_prefix_tag<Tags::NormalDotFlux<Tag>>(
      "NormalDotFlux(Tag)");
  // [normal_dot_flux_name]
  // [normal_dot_numerical_flux_name]
  TestHelpers::db::test_prefix_tag<Tags::NormalDotNumericalFlux<Tag>>(
      "NormalDotNumericalFlux(Tag)");
  // [normal_dot_numerical_flux_name]
  TestHelpers::db::test_prefix_tag<Tags::Previous<Tag>>("Previous(Tag)");
  // [next_name]
  TestHelpers::db::test_prefix_tag<Tags::Next<Tag>>("Next(Tag)");
  // [next_name]
}
