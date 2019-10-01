// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Utilities/TMPL.hpp"

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
  CHECK(db::tag_name<Tags::dt<Tag>>() == "dt(" + db::tag_name<Tag>() + ")");
  /// [dt_name]
  /// [analytic_name]
  CHECK(db::tag_name<Tags::Analytic<Tag>>() ==
        "Analytic(" + db::tag_name<Tag>() + ")");
  /// [analytic_name]
  using Dim = tmpl::size_t<2>;
  using Frame = Frame::Inertial;
  using VariablesTag = Tags::Variables<tmpl::list<TensorTag>>;
  /// [flux_name]
  CHECK(db::tag_name<Tags::Flux<TensorTag, Dim, Frame>>() ==
        "Flux(" + db::tag_name<TensorTag>() + ")");
  CHECK(db::tag_name<Tags::Flux<VariablesTag, Dim, Frame>>() ==
        "Flux(" + db::tag_name<VariablesTag>() + ")");
  /// [flux_name]
  /// [source_name]
  CHECK(db::tag_name<Tags::Source<Tag>>() ==
        "Source(" + db::tag_name<Tag>() + ")");
  /// [source_name]
  CHECK(db::tag_name<Tags::FixedSource<Tag>>() ==
        "FixedSource(" + db::tag_name<Tag>() + ")");
  /// [initial_name]
  CHECK(db::tag_name<Tags::Initial<Tag>>() ==
        "Initial(" + db::tag_name<Tag>() + ")");
  /// [initial_name]
  /// [normal_dot_flux_name]
  CHECK(db::tag_name<Tags::NormalDotFlux<Tag>>() ==
        "NormalDotFlux(" + db::tag_name<Tag>() + ")");
  /// [normal_dot_flux_name]
  /// [normal_dot_numerical_flux_name]
  CHECK(db::tag_name<Tags::NormalDotNumericalFlux<Tag>>() ==
        "NormalDotNumericalFlux(" + db::tag_name<Tag>() + ")");
  /// [normal_dot_numerical_flux_name]
  /// [next_name]
  CHECK(db::tag_name<Tags::Next<Tag>>() == "Next(" + db::tag_name<Tag>() + ")");
  /// [next_name]
}
