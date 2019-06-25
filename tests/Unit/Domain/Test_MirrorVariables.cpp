// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/MirrorVariables.hpp"
#include "Domain/Tags.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarFieldTag1 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarFieldTag"; }
};
struct ScalarFieldTag2 : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "ScalarFieldTag"; }
};

using variables_tag =
    Tags::Variables<tmpl::list<ScalarFieldTag1, ScalarFieldTag2>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.MirrorVariables", "[Unit][Domain]") {
  using compute_mirrored_variables = ::Tags::InterfaceComputeItem<
      ::Tags::BoundaryDirectionsExterior<1>,
      ::Tags::MirrorVariables<1, ::Tags::BoundaryDirectionsInterior<1>,
                              variables_tag, tmpl::list<ScalarFieldTag2>>>;

  Element<1> element{ElementId<1>(0), {{Direction<1>::lower_xi(), {}}}};
  const auto direction = Direction<1>::upper_xi();

  db::item_type<variables_tag> vars{3, 0.};
  get(get<ScalarFieldTag1>(vars)) = DataVector{1., 2., 3.};
  get(get<ScalarFieldTag2>(vars)) = DataVector{4., 5., 6.};

  using face_vars_tag =
      ::Tags::Interface<::Tags::BoundaryDirectionsInterior<1>, variables_tag>;

  auto box =
      db::create<db::AddSimpleTags<::Tags::Element<1>, face_vars_tag>,
                 db::AddComputeTags<::Tags::BoundaryDirectionsExterior<1>,
                                    ::Tags::InterfaceComputeItem<
                                        ::Tags::BoundaryDirectionsExterior<1>,
                                        ::Tags::Direction<1>>,
                                    compute_mirrored_variables>>(
          std::move(element), db::item_type<face_vars_tag>{{direction, vars}});

  const auto& mirrored_vars =
      get<::Tags::Interface<::Tags::BoundaryDirectionsExterior<1>,
                            variables_tag>>(box)
          .at(direction);
  CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag1>(mirrored_vars)),
                        (DataVector{1., 2., 3.}));
  CHECK_ITERABLE_APPROX(get(get<ScalarFieldTag2>(mirrored_vars)),
                        (DataVector{-4., -5., -6.}));
}
