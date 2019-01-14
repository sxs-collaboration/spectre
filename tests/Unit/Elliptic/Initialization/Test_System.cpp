// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <cstddef>
#include <string>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Initialization/System.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

template <typename Tag>
struct SomePrefix : db::PrefixTag, db::SimpleTag {
  static std::string name() noexcept { return "SomePrefix"; }
  using type = db::item_type<Tag>;
  using tag = Tag;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
  using variables_tag = db::add_tag_prefix<SomePrefix, fields_tag>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.System",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    Mesh<1> mesh{4, Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
    const auto box = Elliptic::Initialization::System<System<1>>::initialize(
        db::create<db::AddSimpleTags<Tags::Mesh<1>>>(mesh));

    const Scalar<DataVector> expected_initial_field{{{{4, 0.}}}};
    CHECK(get<ScalarFieldTag>(box) == expected_initial_field);
    // Only check that the variables are initialized
    get<SomePrefix<ScalarFieldTag>>(box);
    get<LinearSolver::Tags::OperatorAppliedTo<SomePrefix<ScalarFieldTag>>>(box);
  }
  {
    INFO("2D");
    Mesh<2> mesh{{{3, 4}},
                 Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
    const auto box = Elliptic::Initialization::System<System<2>>::initialize(
        db::create<db::AddSimpleTags<Tags::Mesh<2>>>(mesh));

    const Scalar<DataVector> expected_initial_field{{{{12, 0.}}}};
    CHECK(get<ScalarFieldTag>(box) == expected_initial_field);
    // Only check that the variables are initialized
    get<SomePrefix<ScalarFieldTag>>(box);
    get<LinearSolver::Tags::OperatorAppliedTo<SomePrefix<ScalarFieldTag>>>(box);
  }
  {
    INFO("3D");
    Mesh<3> mesh{{{3, 4, 2}},
                 Spectral::Basis::Legendre,
                 Spectral::Quadrature::GaussLobatto};
    const auto box = Elliptic::Initialization::System<System<3>>::initialize(
        db::create<db::AddSimpleTags<Tags::Mesh<3>>>(mesh));

    const Scalar<DataVector> expected_initial_field{{{{24, 0.}}}};
    CHECK(get<ScalarFieldTag>(box) == expected_initial_field);
    // Only check that the variables are initialized
    get<SomePrefix<ScalarFieldTag>>(box);
    get<LinearSolver::Tags::OperatorAppliedTo<SomePrefix<ScalarFieldTag>>>(box);
  }
}
