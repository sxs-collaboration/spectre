// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/optional.hpp>
#include <cstddef>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Actions/AddMeshVelocitySourceTerms.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, 3, Frame::Inertial>;
};

struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2>>;
  using sourced_variables = tmpl::list<>;
};

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags =
      tmpl::list<System::variables_tag,
                 db::add_tag_prefix<Tags::dt, System::variables_tag>,
                 domain::Tags::DivMeshVelocity>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<evolution::Actions::AddMeshVelocitySourceTerms>>>;
};

struct Metavariables {
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  enum class Phase { Initialization, Testing, Exit };
};

template <bool HasMeshVelocity>
void test() {
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;

  boost::optional<Scalar<DataVector>> div_frame_velocity{};
  if (HasMeshVelocity) {
    div_frame_velocity = Scalar<DataVector>{{{{5., 6.}}}};
  }

  typename db::add_tag_prefix<Tags::dt, System::variables_tag>::type dt_vars(
      2, 0.);
  typename System::variables_tag::type vars(2);
  get(get<Var1>(vars)) = 3.;
  for (size_t i = 0; i < 3; ++i) {
    get<Var2>(vars)[i] = 5. + i;
  }

  using simple_tags = typename component<Metavariables>::simple_tags;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component<Metavariables>>(
      &runner, 0, {vars, dt_vars, div_frame_velocity});
  ActionTesting::set_phase(make_not_null(&runner),
                           Metavariables::Phase::Testing);
  runner.next_action<component<Metavariables>>(0);

  const auto& box =
      ActionTesting::get_databox<component<Metavariables>, simple_tags>(runner,
                                                                        0);

  const DataVector expected = []() noexcept {
    if (HasMeshVelocity) {
      return DataVector{-5., -6.};
    }
    return DataVector{0., 0.};
  }();

  CHECK(get(db::get<Tags::dt<Var1>>(box)) == expected * get(get<Var1>(vars)));
  for (size_t i = 0; i < 3; ++i) {
    CHECK(db::get<Tags::dt<Var2>>(box).get(i) ==
          expected * get<Var2>(vars).get(i));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.AddMeshVelocitySourceTerms",
                  "[Unit][Evolution][Actions]") {
  test<true>();
  test<false>();
}
