// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/optional.hpp>
#include <cstddef>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Actions/AddMeshVelocityNonconservative.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim, typename GradientsTags>
struct System {
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2<Dim>>>;
  using gradients_tags = GradientsTags;
};

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = int;
  using simple_tags = tmpl::list<
      db::add_tag_prefix<
          ::Tags::deriv, typename Metavariables::system::variables_tag,
          tmpl::size_t<Metavariables::volume_dim>, Frame::Inertial>,
      db::add_tag_prefix<Tags::dt,
                         typename Metavariables::system::variables_tag>,
      domain::Tags::MeshVelocity<Metavariables::volume_dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<evolution::Actions::AddMeshVelocityNonconservative>>>;
};

template <typename System>
struct Metavariables {
  static constexpr size_t volume_dim = System::volume_dim;
  using component_list = tmpl::list<component<Metavariables>>;
  using system = System;
  enum class Phase { Initialization, Testing, Exit };
};

template <bool HasMeshVelocity, size_t Dim, typename GradientsTags>
void test() noexcept {
  using system = System<Dim, GradientsTags>;
  using metavars = Metavariables<system>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;

  boost::optional<tnsr::I<DataVector, Dim, Frame::Inertial>> frame_velocity{};
  if (HasMeshVelocity) {
    frame_velocity = tnsr::I<DataVector, Dim, Frame::Inertial>{};
    for (size_t d = 0; d < Dim; ++d) {
      frame_velocity->get(d) = DataVector{(1. + d) * 5., (1. + d) * 6.};
    }
  }

  db::item_type<db::add_tag_prefix<Tags::dt, typename system::variables_tag>>
      dt_vars(2, 0.);
  using deriv_tag =
      db::add_tag_prefix<::Tags::deriv, typename system::variables_tag,
                         tmpl::size_t<Dim>, Frame::Inertial>;
  db::item_type<deriv_tag> deriv_vars(2);
  for (size_t i = 0; i < Dim; ++i) {
    get<::Tags::deriv<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(deriv_vars)
        .get(i) = (1. + i) * 3.;
    for (size_t d = 0; d < Dim; ++d) {
      get<::Tags::deriv<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>(
          deriv_vars)
          .get(d, i) = (1. + d) * (5. + i);
    }
  }

  using simple_tags = typename component<metavars>::simple_tags;
  MockRuntimeSystem runner{{}};
  ActionTesting::emplace_component_and_initialize<component<metavars>>(
      &runner, 0, {deriv_vars, dt_vars, frame_velocity});
  ActionTesting::set_phase(make_not_null(&runner), metavars ::Phase::Testing);
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner), 0);

  const auto& box =
      ActionTesting::get_databox<component<metavars>, simple_tags>(runner, 0);

  for (size_t i = 0; i < Dim; ++i) {
    DataVector expected{2, 0.};
    if (HasMeshVelocity) {
      for (size_t d = 0; d < Dim; ++d) {
        expected +=
            frame_velocity->get(d) *
            get<::Tags::deriv<Var2<Dim>, tmpl::size_t<Dim>, Frame::Inertial>>(
                deriv_vars)
                .get(d, i);
      }
    }
    CHECK_ITERABLE_APPROX(db::get<Tags::dt<Var2<Dim>>>(box).get(i), expected);
  }

  DataVector expected{2, 0.};
  if (HasMeshVelocity) {
    for (size_t d = 0; d < Dim; ++d) {
      expected += frame_velocity->get(d) *
                  get<::Tags::deriv<Var1, tmpl::size_t<Dim>, Frame::Inertial>>(
                      deriv_vars)
                      .get(d);
    }
  }
  CHECK_ITERABLE_APPROX(get(db::get<Tags::dt<Var1>>(box)), expected);
}

template <size_t Dim>
void test_dispatch() noexcept {
  test<true, Dim, tmpl::list<Var1, Var2<Dim>>>();
  test<false, Dim, tmpl::list<Var1, Var2<Dim>>>();

  test<true, Dim, tmpl::list<Var1>>();
  test<false, Dim, tmpl::list<Var1>>();

  test<true, Dim, tmpl::list<Var2<Dim>>>();
  test<false, Dim, tmpl::list<Var2<Dim>>>();

  test<true, Dim, tmpl::list<>>();
  test<false, Dim, tmpl::list<>>();
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.AddMeshVelocityNonconservative",
                  "[Unit][Evolution][Actions]") {
  test_dispatch<1>();
  test_dispatch<2>();
  test_dispatch<3>();
}
