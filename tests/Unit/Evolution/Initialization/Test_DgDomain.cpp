// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.hpp"
#include "Domain/CoordinateMaps/TimeDependent/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace {
using TranslationMap = domain::CoordinateMaps::TimeDependent::Translation;
using TranslationMap2d =
    domain::CoordinateMaps::TimeDependent::ProductOf2Maps<TranslationMap,
                                                          TranslationMap>;
using TranslationMap3d = domain::CoordinateMaps::TimeDependent::ProductOf3Maps<
    TranslationMap, TranslationMap, TranslationMap>;

using AffineMap = domain::CoordinateMaps::Affine;
using AffineMap2d =
    domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
using AffineMap3d =
    domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;

template <size_t MeshDim, typename TargetFrame>
using TimeIndependentMap = tmpl::conditional_t<
    MeshDim == 1, domain::CoordinateMap<Frame::Logical, TargetFrame, AffineMap>,
    tmpl::conditional_t<
        MeshDim == 2,
        domain::CoordinateMap<Frame::Logical, TargetFrame, AffineMap2d>,
        domain::CoordinateMap<Frame::Logical, TargetFrame, AffineMap3d>>>;

template <size_t MeshDim>
using TimeDependentMap = tmpl::conditional_t<
    MeshDim == 1,
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>,
    tmpl::conditional_t<
        MeshDim == 2,
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap2d>,
        domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap3d>>>;

template <size_t MeshDim, typename TargetFrame>
struct CreateAffineMap;

template <typename TargetFrame>
struct CreateAffineMap<1, TargetFrame> {
  static TimeIndependentMap<1, TargetFrame> apply() {
    return TimeIndependentMap<1, TargetFrame>{AffineMap{-1.0, 1.0, 2.0, 7.2}};
  }
};

template <typename TargetFrame>
struct CreateAffineMap<2, TargetFrame> {
  static TimeIndependentMap<2, TargetFrame> apply() {
    return TimeIndependentMap<2, TargetFrame>{
        {AffineMap{-1.0, 1.0, -2.0, 2.2}, AffineMap{-1.0, 1.0, 2.0, 7.2}}};
  }
};

template <typename TargetFrame>
struct CreateAffineMap<3, TargetFrame> {
  static TimeIndependentMap<3, TargetFrame> apply() {
    return TimeIndependentMap<3, TargetFrame>{{AffineMap{-1.0, 1.0, -2.0, 2.2},
                                               AffineMap{-1.0, 1.0, 2.0, 7.2},
                                               AffineMap{-1.0, 1.0, 1.0, 3.5}}};
  }
};

template <size_t MeshDim, typename TargetFrame>
TimeIndependentMap<MeshDim, TargetFrame> create_affine_map() {
  return CreateAffineMap<MeshDim, TargetFrame>::apply();
}

template <size_t MeshDim>
TimeDependentMap<MeshDim> create_translation_map(
    const std::array<std::string, 3>& f_of_t_names);

template <>
TimeDependentMap<1> create_translation_map<1>(
    const std::array<std::string, 3>& f_of_t_names) {
  return TimeDependentMap<1>{TranslationMap{f_of_t_names[0]}};
}

template <>
TimeDependentMap<2> create_translation_map<2>(
    const std::array<std::string, 3>& f_of_t_names) {
  return TimeDependentMap<2>{
      {TranslationMap{f_of_t_names[0]}, TranslationMap{f_of_t_names[1]}}};
}

template <>
TimeDependentMap<3> create_translation_map<3>(
    const std::array<std::string, 3>& f_of_t_names) {
  return TimeDependentMap<3>{{TranslationMap{f_of_t_names[0]},
                              TranslationMap{f_of_t_names[1]},
                              TranslationMap{f_of_t_names[2]}}};
}

namespace Actions {
struct IncrementTime {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            typename ActionList, typename ParallelComponent,
            typename ArrayIndex>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::GlobalCache<Metavariables>& /*cache*/,
                    const ArrayIndex& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    db::mutate<Tags::Time>(
        make_not_null(&box),
        [](const gsl::not_null<double*> time) noexcept { *time += 1.2; });
    return std::forward_as_tuple(std::move(box));
  }
};
}  // namespace Actions

template <typename Metavariables>
struct Component {
  using metavariables = Metavariables;
  static constexpr size_t dim = metavariables::dim;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<dim>>;

  using simple_tags =
      db::AddSimpleTags<domain::Tags::InitialExtents<dim>,
                        domain::Tags::InitialRefinementLevels<dim>,
                        domain::Tags::InitialFunctionsOfTime<dim>, Tags::Time>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<evolution::dg::Initialization::Domain<dim>,
                     Actions::IncrementTime, Actions::IncrementTime>>>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  static constexpr size_t dim = Dim;

  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim, bool TimeDependent>
void test() noexcept {
  using metavars = Metavariables<Dim>;
  using component = Component<metavars>;

  PUPable_reg(SINGLE_ARG(TimeIndependentMap<Dim, Frame::Inertial>));
  PUPable_reg(SINGLE_ARG(TimeIndependentMap<Dim, Frame::Grid>));
  PUPable_reg(TimeDependentMap<Dim>);
  PUPable_reg(domain::FunctionsOfTime::PiecewisePolynomial<2>);

  const std::vector<std::array<size_t, Dim>> initial_extents{
      make_array<Dim>(4_st)};
  const std::vector<std::array<size_t, Dim>> initial_refinement{
      make_array<Dim>(0_st)};
  const size_t num_pts = pow<Dim>(4_st);
  const std::array<double, 3> velocity{{1.2, 0.2, -8.9}};
  const double initial_time = 0.0;
  const std::array<std::string, 3> functions_of_time_names{
      {"TranslationX", "TranslationY", "TranslationZ"}};
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  for (size_t i = 0; i < velocity.size(); ++i) {
    functions_of_time.insert(std::make_pair(
        gsl::at(functions_of_time_names, i),
        std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
            initial_time, std::array<DataVector, 3>{
                              {{0.0}, {gsl::at(velocity, i)}, {0.0}}})));
  }

  std::vector<Block<Dim>> blocks{1};
  blocks[0] =
      Block<Dim>{std::make_unique<TimeIndependentMap<Dim, Frame::Inertial>>(
                     create_affine_map<Dim, Frame::Inertial>()),
                 0,
                 {}};
  Domain<Dim> domain{std::move(blocks)};

  if (TimeDependent) {
    domain.inject_time_dependent_map_for_block(
        0, std::make_unique<TimeDependentMap<Dim>>(
               create_translation_map<Dim>(functions_of_time_names)));
  }

  const ElementId<Dim> self_id(0);
  ActionTesting::MockRuntimeSystem<metavars> runner{{std::move(domain)}};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, self_id,
      {initial_extents, initial_refinement,
       std::move(clone_unique_ptrs(functions_of_time)), initial_time});
  runner.set_phase(metavars::Phase::Testing);
  CHECK(ActionTesting::get_next_action_index<component>(runner, self_id) == 0);
  ActionTesting::next_action<component>(make_not_null(&runner), self_id);

  // Set up data to be used for checking correctness
  const auto logical_to_grid_map = create_affine_map<Dim, Frame::Grid>();
  const auto grid_to_inertial_map =
      create_translation_map<Dim>(functions_of_time_names);
  const auto& logical_coords = ActionTesting::get_databox_tag<
      component, domain::Tags::Coordinates<Dim, Frame::Logical>>(runner,
                                                                 self_id);

  const auto check_domain_tags_time_dependent = [&functions_of_time,
                                                 &grid_to_inertial_map,
                                                 &logical_coords,
                                                 &logical_to_grid_map, num_pts,
                                                 &runner, &self_id,
                                                 &velocity](const double
                                                                time) noexcept {
    REQUIRE(ActionTesting::get_databox_tag<component, Tags::Time>(
                runner, self_id) == time);
    CHECK(ActionTesting::get_databox_tag<
              component, domain::Tags::Coordinates<Dim, Frame::Inertial>>(
              runner, self_id) ==
          grid_to_inertial_map(logical_to_grid_map(logical_coords), time,
                               functions_of_time));

    const auto expected_grid_coords = logical_to_grid_map(logical_coords);
    const tnsr::I<DataVector, Dim, Frame::Inertial> expected_coords =
        grid_to_inertial_map(expected_grid_coords, time, functions_of_time);

    CHECK(ActionTesting::get_databox_tag<
              component, domain::Tags::Coordinates<Dim, Frame::Grid>>(
              runner, self_id) == expected_grid_coords);

    const auto expected_logical_to_grid_inv_jacobian =
        logical_to_grid_map.inv_jacobian(logical_coords);

    CHECK(ActionTesting::get_databox_tag<
              component,
              domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Grid>>(
              runner, self_id) == expected_logical_to_grid_inv_jacobian);

    const InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>
        expected_inv_jacobian_grid_to_inertial =
            grid_to_inertial_map.inv_jacobian(expected_grid_coords, time,
                                              functions_of_time);

    InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
        expected_logical_to_inertial_inv_jacobian{num_pts};

    for (size_t logical_i = 0; logical_i < Dim; ++logical_i) {
      for (size_t inertial_i = 0; inertial_i < Dim; ++inertial_i) {
        expected_logical_to_inertial_inv_jacobian.get(logical_i, inertial_i) =
            expected_logical_to_grid_inv_jacobian.get(logical_i, 0) *
            expected_inv_jacobian_grid_to_inertial.get(0, inertial_i);
        for (size_t grid_i = 1; grid_i < Dim; ++grid_i) {
          expected_logical_to_inertial_inv_jacobian.get(logical_i,
                                                        inertial_i) +=
              expected_logical_to_grid_inv_jacobian.get(logical_i, grid_i) *
              expected_inv_jacobian_grid_to_inertial.get(grid_i, inertial_i);
        }
      }
    }

    REQUIRE(static_cast<bool>(
        ActionTesting::get_databox_tag<
            component, domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(
            runner, self_id)));
    const auto& coordinates_mesh_velocity_and_jacobians =
        *ActionTesting::get_databox_tag<
            component, domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(
            runner, self_id);

    for (size_t i = 0; i < Dim; ++i) {
      // Check that the `const_cast`s and set_data_ref inside the compute tag
      // functions worked correctly
      CHECK(ActionTesting::get_databox_tag<
                component, domain::Tags::Coordinates<Dim, Frame::Inertial>>(
                runner, self_id)
                .get(i)
                .data() ==
            std::get<0>(coordinates_mesh_velocity_and_jacobians).get(i).data());
    }
    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component, domain::Tags::Coordinates<Dim, Frame::Inertial>>(
            runner, self_id)),
        expected_coords);

    for (size_t i = 0;
         i < ActionTesting::get_databox_tag<
                 component, domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                          Frame::Inertial>>(
                 runner, self_id)
                 .size();
         ++i) {
      CHECK(ActionTesting::get_databox_tag<
                component, domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                         Frame::Inertial>>(
                runner, self_id)[i]
                .data() !=
            std::get<1>(coordinates_mesh_velocity_and_jacobians)[i].data());
    }
    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component, domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                     Frame::Inertial>>(
            runner, self_id)),
        expected_logical_to_inertial_inv_jacobian);

    const Scalar<DataVector> expected_logical_to_inertial_det_inv_jacobian =
        determinant(expected_logical_to_inertial_inv_jacobian);
    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component,
            domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>(
            runner, self_id)),
        expected_logical_to_inertial_det_inv_jacobian);

    const auto expected_coords_mesh_velocity_jacobians =
        grid_to_inertial_map.coords_frame_velocity_jacobians(
            ActionTesting::get_databox_tag<
                component, domain::Tags::Coordinates<Dim, Frame::Grid>>(
                runner, self_id),
            ActionTesting::get_databox_tag<component, ::Tags::Time>(runner,
                                                                    self_id),
            ActionTesting::get_databox_tag<component,
                                           domain::Tags::FunctionsOfTime>(
                runner, self_id));

    for (size_t i = 0; i < Dim; ++i) {
      // Check that the `const_cast`s and set_data_ref inside the compute tag
      // functions worked correctly
      CHECK(ActionTesting::get_databox_tag<component,
                                           domain::Tags::MeshVelocity<Dim>>(
                runner, self_id)
                ->get(i)
                .data() ==
            std::get<3>(coordinates_mesh_velocity_and_jacobians).get(i).data());
    }
    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<component,
                                        domain::Tags::MeshVelocity<Dim>>(
             runner, self_id))
            .get(),
        std::get<3>(expected_coords_mesh_velocity_jacobians));

    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          (ActionTesting::get_databox_tag<component,
                                          domain::Tags::MeshVelocity<Dim>>(
               runner, self_id)
               .get()
               .get(i)),
          DataVector(num_pts, gsl::at(velocity, i)));
    }
  };

  const auto check_domain_tags_time_independent = [&logical_coords,
                                                   &logical_to_grid_map,
                                                   &runner, &self_id](
                                                      const double
                                                          time) noexcept {
    const auto logical_to_inertial_map =
        create_affine_map<Dim, Frame::Inertial>();
    REQUIRE(ActionTesting::get_databox_tag<component, Tags::Time>(
                runner, self_id) == time);
    CHECK(ActionTesting::get_databox_tag<
              component, domain::Tags::Coordinates<Dim, Frame::Inertial>>(
              runner, self_id) == logical_to_inertial_map(logical_coords));

    const auto expected_grid_coords = logical_to_grid_map(logical_coords);
    CHECK(ActionTesting::get_databox_tag<
              component, domain::Tags::Coordinates<Dim, Frame::Grid>>(
              runner, self_id) == expected_grid_coords);
    for (size_t i = 0; i < Dim; ++i) {
      CHECK(ActionTesting::get_databox_tag<
                component, domain::Tags::Coordinates<Dim, Frame::Inertial>>(
                runner, self_id)
                .get(i) == expected_grid_coords.get(i));
    }

    for (size_t i = 0; i < Dim; ++i) {
      // Check that the `const_cast`s and set_data_ref inside the compute
      // tag functions worked correctly
      CHECK(ActionTesting::get_databox_tag<
                component, domain::Tags::Coordinates<Dim, Frame::Inertial>>(
                runner, self_id)
                .get(i)
                .data() ==
            ActionTesting::get_databox_tag<
                component, domain::Tags::Coordinates<Dim, Frame::Grid>>(runner,
                                                                        self_id)
                .get(i)
                .data());
    }

    const auto expected_logical_to_grid_inv_jacobian =
        logical_to_grid_map.inv_jacobian(logical_coords);

    CHECK(ActionTesting::get_databox_tag<
              component,
              domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Grid>>(
              runner, self_id) == expected_logical_to_grid_inv_jacobian);

    const InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial>
        expected_logical_to_inertial_inv_jacobian =
            logical_to_inertial_map.inv_jacobian(logical_coords);

    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component, domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                     Frame::Inertial>>(
            runner, self_id)),
        expected_logical_to_inertial_inv_jacobian);
    for (size_t i = 0;
         i < ActionTesting::get_databox_tag<
                 component, domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                          Frame::Inertial>>(
                 runner, self_id)
                 .size();
         ++i) {
      // Check that the `const_cast`s and set_data_ref inside the compute
      // tag functions worked correctly
      CHECK(
          ActionTesting::get_databox_tag<
              component, domain::Tags::InverseJacobian<Dim, Frame::Logical,
                                                       Frame::Inertial>>(
              runner, self_id)[i]
              .data() ==
          ActionTesting::get_databox_tag<
              component,
              domain::Tags::InverseJacobian<Dim, Frame::Logical, Frame::Grid>>(
              runner, self_id)[i]
              .data());
    }

    const Scalar<DataVector> expected_logical_to_inertial_det_inv_jacobian =
        determinant(expected_logical_to_inertial_inv_jacobian);
    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component,
            domain::Tags::DetInvJacobian<Frame::Logical, Frame::Inertial>>(
            runner, self_id)),
        expected_logical_to_inertial_det_inv_jacobian);

    CHECK_FALSE(static_cast<bool>(
        ActionTesting::get_databox_tag<component,
                                       domain::Tags::MeshVelocity<Dim>>(
            runner, self_id)));
    CHECK_FALSE(static_cast<bool>(
        ActionTesting::get_databox_tag<component,
                                       domain::Tags::DivMeshVelocity>(
            runner, self_id)));
  };

  if (TimeDependent) {
    check_domain_tags_time_dependent(0.0);

    ActionTesting::next_action<component>(make_not_null(&runner), self_id);
    check_domain_tags_time_dependent(1.2);

    ActionTesting::next_action<component>(make_not_null(&runner), self_id);
    check_domain_tags_time_dependent(2.4);
  } else {
    check_domain_tags_time_independent(0.0);

    ActionTesting::next_action<component>(make_not_null(&runner), self_id);
    check_domain_tags_time_independent(1.2);

    ActionTesting::next_action<component>(make_not_null(&runner), self_id);
    check_domain_tags_time_independent(2.4);
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.Initialization.DgDomain",
                  "[Parallel][Unit]") {
  test<1, true>();
  test<2, true>();
  test<3, true>();

  test<1, false>();
  test<2, false>();
  test<3, false>();
}
}  // namespace
