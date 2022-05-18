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
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/Phase.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/TMPL.hpp"

namespace control_system::Tags {
struct FunctionsOfTimeInitialize;
}  // namespace control_system::Tags

namespace {
template <size_t MeshDim>
using TranslationMap =
    domain::CoordinateMaps::TimeDependent::Translation<MeshDim>;

using AffineMap = domain::CoordinateMaps::Affine;
using AffineMap2d =
    domain::CoordinateMaps::ProductOf2Maps<AffineMap, AffineMap>;
using AffineMap3d =
    domain::CoordinateMaps::ProductOf3Maps<AffineMap, AffineMap, AffineMap>;

template <size_t MeshDim, typename SourceFrame, typename TargetFrame>
using TimeIndependentMap = tmpl::conditional_t<
    MeshDim == 1, domain::CoordinateMap<SourceFrame, TargetFrame, AffineMap>,
    tmpl::conditional_t<
        MeshDim == 2,
        domain::CoordinateMap<SourceFrame, TargetFrame, AffineMap2d>,
        domain::CoordinateMap<SourceFrame, TargetFrame, AffineMap3d>>>;

template <size_t MeshDim>
using TimeDependentMap = domain::CoordinateMap<Frame::Grid, Frame::Inertial,
                                               TranslationMap<MeshDim>>;

template <size_t MeshDim, typename SourceFrame, typename TargetFrame>
struct CreateAffineMap;

template <typename SourceFrame, typename TargetFrame>
struct CreateAffineMap<1, SourceFrame, TargetFrame> {
  static TimeIndependentMap<1, SourceFrame, TargetFrame> apply() {
    return TimeIndependentMap<1, SourceFrame, TargetFrame>{
        AffineMap{-1.0, 1.0, 2.0, 7.2}};
  }
};

template <typename SourceFrame, typename TargetFrame>
struct CreateAffineMap<2, SourceFrame, TargetFrame> {
  static TimeIndependentMap<2, SourceFrame, TargetFrame> apply() {
    return TimeIndependentMap<2, SourceFrame, TargetFrame>{
        {AffineMap{-1.0, 1.0, -2.0, 2.2}, AffineMap{-1.0, 1.0, 2.0, 7.2}}};
  }
};

template <typename SourceFrame, typename TargetFrame>
struct CreateAffineMap<3, SourceFrame, TargetFrame> {
  static TimeIndependentMap<3, SourceFrame, TargetFrame> apply() {
    return TimeIndependentMap<3, SourceFrame, TargetFrame>{
        {AffineMap{-1.0, 1.0, -2.0, 2.2}, AffineMap{-1.0, 1.0, 2.0, 7.2},
         AffineMap{-1.0, 1.0, 1.0, 3.5}}};
  }
};

template <size_t MeshDim, typename SourceFrame, typename TargetFrame>
TimeIndependentMap<MeshDim, SourceFrame, TargetFrame> create_affine_map() {
  return CreateAffineMap<MeshDim, SourceFrame, TargetFrame>::apply();
}

template <size_t MeshDim>
TimeDependentMap<MeshDim> create_translation_map(
    const std::string& f_of_t_name) {
  return TimeDependentMap<MeshDim>{TranslationMap<MeshDim>{f_of_t_name}};
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
                    const ParallelComponent* const /*meta*/) {
    db::mutate<Tags::Time>(
        make_not_null(&box),
        [](const gsl::not_null<double*> time) { *time += 1.2; });
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
  using mutable_global_cache_tags =
      tmpl::list<domain::Tags::FunctionsOfTimeInitialize>;

  using simple_tags =
      db::AddSimpleTags<domain::Tags::InitialExtents<dim>,
                        domain::Tags::InitialRefinementLevels<dim>,
                        evolution::dg::Tags::Quadrature, Tags::Time>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<simple_tags>>>,

      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<::Actions::SetupDataBox,
                     evolution::dg::Initialization::Domain<dim>,
                     Actions::IncrementTime, Actions::IncrementTime>>>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  static constexpr size_t dim = Dim;

  using Phase = Parallel::Phase;
};

template <size_t Dim, bool TimeDependent>
void test(const Spectral::Quadrature quadrature) {
  CAPTURE(Dim);
  CAPTURE(TimeDependent);
  CAPTURE(quadrature);
  using metavars = Metavariables<Dim>;
  using component = Component<metavars>;

  static_assert(
      std::is_same_v<typename evolution::dg::Initialization::Domain<
                         Dim>::mutable_global_cache_tags,
                     tmpl::list<::domain::Tags::FunctionsOfTimeInitialize>>);
  static_assert(std::is_same_v<
                typename evolution::dg::Initialization::Domain<
                    Dim, false, true>::mutable_global_cache_tags,
                tmpl::list<control_system::Tags::FunctionsOfTimeInitialize>>);

  PUPable_reg(SINGLE_ARG(
      TimeIndependentMap<Dim, Frame::BlockLogical, Frame::Inertial>));
  PUPable_reg(
      SINGLE_ARG(TimeIndependentMap<Dim, Frame::BlockLogical, Frame::Grid>));
  PUPable_reg(SINGLE_ARG(
      TimeIndependentMap<Dim, Frame::ElementLogical, Frame::Inertial>));
  PUPable_reg(
      SINGLE_ARG(TimeIndependentMap<Dim, Frame::ElementLogical, Frame::Grid>));
  PUPable_reg(TimeDependentMap<Dim>);
  PUPable_reg(domain::FunctionsOfTime::PiecewisePolynomial<2>);

  const std::vector<std::array<size_t, Dim>> initial_extents{
      make_array<Dim>(4_st)};
  const std::vector<std::array<size_t, Dim>> initial_refinement{
      make_array<Dim>(0_st)};
  const size_t num_pts = pow<Dim>(4_st);
  const DataVector velocity{Dim, 3.6};
  const double initial_time = 0.0;
  const double expiration_time = 2.5;
  const std::string function_of_time_name = "Translation";
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time.insert(std::make_pair(
      function_of_time_name,
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time,
          std::array<DataVector, 3>{{{Dim, 0.0}, velocity, {Dim, 0.0}}},
          expiration_time)));

  std::vector<Block<Dim>> blocks{1};
  blocks[0] = Block<Dim>{
      std::make_unique<
          TimeIndependentMap<Dim, Frame::BlockLogical, Frame::Inertial>>(
          create_affine_map<Dim, Frame::BlockLogical, Frame::Inertial>()),
      0,
      {}};
  Domain<Dim> domain{std::move(blocks)};

  if (TimeDependent) {
    domain.inject_time_dependent_map_for_block(
        0, std::make_unique<TimeDependentMap<Dim>>(
               create_translation_map<Dim>(function_of_time_name)));
  }

  const ElementId<Dim> self_id(0);
  ActionTesting::MockRuntimeSystem<metavars> runner{
      {std::move(domain)}, {std::move(clone_unique_ptrs(functions_of_time))}};

  ActionTesting::emplace_component_and_initialize<component>(
      &runner, self_id,
      {initial_extents, initial_refinement, quadrature, initial_time});
  runner.set_phase(metavars::Phase::Testing);
  CHECK(ActionTesting::get_next_action_index<component>(runner, self_id) == 0);
  for (size_t i = 0; i < 2; ++i) {
    ActionTesting::next_action<component>(make_not_null(&runner), self_id);
  }

  // Set up data to be used for checking correctness
  const auto logical_to_grid_map =
      create_affine_map<Dim, Frame::ElementLogical, Frame::Grid>();
  const auto grid_to_inertial_map =
      create_translation_map<Dim>(function_of_time_name);
  const auto& logical_coords = ActionTesting::get_databox_tag<
      component, domain::Tags::Coordinates<Dim, Frame::ElementLogical>>(
      runner, self_id);

  const auto check_domain_tags_time_dependent = [&functions_of_time,
                                                 &grid_to_inertial_map,
                                                 &logical_coords,
                                                 &logical_to_grid_map, num_pts,
                                                 &runner, &self_id,
                                                 &velocity](const double time) {
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

    CHECK(
        ActionTesting::get_databox_tag<
            component, domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                     Frame::Grid>>(
            runner, self_id) == expected_logical_to_grid_inv_jacobian);

    const InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>
        expected_inv_jacobian_grid_to_inertial =
            grid_to_inertial_map.inv_jacobian(expected_grid_coords, time,
                                              functions_of_time);

    InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
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

    REQUIRE(
        ActionTesting::get_databox_tag<
            component, domain::Tags::CoordinatesMeshVelocityAndJacobians<Dim>>(
            runner, self_id)
            .has_value());
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
                 component, domain::Tags::InverseJacobian<
                                Dim, Frame::ElementLogical, Frame::Inertial>>(
                 runner, self_id)
                 .size();
         ++i) {
      CHECK(ActionTesting::get_databox_tag<
                component, domain::Tags::InverseJacobian<
                               Dim, Frame::ElementLogical, Frame::Inertial>>(
                runner, self_id)[i]
                .data() !=
            std::get<1>(coordinates_mesh_velocity_and_jacobians)[i].data());
    }
    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component, domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                     Frame::Inertial>>(
            runner, self_id)),
        expected_logical_to_inertial_inv_jacobian);

    const Scalar<DataVector> expected_logical_to_inertial_det_inv_jacobian =
        determinant(expected_logical_to_inertial_inv_jacobian);
    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component, domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                                    Frame::Inertial>>(runner,
                                                                      self_id)),
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
            .value(),
        std::get<3>(expected_coords_mesh_velocity_jacobians));

    for (size_t i = 0; i < Dim; ++i) {
      CHECK_ITERABLE_APPROX(
          (ActionTesting::get_databox_tag<component,
                                          domain::Tags::MeshVelocity<Dim>>(
               runner, self_id)
               .value()
               .get(i)),
          DataVector(num_pts, gsl::at(velocity, i)));
    }
  };

  const auto check_domain_tags_time_independent = [&logical_coords,
                                                   &logical_to_grid_map,
                                                   &runner, &self_id](
                                                      const double time) {
    const auto logical_to_inertial_map =
        create_affine_map<Dim, Frame::ElementLogical, Frame::Inertial>();
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

    CHECK(
        ActionTesting::get_databox_tag<
            component, domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                     Frame::Grid>>(
            runner, self_id) == expected_logical_to_grid_inv_jacobian);

    const InverseJacobian<DataVector, Dim, Frame::ElementLogical,
                          Frame::Inertial>
        expected_logical_to_inertial_inv_jacobian =
            logical_to_inertial_map.inv_jacobian(logical_coords);

    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component, domain::Tags::InverseJacobian<Dim, Frame::ElementLogical,
                                                     Frame::Inertial>>(
            runner, self_id)),
        expected_logical_to_inertial_inv_jacobian);
    for (size_t i = 0;
         i < ActionTesting::get_databox_tag<
                 component, domain::Tags::InverseJacobian<
                                Dim, Frame::ElementLogical, Frame::Inertial>>(
                 runner, self_id)
                 .size();
         ++i) {
      // Check that the `const_cast`s and set_data_ref inside the compute
      // tag functions worked correctly
      CHECK(ActionTesting::get_databox_tag<
                component, domain::Tags::InverseJacobian<
                               Dim, Frame::ElementLogical, Frame::Inertial>>(
                runner, self_id)[i]
                .data() ==
            ActionTesting::get_databox_tag<
                component, domain::Tags::InverseJacobian<
                               Dim, Frame::ElementLogical, Frame::Grid>>(
                runner, self_id)[i]
                .data());
    }

    const Scalar<DataVector> expected_logical_to_inertial_det_inv_jacobian =
        determinant(expected_logical_to_inertial_inv_jacobian);
    CHECK_ITERABLE_APPROX(
        (ActionTesting::get_databox_tag<
            component, domain::Tags::DetInvJacobian<Frame::ElementLogical,
                                                    Frame::Inertial>>(runner,
                                                                      self_id)),
        expected_logical_to_inertial_det_inv_jacobian);

    CHECK_FALSE(ActionTesting::get_databox_tag<component,
                                               domain::Tags::MeshVelocity<Dim>>(
                    runner, self_id)
                    .has_value());
    CHECK_FALSE(ActionTesting::get_databox_tag<component,
                                               domain::Tags::DivMeshVelocity>(
                    runner, self_id)
                    .has_value());
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
  for (const auto quadrature :
       {Spectral::Quadrature::Gauss, Spectral::Quadrature::GaussLobatto}) {
    test<1, true>(quadrature);
    test<2, true>(quadrature);
    test<3, true>(quadrature);

    test<1, false>(quadrature);
    test<2, false>(quadrature);
    test<3, false>(quadrature);
  }
}
}  // namespace
