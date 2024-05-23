// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
#include <optional>
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
#include "Domain/CoordinateMaps/Tags.hpp"
#include "Domain/CoordinateMaps/TimeDependent/Translation.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "Domain/TagsTimeDependent.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/Domain/CoordinateMaps/TestMapHelpers.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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
  static Parallel::iterable_action_return_t apply(
      DataBox& box, const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      const Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, const ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {
    db::mutate<Tags::Time>(
        [](const gsl::not_null<double*> time) { *time += 1.2; },
        make_not_null(&box));
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
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
          tmpl::list<Initialization::Actions::InitializeItems<
                         evolution::dg::Initialization::Domain<dim>>,
                     Actions::IncrementTime, Actions::IncrementTime>>>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<Component<Metavariables>>;
  static constexpr size_t dim = Dim;

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
                    Dim, true>::mutable_global_cache_tags,
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
  runner.set_phase(Parallel::Phase::Testing);
  CHECK(ActionTesting::get_next_action_index<component>(runner, self_id) == 0);
  ActionTesting::next_action<component>(make_not_null(&runner), self_id);

  CHECK(ActionTesting::get_databox_tag<component,
                                       domain::Tags::NeighborMesh<Dim>>(runner,
                                                                        self_id)
            .empty());

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

namespace test_projectors {
struct TestMetavariables {
  using component_list = tmpl::list<>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<1>>;
};

using items_type =
    tuples::TaggedTuple<Parallel::Tags::GlobalCacheImpl<TestMetavariables>,
                        ::domain::Tags::ElementMap<1, Frame::Grid>,
                        ::domain::CoordinateMaps::Tags::CoordinateMap<
                            1, Frame::Grid, Frame::Inertial>,
                        ::domain::Tags::Element<1>>;

using TranslationMap = domain::CoordinateMaps::TimeDependent::Translation<1>;
using AffineMap = domain::CoordinateMaps::Affine;
template <typename TargetFrame>
using TimeIndependentMap =
    domain::CoordinateMap<Frame::BlockLogical, TargetFrame, AffineMap>;
using TimeDependentMap =
    domain::CoordinateMap<Frame::Grid, Frame::Inertial, TranslationMap>;
using GridToInertialMap =
    domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 1>;

template <typename TargetFrame>
TimeIndependentMap<TargetFrame> create_affine_map() {
  return TimeIndependentMap<TargetFrame>{AffineMap{-1.0, 1.0, 2.0, 7.2}};
}

template <bool IsTimeDependent>
Parallel::GlobalCache<TestMetavariables> make_global_cache() {
  std::vector<Block<1>> blocks{1};
  blocks[0] = Block<1>{std::make_unique<TimeIndependentMap<Frame::Inertial>>(
                           create_affine_map<Frame::Inertial>()),
                       0,
                       {}};
  Domain<1> domain{std::move(blocks)};

  if (IsTimeDependent) {
    domain.inject_time_dependent_map_for_block(
        0, std::make_unique<TimeDependentMap>(
               TimeDependentMap(TranslationMap("Translation"))));
  }

  tuples::TaggedTuple<domain::Tags::Domain<1>> const_global_cache_items(
      std::move(domain));

  return {std::move(const_global_cache_items)};
}

template <bool IsTimeDependent>
void check_maps(const ElementMap<1, Frame::Grid>& element_map,
                const GridToInertialMap& grid_to_inertial_map) {
  const auto expected_block_map = create_affine_map<Frame::Grid>();
  CHECK(are_maps_equal(expected_block_map, element_map.block_map()));
  if constexpr (IsTimeDependent) {
    const auto expected_grid_to_inertial_map =
        TimeDependentMap{TranslationMap("Translation")};
    CHECK(are_maps_equal(expected_grid_to_inertial_map, grid_to_inertial_map));
  } else {
    const auto expected_grid_to_inertial_map =
        ::domain::make_coordinate_map<Frame::Grid, Frame::Inertial>(
            ::domain::CoordinateMaps::Identity<1>{});
    CHECK(are_maps_equal(expected_grid_to_inertial_map, grid_to_inertial_map));
  }
}

template <bool IsTimeDependent>
void test_p_refine() {
  auto global_cache = make_global_cache<IsTimeDependent>();
  const ElementId<1> element_id{0};
  Element<1> element{element_id, DirectionMap<1, Neighbors<1>>{}};
  const Domain<1>& domain = get<::domain::Tags::Domain<1>>(global_cache);
  const auto& my_block = domain.blocks()[element_id.block_id()];
  ElementMap<1, Frame::Grid> element_map{
      element_id, my_block.is_time_dependent()
                      ? my_block.moving_mesh_logical_to_grid_map().get_clone()
                      : my_block.stationary_map().get_to_grid_frame()};
  std::unique_ptr<GridToInertialMap> grid_to_inertial_map = nullptr;
  if (my_block.is_time_dependent()) {
    grid_to_inertial_map =
        my_block.moving_mesh_grid_to_inertial_map().get_clone();
  } else {
    grid_to_inertial_map =
        ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            ::domain::CoordinateMaps::Identity<1>{});
  }

  auto box = db::create<
      db::AddSimpleTags<Parallel::Tags::GlobalCacheImpl<TestMetavariables>,
                        ::domain::Tags::ElementMap<1, Frame::Grid>,
                        ::domain::CoordinateMaps::Tags::CoordinateMap<
                            1, Frame::Grid, Frame::Inertial>,
                        ::domain::Tags::Element<1>>,
      tmpl::list<Parallel::Tags::FromGlobalCache<domain::Tags::Domain<1>>>>(
      &global_cache, std::move(element_map), std::move(grid_to_inertial_map),
      std::move(element));

  const Mesh<1> mesh{2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  db::mutate_apply<evolution::dg::Initialization::ProjectDomain<1>>(
      make_not_null(&box), std::make_pair(mesh, element));
  check_maps<IsTimeDependent>(
      db::get<::domain::Tags::ElementMap<1, Frame::Grid>>(box),
      db::get<::domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                            Frame::Inertial>>(
          box));
}

template <bool IsTimeDependent>
void test_split() {
  auto global_cache = make_global_cache<IsTimeDependent>();

  const ElementId<1> parent_id{0};
  const ElementId<1> child_1_id{0, std::array{SegmentId{1, 0}}};
  const ElementId<1> child_2_id{0, std::array{SegmentId{1, 1}}};

  Element<1> parent{parent_id, DirectionMap<1, Neighbors<1>>{}};
  const Domain<1>& domain = get<::domain::Tags::Domain<1>>(global_cache);
  const auto& my_block = domain.blocks()[parent_id.block_id()];
  ElementMap<1, Frame::Grid> element_map{
      parent_id, my_block.is_time_dependent()
                     ? my_block.moving_mesh_logical_to_grid_map().get_clone()
                     : my_block.stationary_map().get_to_grid_frame()};
  std::unique_ptr<GridToInertialMap> grid_to_inertial_map = nullptr;
  if (my_block.is_time_dependent()) {
    grid_to_inertial_map =
        my_block.moving_mesh_grid_to_inertial_map().get_clone();
  } else {
    grid_to_inertial_map =
        ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            ::domain::CoordinateMaps::Identity<1>{});
  }
  const items_type parent_items{&global_cache, std::move(element_map),
                                std::move(grid_to_inertial_map),
                                std::move(parent)};

  Element<1> child_1{
      child_1_id,
      DirectionMap<1, Neighbors<1>>{std::pair{
          Direction<1>::upper_xi(),
          Neighbors<1>{std::unordered_set{child_2_id}, OrientationMap<1>{}}}}};
  auto child_1_box = db::create<
      db::AddSimpleTags<Parallel::Tags::GlobalCacheImpl<TestMetavariables>,
                        ::domain::Tags::ElementMap<1, Frame::Grid>,
                        ::domain::CoordinateMaps::Tags::CoordinateMap<
                            1, Frame::Grid, Frame::Inertial>,
                        ::domain::Tags::Element<1>>,
      tmpl::list<Parallel::Tags::FromGlobalCache<domain::Tags::Domain<1>>>>(
      &global_cache, ElementMap<1, Frame::Grid>{},
      std::unique_ptr<GridToInertialMap>{nullptr}, std::move(child_1));

  db::mutate_apply<evolution::dg::Initialization::ProjectDomain<1>>(
      make_not_null(&child_1_box), parent_items);
  check_maps<IsTimeDependent>(
      db::get<::domain::Tags::ElementMap<1, Frame::Grid>>(child_1_box),
      db::get<::domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                            Frame::Inertial>>(
          child_1_box));

  Element<1> child_2{
      child_2_id,
      DirectionMap<1, Neighbors<1>>{std::pair{
          Direction<1>::lower_xi(),
          Neighbors<1>{std::unordered_set{child_1_id}, OrientationMap<1>{}}}}};
  auto child_2_box = db::create<
      db::AddSimpleTags<Parallel::Tags::GlobalCacheImpl<TestMetavariables>,
                        ::domain::Tags::ElementMap<1, Frame::Grid>,
                        ::domain::CoordinateMaps::Tags::CoordinateMap<
                            1, Frame::Grid, Frame::Inertial>,
                        ::domain::Tags::Element<1>>,
      tmpl::list<Parallel::Tags::FromGlobalCache<domain::Tags::Domain<1>>>>(
      &global_cache, ElementMap<1, Frame::Grid>{},
      std::unique_ptr<GridToInertialMap>{nullptr}, std::move(child_2));

  db::mutate_apply<evolution::dg::Initialization::ProjectDomain<1>>(
      make_not_null(&child_2_box), parent_items);
  check_maps<IsTimeDependent>(
      db::get<::domain::Tags::ElementMap<1, Frame::Grid>>(child_2_box),
      db::get<::domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                            Frame::Inertial>>(
          child_2_box));
}

template <bool IsTimeDependent>
void test_join() {
  auto global_cache = make_global_cache<IsTimeDependent>();

  const ElementId<1> parent_id{0};
  const ElementId<1> child_1_id{0, std::array{SegmentId{1, 0}}};
  const ElementId<1> child_2_id{0, std::array{SegmentId{1, 1}}};

  Element<1> child_1{
      child_1_id,
      DirectionMap<1, Neighbors<1>>{std::pair{
          Direction<1>::upper_xi(),
          Neighbors<1>{std::unordered_set{child_2_id}, OrientationMap<1>{}}}}};
  Element<1> child_2{
      child_2_id,
      DirectionMap<1, Neighbors<1>>{std::pair{
          Direction<1>::lower_xi(),
          Neighbors<1>{std::unordered_set{child_1_id}, OrientationMap<1>{}}}}};
  const Domain<1>& domain = get<::domain::Tags::Domain<1>>(global_cache);
  const auto& my_block = domain.blocks()[child_1_id.block_id()];
  ElementMap<1, Frame::Grid> element_map_1{
      child_1_id, my_block.is_time_dependent()
                      ? my_block.moving_mesh_logical_to_grid_map().get_clone()
                      : my_block.stationary_map().get_to_grid_frame()};
  ElementMap<1, Frame::Grid> element_map_2{
      child_2_id, my_block.is_time_dependent()
                      ? my_block.moving_mesh_logical_to_grid_map().get_clone()
                      : my_block.stationary_map().get_to_grid_frame()};
  std::unique_ptr<GridToInertialMap> grid_to_inertial_map_1 = nullptr;
  std::unique_ptr<GridToInertialMap> grid_to_inertial_map_2 = nullptr;
  if (my_block.is_time_dependent()) {
    grid_to_inertial_map_1 =
        my_block.moving_mesh_grid_to_inertial_map().get_clone();
    grid_to_inertial_map_2 =
        my_block.moving_mesh_grid_to_inertial_map().get_clone();
  } else {
    grid_to_inertial_map_1 =
        ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            ::domain::CoordinateMaps::Identity<1>{});
    grid_to_inertial_map_2 =
        ::domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            ::domain::CoordinateMaps::Identity<1>{});
  }
  std::unordered_map<ElementId<1>, items_type> children_items;
  children_items.emplace(
      child_1_id,
      items_type{&global_cache, std::move(element_map_1),
                 std::move(grid_to_inertial_map_1), std::move(child_1)});
  children_items.emplace(
      child_2_id,
      items_type{&global_cache, std::move(element_map_2),
                 std::move(grid_to_inertial_map_2), std::move(child_2)});

  Element<1> parent{parent_id, DirectionMap<1, Neighbors<1>>{}};
  auto parent_box = db::create<
      db::AddSimpleTags<Parallel::Tags::GlobalCacheImpl<TestMetavariables>,
                        ::domain::Tags::ElementMap<1, Frame::Grid>,
                        ::domain::CoordinateMaps::Tags::CoordinateMap<
                            1, Frame::Grid, Frame::Inertial>,
                        ::domain::Tags::Element<1>>,
      tmpl::list<Parallel::Tags::FromGlobalCache<domain::Tags::Domain<1>>>>(
      &global_cache, ElementMap<1, Frame::Grid>{},
      std::unique_ptr<GridToInertialMap>{nullptr}, std::move(parent));
  db::mutate_apply<evolution::dg::Initialization::ProjectDomain<1>>(
      make_not_null(&parent_box), children_items);
  check_maps<IsTimeDependent>(
      db::get<::domain::Tags::ElementMap<1, Frame::Grid>>(parent_box),
      db::get<::domain::CoordinateMaps::Tags::CoordinateMap<1, Frame::Grid,
                                                            Frame::Inertial>>(
          parent_box));
}
}  // namespace test_projectors

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
  static_assert(
      tt::assert_conforms_to_v<evolution::dg::Initialization::ProjectDomain<1>,
                               amr::protocols::Projector>);
  test_projectors::test_p_refine<true>();
  test_projectors::test_p_refine<false>();
  test_projectors::test_split<true>();
  test_projectors::test_split<false>();
  test_projectors::test_join<true>();
  test_projectors::test_join<false>();
}
}  // namespace
