// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Block.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/Domain.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/SimpleMortarData.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/CollectDataForFluxes.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeMortars.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
constexpr size_t volume_dim = 2;
constexpr size_t face_dim = volume_dim - 1;

using TemporalId = int;

struct TemporalIdTag : db::SimpleTag {
  using type = TemporalId;
};

struct BoundaryData {
  bool is_projected;
  TemporalId temporal_id;
  size_t num_points;
  BoundaryData project_to_mortar(
      const Mesh<face_dim>& /*face_mesh*/,
      const Mesh<face_dim>& /*mortar_mesh*/,
      const std::array<Spectral::MortarSize, face_dim>& /*mortar_size*/) const
      noexcept {
    return {true, temporal_id, num_points};
  }
};

using MortarData = dg::SimpleMortarData<TemporalId, BoundaryData, BoundaryData>;

struct MortarDataTag : db::SimpleTag {
  using type = MortarData;
};

struct DgBoundaryScheme {
  static constexpr size_t volume_dim = ::volume_dim;
  using temporal_id_tag = TemporalIdTag;
  using receive_temporal_id_tag = temporal_id_tag;
  using mortar_data_tag = MortarDataTag;
  struct boundary_data_computer {
    using argument_tags =
        tmpl::list<TemporalIdTag, domain::Tags::Mesh<face_dim>>;
    using volume_tags = tmpl::list<TemporalIdTag>;
    static BoundaryData apply(const TemporalId& temporal_id,
                              const Mesh<face_dim>& face_mesh) noexcept {
      return {false, temporal_id, face_mesh.number_of_grid_points()};
    }
  };
};

template <typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<volume_dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<volume_dim>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<db::AddSimpleTags<
                  domain::Tags::InitialRefinementLevels<volume_dim>,
                  domain::Tags::InitialExtents<volume_dim>, TemporalIdTag,
                  ::Tags::Next<TemporalIdTag>>>,
              dg::Actions::InitializeDomain<volume_dim>,
              Initialization::Actions::AddComputeTags<tmpl::list<
                  domain::Tags::InternalDirections<volume_dim>,
                  domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                  domain::Tags::BoundaryDirectionsExterior<volume_dim>,
                  domain::Tags::InterfaceCompute<
                      domain::Tags::InternalDirections<volume_dim>,
                      domain::Tags::Direction<volume_dim>>,
                  domain::Tags::InterfaceCompute<
                      domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                      domain::Tags::Direction<volume_dim>>,
                  domain::Tags::InterfaceCompute<
                      domain::Tags::BoundaryDirectionsExterior<volume_dim>,
                      domain::Tags::Direction<volume_dim>>,
                  domain::Tags::InterfaceCompute<
                      domain::Tags::InternalDirections<volume_dim>,
                      domain::Tags::InterfaceMesh<volume_dim>>,
                  domain::Tags::InterfaceCompute<
                      domain::Tags::BoundaryDirectionsInterior<volume_dim>,
                      domain::Tags::InterfaceMesh<volume_dim>>,
                  domain::Tags::InterfaceCompute<
                      domain::Tags::BoundaryDirectionsExterior<volume_dim>,
                      domain::Tags::InterfaceMesh<volume_dim>>>>,
              dg::Actions::InitializeMortars<DgBoundaryScheme>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<
              dg::Actions::CollectDataForFluxes<
                  DgBoundaryScheme,
                  domain::Tags::InternalDirections<volume_dim>>,
              dg::Actions::CollectDataForFluxes<
                  DgBoundaryScheme,
                  domain::Tags::BoundaryDirectionsInterior<volume_dim>>>>>;
};

struct Metavariables {
  using component_list = tmpl::list<ElementArray<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Actions.CollectDataForFluxes",
                  "[Unit][NumericalAlgorithms][Actions]") {
  using element_array = ElementArray<Metavariables>;

  // Reference element:
  // ^ eta
  // +-+-+> xi
  // |X| |
  // +-+-+
  // | | |
  // +-+-+
  const ElementId<volume_dim> self_id{0, {{{1, 0}, {0, 0}}}};
  const ElementId<volume_dim> east_id{0, {{{1, 1}, {0, 0}}}};
  const auto mortar_id_east =
      std::make_pair(Direction<volume_dim>::upper_xi(), east_id);
  const auto mortar_id_west =
      std::make_pair(Direction<volume_dim>::lower_xi(),
                     ElementId<volume_dim>::external_boundary_id());
  const ElementId<volume_dim> south_id{1, {{{1, 0}, {0, 0}}}};
  const auto mortar_id_south =
      std::make_pair(Direction<volume_dim>::lower_eta(), south_id);
  const auto mortar_id_north =
      std::make_pair(Direction<volume_dim>::upper_eta(),
                     ElementId<volume_dim>::external_boundary_id());

  // Setup domain with refinement
  using IdentityMap =
      domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                            domain::CoordinateMaps::Identity<volume_dim>>;
  PUPable_reg(SINGLE_ARG(IdentityMap));
  const IdentityMap identity_map{
      domain::CoordinateMaps::Identity<volume_dim>{}};
  std::vector<Block<volume_dim>> blocks;
  blocks.emplace_back(
      Block<volume_dim>{identity_map.get_clone(),
                        0,
                        {{Direction<volume_dim>::lower_eta(), {1, {}}}}});
  blocks.emplace_back(
      Block<volume_dim>{identity_map.get_clone(),
                        1,
                        {{Direction<volume_dim>::upper_eta(), {0, {}}}}});
  Domain<volume_dim> domain{std::move(blocks)};
  // p-refine the south block
  std::vector<std::array<size_t, volume_dim>> initial_extents{
      {make_array<volume_dim>(size_t{2})}, {make_array<volume_dim>(size_t{3})}};
  std::vector<std::array<size_t, volume_dim>> initial_refinement_levels{
      {{1, 0}}, {{1, 0}}};

  const TemporalId time{1};

  ActionTesting::MockRuntimeSystem<Metavariables> runner{{std::move(domain)}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, self_id,
      {std::move(initial_refinement_levels), std::move(initial_extents), time,
       time + 1});
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  runner.set_phase(Metavariables::Phase::Testing);
  const auto get_tag = [&runner, &self_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner, self_id);
  };

  const auto check_mortar =
      [&get_tag, &time](
          const std::pair<Direction<volume_dim>, ElementId<volume_dim>>&
              mortar_id,
          const size_t num_points, const bool expect_projection) noexcept {
        CAPTURE(mortar_id);
        const auto& all_mortar_data =
            get_tag(::Tags::Mortars<MortarDataTag, volume_dim>{});
        const auto& boundary_data =
            all_mortar_data.at(mortar_id).local_data(time);
        CHECK(boundary_data.temporal_id == time);
        CHECK(boundary_data.num_points == num_points);
        CHECK(boundary_data.is_projected == expect_projection);
      };

  // Collect on internal directions
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  check_mortar(mortar_id_east, 2, false);
  check_mortar(mortar_id_south, 2, true);

  // Collect on external directions
  ActionTesting::next_action<element_array>(make_not_null(&runner), self_id);
  check_mortar(mortar_id_west, 2, false);
  check_mortar(mortar_id_north, 2, false);
}
