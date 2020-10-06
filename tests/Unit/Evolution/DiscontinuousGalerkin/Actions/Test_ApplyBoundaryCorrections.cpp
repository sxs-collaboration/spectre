// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Utilities/TMPL.hpp"

namespace {
namespace TestHelpers = TestHelpers::evolution::dg::Actions;

struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, Dim, Frame::Inertial>;
};

template <size_t Dim>
struct SimpleFaceNormalMagnitude
    : db::ComputeTag,
      ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>> {
  using base = ::Tags::Magnitude<domain::Tags::UnnormalizedFaceNormal<Dim>>;
  using return_type = typename base::type;
  static void function(const gsl::not_null<return_type*> result,
                       const Mesh<Dim - 1>& face_mesh,
                       const Direction<Dim>& direction) noexcept {
    result->get() =
        DataVector{face_mesh.number_of_grid_points(),
                   1.0 + (direction.side() == Side::Upper ? 1.0 : 0.5) +
                       (direction.sign() == 1.0 ? 0.25 : 0.125)};
  }
  using argument_tags =
      tmpl::list<domain::Tags::Mesh<Dim - 1>, domain::Tags::Direction<Dim>>;
};

template <size_t Dim>
struct BoundaryTerms;

template <size_t Dim>
class BoundaryCorrection : public PUP::able {
 public:
  BoundaryCorrection() = default;
  BoundaryCorrection(const BoundaryCorrection&) = default;
  BoundaryCorrection& operator=(const BoundaryCorrection&) = default;
  BoundaryCorrection(BoundaryCorrection&&) = default;
  BoundaryCorrection& operator=(BoundaryCorrection&&) = default;

  ~BoundaryCorrection() override = default;

  WRAPPED_PUPable_abstract(BoundaryCorrection);  // NOLINT

  using creatable_classes = tmpl::list<BoundaryTerms<Dim>>;
};

template <size_t Dim>
struct BoundaryTerms final : public BoundaryCorrection<Dim> {
  struct MaxAbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  /// \cond
  explicit BoundaryTerms(CkMigrateMessage* /*unused*/) noexcept {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BoundaryTerms);  // NOLINT
  /// \endcond
  BoundaryTerms() = default;
  BoundaryTerms(const BoundaryTerms&) = default;
  BoundaryTerms& operator=(const BoundaryTerms&) = default;
  BoundaryTerms(BoundaryTerms&&) = default;
  BoundaryTerms& operator=(BoundaryTerms&&) = default;
  ~BoundaryTerms() override = default;

  using variables_tags = tmpl::list<Var1, Var2<Dim>>;
  using variables_tag = Tags::Variables<variables_tags>;

  void pup(PUP::er& p) override {  // NOLINT
    BoundaryCorrection<Dim>::pup(p);
  }

  static constexpr bool need_normal_vector = false;

  using dg_package_field_tags = tmpl::push_back<
      tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>,
                   variables_tags>,
      MaxAbsCharSpeed>;

  void dg_boundary_terms(
      const gsl::not_null<Scalar<DataVector>*> boundary_correction_var1,
      const gsl::not_null<tnsr::I<DataVector, Dim, Frame::Inertial>*>
          boundary_correction_var2,
      const Scalar<DataVector>& interior_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          interior_normal_dot_flux_var2,
      const Scalar<DataVector>& interior_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& interior_var2,
      const Scalar<DataVector>& interior_max_abs_char_speed,
      const Scalar<DataVector>& exterior_normal_dot_flux_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>&
          exterior_normal_dot_flux_var2,
      const Scalar<DataVector>& exterior_var1,
      const tnsr::I<DataVector, Dim, Frame::Inertial>& exterior_var2,
      const Scalar<DataVector>& exterior_max_abs_char_speed,
      const dg::Formulation dg_formulation) const noexcept {
    // using std::max;
    // extra minus sign on exterior normal dot flux because normal faces
    // opposite direction
    get(*boundary_correction_var1) =
        0.5 *
            ((dg_formulation == dg::Formulation::StrongInertial ? 1.0 : -1.0) *
                 get(interior_normal_dot_flux_var1) -
             get(exterior_normal_dot_flux_var1)) -
        0.5 *
            max(get(interior_max_abs_char_speed),
                get(exterior_max_abs_char_speed)) *
            (get(exterior_var1) - get(interior_var1));
    for (size_t i = 0; i < Dim; ++i) {
      boundary_correction_var2->get(i) =
          0.5 * ((dg_formulation == dg::Formulation::StrongInertial ? 1.0
                                                                    : -1.0) *
                     interior_normal_dot_flux_var2.get(i) -
                 exterior_normal_dot_flux_var2.get(i)) -
          0.5 *
              max(get(interior_max_abs_char_speed),
                  get(exterior_max_abs_char_speed)) *
              (exterior_var2.get(i) - interior_var2.get(i));
    }
  }
};

template <size_t Dim>
PUP::able::PUP_ID BoundaryTerms<Dim>::my_PUP_ID = 0;  // NOLINT

struct SetLocalMortarData {
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static std::tuple<db::DataBox<DbTagsList>&&> apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) noexcept {  // NOLINT
    constexpr size_t volume_dim = Metavariables::volume_dim;
    const auto& element = db::get<domain::Tags::Element<volume_dim>>(box);
    const auto& mortar_meshes =
        db::get<evolution::dg::Tags::MortarMesh<volume_dim>>(box);
    const auto& face_meshes = db::get<
        domain::Tags::Interface<domain::Tags::InternalDirections<volume_dim>,
                                domain::Tags::Mesh<volume_dim - 1>>>(box);
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    const auto& next_time_step_id =
        db::get<::Tags::Next<::Tags::TimeStepId>>(box);
    using mortar_tags_list =
        typename BoundaryTerms<volume_dim>::dg_package_field_tags;
    constexpr size_t number_of_dg_package_tags_components =
        Variables<mortar_tags_list>::number_of_independent_components;

    for (const auto& [direction, neighbor_ids] : element.neighbors()) {
      size_t count = 0;
      for (const auto& neighbor_id : neighbor_ids) {
        std::pair mortar_id{direction, neighbor_id};
        const Mesh<volume_dim - 1>& mortar_mesh = mortar_meshes.at(mortar_id);

        std::vector<double> type_erased_boundary_data_on_mortar(
            mortar_mesh.number_of_grid_points() *
            number_of_dg_package_tags_components);
        alg::iota(type_erased_boundary_data_on_mortar,
                  direction.dimension() +
                      10 * static_cast<unsigned long>(direction.side()) +
                      100 * count + 1000);

        db::mutate<evolution::dg::Tags::MortarData<volume_dim>>(
            make_not_null(&box),
            [&face_meshes, &mortar_id, &next_time_step_id, &time_step_id,
             &type_erased_boundary_data_on_mortar](
                const auto mortar_data_ptr) noexcept {
              mortar_data_ptr->at(mortar_id).insert_local_mortar_data(
                  Metavariables::local_time_stepping ? next_time_step_id
                                                     : time_step_id,
                  face_meshes.at(mortar_id.first),
                  std::move(type_erased_boundary_data_on_mortar));
            });
        ++count;
      }
    }
    return {std::move(box)};
  }
};

template <size_t Dim, TestHelpers::SystemType SystemType>
struct System {
  static constexpr size_t volume_dim = Dim;
  using boundary_correction = BoundaryCorrection<Dim>;

  using variables_tag = Tags::Variables<tmpl::list<Var1, Var2<Dim>>>;
  using flux_variables = tmpl::conditional_t<
      SystemType == TestHelpers::SystemType::Conservative,
      tmpl::list<Var1, Var2<Dim>>,
      tmpl::conditional_t<SystemType ==
                              TestHelpers::SystemType::Nonconservative,
                          tmpl::list<>, tmpl::list<Var2<Dim>>>>;
  using gradient_variables = tmpl::conditional_t<
      SystemType == TestHelpers::SystemType::Conservative, tmpl::list<>,
      tmpl::conditional_t<SystemType ==
                              TestHelpers::SystemType::Nonconservative,
                          tmpl::list<Var1, Var2<Dim>>, tmpl::list<Var1>>>;
};

template <typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Metavariables::volume_dim>;

  using internal_directions =
      domain::Tags::InternalDirections<Metavariables::volume_dim>;
  using boundary_directions_interior =
      domain::Tags::BoundaryDirectionsInterior<Metavariables::volume_dim>;

  using simple_tags = tmpl::list<
      ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
      db::add_tag_prefix<::Tags::dt,
                         typename Metavariables::system::variables_tag>,
      domain::Tags::Mesh<Metavariables::volume_dim>,
      domain::Tags::Element<Metavariables::volume_dim>,
      domain::Tags::Coordinates<Metavariables::volume_dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Metavariables::volume_dim, Frame::Logical,
                                    Frame::Inertial>,
      evolution::dg::Tags::Quadrature>;
  using compute_tags = tmpl::list<
      domain::Tags::JacobianCompute<Metavariables::volume_dim, Frame::Logical,
                                    Frame::Inertial>,
      domain::Tags::DetInvJacobianCompute<Metavariables::volume_dim,
                                          Frame::Logical, Frame::Inertial>,
      domain::Tags::InternalDirectionsCompute<Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          internal_directions,
          SimpleFaceNormalMagnitude<Metavariables::volume_dim>>,

      domain::Tags::BoundaryDirectionsInteriorCompute<
          Metavariables::volume_dim>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          domain::Tags::Direction<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          domain::Tags::InterfaceMesh<Metavariables::volume_dim>>,
      domain::Tags::InterfaceCompute<
          boundary_directions_interior,
          SimpleFaceNormalMagnitude<Metavariables::volume_dim>>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              ::Actions::SetupDataBox,
              ::evolution::dg::Initialization::Mortars<
                  Metavariables::volume_dim>,
              SetLocalMortarData>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<::evolution::dg::Actions::ApplyBoundaryCorrections<
              Metavariables>>>>;
};

template <size_t Dim, TestHelpers::SystemType SystemType,
          bool LocalTimeStepping>
struct Metavariables {
  static constexpr TestHelpers::SystemType system_type = SystemType;
  static constexpr size_t volume_dim = Dim;
  static constexpr bool local_time_stepping = false;
  using system = System<Dim, SystemType>;
  using const_global_cache_tags = tmpl::list<domain::Tags::InitialExtents<Dim>>;

  using component_list = tmpl::list<component<Metavariables>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <typename Tag, typename Metavariables, size_t Dim>
const auto& get_tag(
    const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
    const ElementId<Dim>& self_id) {
  return ActionTesting::get_databox_tag<component<Metavariables>, Tag>(runner,
                                                                       self_id);
}

template <size_t Dim, TestHelpers::SystemType SystemType>
void test_impl(const Spectral::Quadrature quadrature,
               const ::dg::Formulation dg_formulation) {
  CAPTURE(Dim);
  CAPTURE(SystemType);
  CAPTURE(quadrature);
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<Dim>>();
  using metavars = Metavariables<Dim, SystemType, false>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;

  // The reference element in 2d denoted by X below:
  // ^ eta
  // +-+-+> xi
  // |X| |
  // +-+-+
  // | | |
  // +-+-+
  //
  // The "self_id" for the element that we are considering is marked by an X in
  // the diagram. We consider a configuration with one neighbor in the +xi
  // direction (east_id), and (in 2d and 3d) one in the -eta (south_id)
  // direction.
  //
  // In 1d there aren't any projections to test, and in 3d we only have 1
  // element in the z-direction.
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  ElementId<Dim> self_id{};
  ElementId<Dim> east_id{};
  ElementId<Dim> south_id{};  // not used in 1d

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
  } else if constexpr (Dim == 2) {
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, {}};
  } else {
    static_assert(Dim == 3, "Only implemented tests in 1, 2, and 3d");
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] = Neighbors<Dim>{{east_id}, {}};
    neighbors[Direction<Dim>::lower_eta()] = Neighbors<Dim>{{south_id}, {}};
  }
  const Element<Dim> element{self_id, neighbors};

  MockRuntimeSystem runner{{std::vector<std::array<size_t, Dim>>{
                                make_array<Dim>(2_st), make_array<Dim>(3_st)},
                            std::make_unique<BoundaryTerms<Dim>>(),
                            dg_formulation}};

  const size_t number_of_grid_points_per_dimension = 5;
  const Mesh<Dim> mesh{number_of_grid_points_per_dimension,
                       Spectral::Basis::Legendre, quadrature};

  // Set the Jacobian to not be the identity because otherwise bugs creep in
  // easily.
  ::InverseJacobian<DataVector, Dim, Frame::Logical, Frame::Inertial> inv_jac{
      mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jac.get(i, i) = 2.0;
  }
  const auto det_inv_jacobian = determinant(inv_jac);
  const auto jacobian = determinant_and_inverse(inv_jac).second;

  // We don't need the Jacobian and map to be consistent since we are just
  // checking that given a Jacobian, coordinates, etc., the correct terms are
  // added to the evolution equations.
  const auto logical_coords = logical_coordinates(mesh);
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{};
  for (size_t i = 0; i < logical_coords.size(); ++i) {
    inertial_coords[i] = logical_coords[i];
  }

  Variables<tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2<Dim>>>>
      dt_evolved_vars{mesh.number_of_grid_points(), 0.0};

  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  // next_time_step_id will actually be used once we support local time
  // stepping. Until then we include it as a place holder to make adding local
  // time stepping easier.
  const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 100}}};
  ActionTesting::emplace_component_and_initialize<component<metavars>>(
      &runner, self_id,
      {time_step_id, next_time_step_id, dt_evolved_vars, mesh, element,
       inertial_coords, inv_jac, quadrature});

  // Run SetupDataBox action.
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);

  // Initialize both the mortars
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);
  // Set the local mortar data
  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);

  // Start testing the actual dg::ApplyBoundaryCorrections action
  ActionTesting::set_phase(make_not_null(&runner), metavars::Phase::Testing);

  // Make a copy of the mortar data so we can check against it locally
  auto all_mortar_data =
      get_tag<evolution::dg::Tags::MortarData<Dim>>(runner, self_id);

  // "Send" mortar data to element
  const auto& face_meshes =
      get_tag<domain::Tags::Interface<domain::Tags::InternalDirections<Dim>,
                                      domain::Tags::Mesh<Dim - 1>>>(runner,
                                                                    self_id);
  const auto& mortar_meshes =
      get_tag<evolution::dg::Tags::MortarMesh<Dim>>(runner, self_id);
  using mortar_tags_list = typename BoundaryTerms<Dim>::dg_package_field_tags;
  constexpr size_t number_of_dg_package_tags_components =
      Variables<mortar_tags_list>::number_of_independent_components;
  for (const auto& [direction, neighbor_ids] : neighbors) {
    size_t count = 0;
    for (const auto& neighbor_id : neighbor_ids) {
      std::pair mortar_id{direction, neighbor_id};
      const Mesh<Dim - 1>& mortar_mesh = mortar_meshes.at(mortar_id);

      std::vector<double> flux_data(mortar_mesh.number_of_grid_points() *
                                    number_of_dg_package_tags_components);
      alg::iota(flux_data,
                direction.dimension() +
                    10 * static_cast<unsigned long>(direction.side()) +
                    100 * count);
      std::tuple<Mesh<Dim - 1>, std::optional<std::vector<double>>,
                 std::optional<std::vector<double>>, ::TimeStepId>
          data{face_meshes.at(direction),
               {},
               {flux_data},
               {metavars::local_time_stepping ? next_time_step_id
                                              : time_step_id}};

      runner.template mock_distributed_objects<component<metavars>>()
          .at(self_id)
          .template receive_data<
              evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
              time_step_id, std::pair{std::pair{direction, neighbor_id}, data});
      all_mortar_data.at(mortar_id).insert_neighbor_mortar_data(
          metavars::local_time_stepping ? next_time_step_id : time_step_id,
          face_meshes.at(direction), flux_data);
      ++count;
    }
  }
  // Check expected inboxes
  REQUIRE(
      runner
          .template nonempty_inboxes<
              component<metavars>,
              evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>()
          .size() == 1);

  REQUIRE(ActionTesting::is_ready<component<metavars>>(runner, self_id));

  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);

  // Check the inboxes are empty
  REQUIRE(
      runner
          .template nonempty_inboxes<
              component<metavars>,
              evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>()
          .empty());

  // Now retrieve dt tag and check that values are correct
  using variables_tag = typename metavars::system::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using dt_variables_tags = db::wrap_tags_in<::Tags::dt, variables_tags>;
  using mortar_tags_list = typename BoundaryTerms<Dim>::dg_package_field_tags;

  const auto& mortar_sizes =
      get_tag<evolution::dg::Tags::MortarSize<Dim>>(runner, self_id);

  Variables<dt_variables_tags> dt_boundary_correction_on_mortar{};
  Variables<dt_variables_tags> dt_boundary_correction_projected_onto_face{};
  Variables<dt_variables_tags> expected_dt_variables_volume{
      mesh.number_of_grid_points(), 0.0};
  for (auto& [mortar_id, mortar_data] : all_mortar_data) {
    if (mortar_id.second == ElementId<Dim>::external_boundary_id()) {
      continue;
    }
    const auto& direction = mortar_id.first;
    const size_t dimension = direction.dimension();
    const Mesh<Dim - 1>& mortar_mesh = mortar_meshes.at(mortar_id);

    Variables<mortar_tags_list> local_data_on_mortar{
        mortar_mesh.number_of_grid_points()};
    Variables<mortar_tags_list> neighbor_data_on_mortar{
        mortar_mesh.number_of_grid_points()};
    std::copy(mortar_data.local_mortar_data()->second.begin(),
              mortar_data.local_mortar_data()->second.end(),
              local_data_on_mortar.data());
    std::copy(mortar_data.neighbor_mortar_data()->second.begin(),
              mortar_data.neighbor_mortar_data()->second.end(),
              neighbor_data_on_mortar.data());

    if (dt_boundary_correction_on_mortar.number_of_grid_points() !=
        mortar_mesh.number_of_grid_points()) {
      dt_boundary_correction_on_mortar.initialize(
          mortar_mesh.number_of_grid_points());
    }

    // Compute boundary terms on the mortar
    BoundaryTerms<Dim>{}.dg_boundary_terms(
        make_not_null(&get<Tags::dt<Var1>>(dt_boundary_correction_on_mortar)),
        make_not_null(
            &get<Tags::dt<Var2<Dim>>>(dt_boundary_correction_on_mortar)),
        get<Tags::NormalDotFlux<Var1>>(local_data_on_mortar),
        get<Tags::NormalDotFlux<Var2<Dim>>>(local_data_on_mortar),
        get<Var1>(local_data_on_mortar), get<Var2<Dim>>(local_data_on_mortar),
        get<typename BoundaryTerms<Dim>::MaxAbsCharSpeed>(local_data_on_mortar),
        get<Tags::NormalDotFlux<Var1>>(neighbor_data_on_mortar),
        get<Tags::NormalDotFlux<Var2<Dim>>>(neighbor_data_on_mortar),
        get<Var1>(neighbor_data_on_mortar),
        get<Var2<Dim>>(neighbor_data_on_mortar),
        get<typename BoundaryTerms<Dim>::MaxAbsCharSpeed>(
            neighbor_data_on_mortar),
        dg_formulation);

    // Project the boundary terms from the mortar to the face
    const std::array<Spectral::MortarSize, Dim - 1>& mortar_size =
        mortar_sizes.at(mortar_id);
    const Mesh<Dim - 1> face_mesh = mesh.slice_away(dimension);

    auto& dt_boundary_correction =
        [&dt_boundary_correction_on_mortar,
         &dt_boundary_correction_projected_onto_face, &face_mesh, &mortar_mesh,
         &mortar_size]() noexcept -> Variables<dt_variables_tags>& {
      if (::dg::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
        dt_boundary_correction_projected_onto_face =
            ::dg::project_from_mortar(dt_boundary_correction_on_mortar,
                                      face_mesh, mortar_mesh, mortar_size);
        return dt_boundary_correction_projected_onto_face;
      }
      return dt_boundary_correction_on_mortar;
    }();

    // Lift the boundary terms from the face into the volume
    const auto& magnitude_of_face_normal =
        mortar_id.second == ElementId<Dim>::external_boundary_id()
            ? get_tag<domain::Tags::Interface<
                  domain::Tags::BoundaryDirectionsInterior<Dim>,
                  ::Tags::Magnitude<
                      domain::Tags::UnnormalizedFaceNormal<Dim>>>>(runner,
                                                                   self_id)
                  .at(direction)
            : get_tag<domain::Tags::Interface<
                  domain::Tags::InternalDirections<Dim>,
                  ::Tags::Magnitude<
                      domain::Tags::UnnormalizedFaceNormal<Dim>>>>(runner,
                                                                   self_id)
                  .at(direction);
    if (mesh.quadrature(0) == Spectral::Quadrature::GaussLobatto) {
      // The lift_flux function lifts only on the slice, it does not add
      // the contribution to the volume.
      ::dg::lift_flux(make_not_null(&dt_boundary_correction),
                      mesh.extents(dimension), magnitude_of_face_normal);

      // Add the flux contribution to the volume data
      add_slice_to_data(make_not_null(&expected_dt_variables_volume),
                        dt_boundary_correction, mesh.extents(), dimension,
                        index_to_slice_at(mesh.extents(), direction));
    } else {
      // Project the volume det jacobian to the face
      Scalar<DataVector> face_det_jacobian{face_mesh.number_of_grid_points()};
      const Matrix identity{};
      auto interpolation_matrices = make_array<Dim>(std::cref(identity));
      const std::pair<Matrix, Matrix>& matrices =
          Spectral::boundary_interpolation_matrices(
              mesh.slice_through(direction.dimension()));
      gsl::at(interpolation_matrices, direction.dimension()) =
          direction.side() == Side::Upper ? matrices.second : matrices.first;
      apply_matrices(make_not_null(&get(face_det_jacobian)),
                     interpolation_matrices,
                     DataVector{1.0 / get(det_inv_jacobian)}, mesh.extents());

      // Lift from the Gauss points into the volume
      evolution::dg::lift_boundary_terms_gauss_points(
          make_not_null(&expected_dt_variables_volume), det_inv_jacobian, mesh,
          direction, dt_boundary_correction, magnitude_of_face_normal,
          face_det_jacobian);
    }
  }
  tmpl::for_each<dt_variables_tags>([&expected_dt_variables_volume, &runner,
                                     &self_id](auto tag_v) noexcept {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CHECK_ITERABLE_APPROX(get<tag>(get_tag<dt_variables_tag>(runner, self_id)),
                          get<tag>(expected_dt_variables_volume));
  });
}

template <size_t Dim>
void test() noexcept {
  for (const auto dg_formulation :
       {::dg::Formulation::StrongInertial, ::dg::Formulation::WeakInertial}) {
    for (const auto quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      test_impl<Dim, TestHelpers::SystemType::Conservative>(quadrature,
                                                            dg_formulation);
      test_impl<Dim, TestHelpers::SystemType::Nonconservative>(quadrature,
                                                               dg_formulation);
      test_impl<Dim, TestHelpers::SystemType::Mixed>(quadrature,
                                                     dg_formulation);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.ApplyBoundaryCorrections",
                  "[Unit][Evolution][Actions]") {
  test<1>();
  test<2>();
  test<3>();
}
}  // namespace
