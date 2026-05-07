// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <optional>
#include <random>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Creators/NonconformingSphericalShells.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Creators/Tags/Domain.hpp"
#include "Domain/Creators/Tags/InitialExtents.hpp"
#include "Domain/Creators/Tags/InitialRefinementLevels.hpp"
#include "Domain/Structure/InitialElementIds.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/BoundaryCorrection.hpp"
#include "Evolution/DgSubcell/Tags/TciStatus.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/BoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/InterpolatedBoundaryData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarInfo.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Evolution/Initialization/DgDomain.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/LiftFromBoundary.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Filter.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.hpp"
#include "NumericalAlgorithms/LinearOperators/Filters/None.tpp"
#include "NumericalAlgorithms/LinearOperators/Filters/Tag.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "Options/Protocols/FactoryCreation.hpp"
#include "Parallel/AlgorithmExecution.hpp"
#include "Parallel/Phase.hpp"
#include "ParallelAlgorithms/Actions/InitializeItems.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags/StepNumberWithinSlab.hpp"
#include "Time/Tags/Time.hpp"
#include "Time/Tags/TimeStep.hpp"
#include "Time/Tags/TimeStepId.hpp"
#include "Time/Tags/TimeStepper.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforth.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeVector.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/Rational.hpp"
#include "Utilities/Serialization/CharmPupable.hpp"
#include "Utilities/Serialization/RegisterDerivedClassesWithCharm.hpp"
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

struct VolumeTag : db::SimpleTag {
  using type = int;
};

template <size_t Dim>
struct BoundaryTerms final : public evolution::BoundaryCorrection {
  struct MaxAbsCharSpeed : db::SimpleTag {
    using type = Scalar<DataVector>;
  };

  explicit BoundaryTerms(CkMigrateMessage* /*unused*/) {}
  using PUP::able::register_constructor;
  WRAPPED_PUPable_decl_template(BoundaryTerms);  // NOLINT
  BoundaryTerms() = default;
  BoundaryTerms(const BoundaryTerms&) = default;
  BoundaryTerms& operator=(const BoundaryTerms&) = default;
  BoundaryTerms(BoundaryTerms&&) = default;
  BoundaryTerms& operator=(BoundaryTerms&&) = default;
  ~BoundaryTerms() override = default;

  using variables_tags = tmpl::list<Var1, Var2<Dim>>;
  using variables_tag = Tags::Variables<variables_tags>;

  std::unique_ptr<BoundaryCorrection> get_clone() const override {
    return std::make_unique<BoundaryTerms>(*this);
  }

  void pup(PUP::er& p) override {  // NOLINT
    BoundaryCorrection::pup(p);
  }

  static constexpr bool need_normal_vector = false;

  using dg_package_field_tags = tmpl::push_back<
      tmpl::append<db::wrap_tags_in<::Tags::NormalDotFlux, variables_tags>,
                   variables_tags>,
      MaxAbsCharSpeed>;
  using dg_boundary_terms_volume_tags = tmpl::list<VolumeTag>;

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
      const dg::Formulation dg_formulation, const int& volume_tag) const {
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
    CHECK(volume_tag == 10);
  }
};

template <size_t Dim>
PUP::able::PUP_ID BoundaryTerms<Dim>::my_PUP_ID = 0;  // NOLINT

template <bool LocalTimeStepping>
struct SetLocalMortarData {
  template <typename DbTagsList, typename... InboxTags, typename ArrayIndex,
            typename ActionList, typename ParallelComponent,
            typename Metavariables>
  static Parallel::iterable_action_return_t apply(
      db::DataBox<DbTagsList>& box,
      tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
      Parallel::GlobalCache<Metavariables>& /*cache*/,
      const ArrayIndex& /*array_index*/, ActionList /*meta*/,
      const ParallelComponent* const /*meta*/) {  // NOLINT
    const auto& element =
        db::get<domain::Tags::Element<Metavariables::volume_dim>>(box);
    const auto& volume_mesh =
        db::get<domain::Tags::Mesh<Metavariables::volume_dim>>(box);
    const auto& mortar_meshes =
        db::get<evolution::dg::Tags::MortarMesh<Metavariables::volume_dim>>(
            box);
    const auto& time_step_id = db::get<::Tags::TimeStepId>(box);
    using mortar_tags_list = typename BoundaryTerms<
        Metavariables::volume_dim>::dg_package_field_tags;
    constexpr size_t number_of_dg_package_tags_components =
        Variables<mortar_tags_list>::number_of_independent_components;

    MAKE_GENERATOR(generator);
    std::uniform_real_distribution<> dist_positive(0.5, 1.);

    using CovectorAndMag = Variables<tmpl::list<
        evolution::dg::Tags::MagnitudeOfNormal,
        evolution::dg::Tags::NormalCovector<Metavariables::volume_dim>>>;
    const Scalar<DataVector> det_inv_jacobian = determinant(
        db::get<::domain::Tags::InverseJacobian<
            Metavariables::volume_dim, Frame::ElementLogical, Frame::Inertial>>(
            box));

    for (const auto& [direction, neighbor_ids] : element.neighbors()) {
      size_t count = 0;
      const Mesh<Metavariables::volume_dim - 1> face_mesh =
          volume_mesh.slice_away(direction.dimension());
      CovectorAndMag covector_and_mag{face_mesh.number_of_grid_points()};
      get<evolution::dg::Tags::MagnitudeOfNormal>(covector_and_mag) =
          make_with_random_values<Scalar<DataVector>>(
              make_not_null(&generator), make_not_null(&dist_positive),
              face_mesh.number_of_grid_points());
      db::mutate<evolution::dg::Tags::NormalCovectorAndMagnitude<
          Metavariables::volume_dim>>(
          [&covector_and_mag](const auto covector_and_mag_ptr,
                              const auto& local_direction) {
            (*covector_and_mag_ptr)[local_direction] = covector_and_mag;
          },
          make_not_null(&box), direction);

      for (const auto& neighbor_id : neighbor_ids) {
        DirectionalId<Metavariables::volume_dim> mortar_id{direction,
                                                           neighbor_id};
        const Mesh<Metavariables::volume_dim - 1>& mortar_mesh =
            mortar_meshes.at(mortar_id);

        // Provide data of the wrong size to make sure it is projected
        // properly.
        auto unprojected_mortar_extents = mortar_mesh.extents().indices();
        if constexpr (not unprojected_mortar_extents.empty()) {
          ++unprojected_mortar_extents[0];
        }
        const Mesh<Metavariables::volume_dim - 1> unprojected_mortar_mesh(
            unprojected_mortar_extents, mortar_mesh.basis(),
            mortar_mesh.quadrature());
        DataVector type_erased_boundary_data_on_mortar{
            unprojected_mortar_mesh.number_of_grid_points() *
                number_of_dg_package_tags_components,
            0.0};
        alg::iota(type_erased_boundary_data_on_mortar,
                  direction.dimension() +
                      10 * static_cast<unsigned long>(direction.side()) +
                      100 * count + 1000);

        db::mutate<evolution::dg::Tags::MortarData<Metavariables::volume_dim>>(
            [&face_mesh, &mortar_id, &unprojected_mortar_mesh,
             &type_erased_boundary_data_on_mortar](const auto mortar_data_ptr) {
              // when using local time stepping, we reset the local mortar data
              // at the end of the SetLocalMortarData action since the
              // ComputeTimeDerivative action would've moved the data into the
              // boundary history.
              mortar_data_ptr->at(mortar_id).local().face_mesh = face_mesh;
              mortar_data_ptr->at(mortar_id).local().mortar_mesh =
                  unprojected_mortar_mesh;
              mortar_data_ptr->at(mortar_id).local().mortar_data =
                  std::move(type_erased_boundary_data_on_mortar);
            },
            make_not_null(&box));
        ++count;

        const TimeStepId past_time_step_id{true, 3,
                                           Time{Slab{0.2, 3.4}, {1, 4}}};
        // In LTS, pass an incorrect slab end for the east element to
        // simulate the slab size changing.  This previously caused
        // a bug when a slab-size change happened at a time only
        // needed on the remote side.
        const auto remote_past_time_step_id =
            LocalTimeStepping
                ? direction == Direction<Metavariables::volume_dim>::upper_xi()
                      ? TimeStepId{true, 3, Time{Slab{0.2, 1.3}, {0, 4}}}
                      : past_time_step_id
                : time_step_id;
        db::mutate<evolution::dg::Tags::MortarNextTemporalId<
            Metavariables::volume_dim>>(
            [&mortar_id, &remote_past_time_step_id](
                const auto mortar_next_temporal_id_ptr) {
              mortar_next_temporal_id_ptr->at(mortar_id) =
                  remote_past_time_step_id;
            },
            make_not_null(&box));
        if (LocalTimeStepping) {
          // We also need to set the local history one step back to get to 2nd
          // order in time.
          type_erased_boundary_data_on_mortar.destructive_resize(
              mortar_mesh.number_of_grid_points() *
              number_of_dg_package_tags_components);
          alg::iota(type_erased_boundary_data_on_mortar,
                    direction.dimension() +
                        10 * static_cast<unsigned long>(direction.side()) +
                        100 * count + 1000);
          count++;
          evolution::dg::MortarData<Metavariables::volume_dim>
              past_mortar_data{};
          past_mortar_data.face_mesh = face_mesh;
          past_mortar_data.mortar_mesh = mortar_mesh;
          past_mortar_data.mortar_data =
              std::move(type_erased_boundary_data_on_mortar);
          Scalar<DataVector> local_face_normal_magnitude{
              face_mesh.number_of_grid_points()};
          alg::iota(get(local_face_normal_magnitude),
                    direction.dimension() +
                        10 * static_cast<unsigned long>(direction.side()) +
                        100 * count + 100000);
          past_mortar_data.face_normal_magnitude = local_face_normal_magnitude;
          if (volume_mesh.quadrature(direction.dimension()) ==
              Spectral::Quadrature::Gauss) {
            Scalar<DataVector> local_face_det_jacobian{
                face_mesh.number_of_grid_points()};
            alg::iota(get(local_face_det_jacobian),
                      direction.dimension() +
                          10 * static_cast<unsigned long>(direction.side()) +
                          100 * count + 200000);
            Scalar<DataVector> local_volume_det_inv_jacobian{
                volume_mesh.number_of_grid_points()};
            alg::iota(get(local_volume_det_inv_jacobian),
                      direction.dimension() +
                          10 * static_cast<unsigned long>(direction.side()) +
                          100 * count + 300000);
            past_mortar_data.volume_det_inv_jacobian =
                local_volume_det_inv_jacobian;
            past_mortar_data.volume_mesh = volume_mesh;
            past_mortar_data.face_det_jacobian = local_face_det_jacobian;
          }
          db::mutate<evolution::dg::Tags::MortarData<Metavariables::volume_dim>,
                     evolution::dg::Tags::MortarDataHistory<
                         Metavariables::volume_dim>>(
              [&det_inv_jacobian, &mortar_id, &volume_mesh, &past_mortar_data,
               &past_time_step_id, &time_step_id](
                  const auto mortar_data_ptr,
                  const auto mortar_data_history_ptr,
                  const Mesh<Metavariables::volume_dim>& mesh,
                  const DirectionMap<Metavariables::volume_dim,
                                     std::optional<Variables<tmpl::list<
                                         evolution::dg::Tags::MagnitudeOfNormal,
                                         evolution::dg::Tags::NormalCovector<
                                             Metavariables::volume_dim>>>>>&
                      normal_covector_and_magnitude) {
                mortar_data_history_ptr->at(mortar_id).local().insert(
                    past_time_step_id, 2, past_mortar_data);

                // Now add the current data into the history.
                evolution::dg::MortarData<Metavariables::volume_dim>&
                    local_mortar_data = mortar_data_ptr->at(mortar_id).local();

                const Scalar<DataVector>& face_normal_magnitude =
                    get<evolution::dg::Tags::MagnitudeOfNormal>(
                        *normal_covector_and_magnitude.at(
                            mortar_id.direction()));

                local_mortar_data.face_normal_magnitude = face_normal_magnitude;
                if (mesh.quadrature(mortar_id.direction().dimension()) ==
                    Spectral::Quadrature::Gauss) {
                  const Scalar<DataVector> det_jacobian{
                      DataVector{1.0 / get(det_inv_jacobian)}};
                  Scalar<DataVector> face_det_jacobian{
                      mesh.slice_away(mortar_id.direction().dimension())
                          .number_of_grid_points()};
                  const Matrix identity{};
                  auto interpolation_matrices =
                      make_array<Metavariables::volume_dim>(
                          std::cref(identity));
                  const std::pair<Matrix, Matrix>& matrices =
                      Spectral::boundary_interpolation_matrices(
                          mesh.slice_through(
                              mortar_id.direction().dimension()));
                  gsl::at(interpolation_matrices,
                          mortar_id.direction().dimension()) =
                      mortar_id.direction().side() == Side::Upper
                          ? matrices.second
                          : matrices.first;
                  apply_matrices(make_not_null(&get(face_det_jacobian)),
                                 interpolation_matrices, get(det_jacobian),
                                 mesh.extents());
                  local_mortar_data.volume_det_inv_jacobian = det_inv_jacobian;
                  local_mortar_data.volume_mesh = volume_mesh;
                  local_mortar_data.face_det_jacobian = face_det_jacobian;
                }
                mortar_data_history_ptr->at(mortar_id).local().insert(
                    time_step_id, 2, std::move(local_mortar_data));
                local_mortar_data = {};
              },
              make_not_null(&box),
              db::get<domain::Tags::Mesh<Metavariables::volume_dim>>(box),
              db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<
                  Metavariables::volume_dim>>(box));
        }
      }
    }
    return {Parallel::AlgorithmExecution::Continue, std::nullopt};
  }
};

template <size_t Dim, TestHelpers::SystemType SystemType>
struct System {
  static constexpr size_t volume_dim = Dim;

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
  static constexpr bool local_time_stepping =
      Metavariables::local_time_stepping;

  using internal_directions =
      domain::Tags::InternalDirections<Metavariables::volume_dim>;
  using boundary_directions_interior =
      domain::Tags::BoundaryDirectionsInterior<Metavariables::volume_dim>;

  using simple_tags = tmpl::list<
      VolumeTag, ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
      ::Tags::TimeStep, Tags::ConcreteTimeStepper<LtsTimeStepper>,
      db::add_tag_prefix<::Tags::dt,
                         typename Metavariables::system::variables_tag>,
      typename Metavariables::system::variables_tag,
      domain::Tags::Mesh<Metavariables::volume_dim>,
      domain::Tags::Element<Metavariables::volume_dim>,
      domain::Tags::Coordinates<Metavariables::volume_dim, Frame::Inertial>,
      domain::Tags::InverseJacobian<Metavariables::volume_dim,
                                    Frame::ElementLogical, Frame::Inertial>,
      evolution::dg::Tags::Quadrature,
      domain::Tags::NeighborMesh<Metavariables::volume_dim>,
      Filters::Tags::SpectralFilter<
          Metavariables::volume_dim,
          typename Metavariables::system::variables_tag::tags_list>,
      ::Tags::StepNumberWithinSlab,
      domain::Tags::Jacobian<Metavariables::volume_dim, Frame::Grid,
                             Frame::Inertial>,
      domain::Tags::InverseJacobian<Metavariables::volume_dim, Frame::Grid,
                                    Frame::Inertial>>;
  using compute_tags = tmpl::push_back<
      time_stepper_ref_tags<LtsTimeStepper>,
      domain::Tags::JacobianCompute<Metavariables::volume_dim,
                                    Frame::ElementLogical, Frame::Inertial>,
      domain::Tags::DetInvJacobianCompute<
          Metavariables::volume_dim, Frame::ElementLogical, Frame::Inertial>>;

  using lts_action = ::evolution::dg::Actions::ApplyLtsBoundaryCorrections<
      Metavariables::volume_dim, Metavariables::use_nodegroup_dg_elements>;
  using gts_action =
      ::evolution::dg::Actions::ApplyBoundaryCorrectionsToTimeDerivative<
          Metavariables::volume_dim, Metavariables::use_nodegroup_dg_elements>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              ::evolution::dg::Initialization::Mortars<
                  Metavariables::volume_dim>,
              SetLocalMortarData<local_time_stepping>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<tmpl::conditional_t<local_time_stepping,
                                         // Apply the incorrect action first to
                                         // verify it doesn't do anything.
                                         tmpl::list<gts_action, lts_action>,
                                         tmpl::list<lts_action, gts_action>>>>>;
};

template <size_t Dim, TestHelpers::SystemType SystemType,
          bool LocalTimeStepping, bool UseNodegroupDgElements>
struct Metavariables {
  static constexpr TestHelpers::SystemType system_type = SystemType;
  static constexpr size_t volume_dim = Dim;
  static constexpr bool local_time_stepping = LocalTimeStepping;
  static constexpr bool use_nodegroup_dg_elements = UseNodegroupDgElements;
  using system = System<Dim, SystemType>;
  using const_global_cache_tags =
      tmpl::list<domain::Tags::Domain<Dim>, domain::Tags::InitialExtents<Dim>>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes =
        tmpl::map<tmpl::pair<evolution::BoundaryCorrection,
                             tmpl::list<BoundaryTerms<Dim>>>>;
  };

  using component_list = tmpl::list<component<Metavariables>>;
};

template <typename Tag, typename Metavariables, size_t Dim>
const auto& get_tag(
    const ActionTesting::MockRuntimeSystem<Metavariables>& runner,
    const ElementId<Dim>& self_id) {
  return ActionTesting::get_databox_tag<component<Metavariables>, Tag>(runner,
                                                                       self_id);
}

template <size_t Dim, TestHelpers::SystemType SystemType,
          bool UseLocalTimeStepping, bool UseNodegroupDgElements>
void test_impl(const Spectral::Quadrature quadrature,
               const ::dg::Formulation dg_formulation) {
  CAPTURE(Dim);
  CAPTURE(SystemType);
  CAPTURE(quadrature);
  CAPTURE(UseLocalTimeStepping);
  using metavars = Metavariables<Dim, SystemType, UseLocalTimeStepping,
                                 UseNodegroupDgElements>;
  register_factory_classes_with_charm<metavars>();
  using comp = component<metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using variables_tag = typename metavars::system::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using dt_variables_tags = db::wrap_tags_in<::Tags::dt, variables_tags>;
  using mortar_tags_list = typename BoundaryTerms<Dim>::dg_package_field_tags;

  // Use a second-order time stepper so that we test the local
  // Jacobian and normal magnitude history is handled correctly.  Use
  // higher-order on the element doing nontrivial LTS to test that the
  // correct TimeStepId is stored in the history, as at slab
  // boundaries only the local TimeStepId is used for equal-order
  // boundaries.
  const size_t common_integration_order = 2;
  const size_t east_integration_order = 3;
  const TimeSteppers::AdamsBashforth time_stepper{std::nullopt};

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
  //
  // We choose the east_id element to be running at a refinement of 2 in time
  // relative to the self_id element.
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  ElementId<Dim> self_id{};
  ElementId<Dim> east_id{};
  ElementId<Dim> south_id{};  // not used in 1d
  std::vector<DirectionalId<Dim>> order_to_send_neighbor_data_in{};

  if constexpr (Dim == 1) {
    self_id = ElementId<Dim>{0, {{{1, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}}}};
    neighbors[Direction<Dim>::upper_xi()] =
        Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
  } else if constexpr (Dim == 2) {
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] =
        Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
    neighbors[Direction<Dim>::lower_eta()] =
        Neighbors<Dim>{{south_id}, OrientationMap<Dim>::create_aligned()};
  } else {
    static_assert(Dim == 3, "Only implemented tests in 1, 2, and 3d");
    self_id = ElementId<Dim>{0, {{{1, 0}, {0, 0}, {0, 0}}}};
    east_id = ElementId<Dim>{0, {{{1, 1}, {0, 0}, {0, 0}}}};
    south_id = ElementId<Dim>{1, {{{1, 0}, {0, 0}, {0, 0}}}};
    neighbors[Direction<Dim>::upper_xi()] =
        Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
    neighbors[Direction<Dim>::lower_eta()] =
        Neighbors<Dim>{{south_id}, OrientationMap<Dim>::create_aligned()};
  }
  if constexpr (Dim > 1) {
    order_to_send_neighbor_data_in.push_back(
        DirectionalId<Dim>{Direction<Dim>::lower_eta(), south_id});
  }
  order_to_send_neighbor_data_in.push_back(
      DirectionalId<Dim>{Direction<Dim>::upper_xi(), east_id});

  const Element<Dim> element{self_id, neighbors};

  std::vector<Block<Dim>> blocks{Dim == 1 ? 1 : 2};
  if constexpr (Dim == 1) {
    blocks[0] = Block<Dim>(nullptr, element.id().block_id(), {});
  } else {
    blocks[0] = Block<Dim>(nullptr, 0,
                           {{Direction<Dim>::lower_eta(),
                             {1, OrientationMap<Dim>::create_aligned()}}});
    blocks[1] = Block<Dim>(nullptr, 1,
                           {{Direction<Dim>::upper_eta(),
                             {0, OrientationMap<Dim>::create_aligned()}}});
  }
  Domain<Dim> domain{std::move(blocks)};
  MockRuntimeSystem runner{{std::move(domain),
                            std::vector<std::array<size_t, Dim>>{
                                make_array<Dim>(2_st), make_array<Dim>(3_st)},
                            std::make_unique<BoundaryTerms<Dim>>(),
                            dg_formulation}};

  const size_t number_of_grid_points_per_dimension = 5;
  const Mesh<Dim> mesh{number_of_grid_points_per_dimension,
                       Spectral::Basis::Legendre, quadrature};
  typename domain::Tags::NeighborMesh<Dim>::type neighbor_mesh{};
  neighbor_mesh[{Direction<Dim>::upper_xi(), east_id}] = mesh;
  if constexpr (Dim > 1) {
    neighbor_mesh[{Direction<Dim>::lower_eta(), south_id}] = mesh;
  }

  // Set the Jacobian to not be the identity because otherwise bugs creep in
  // easily.
  ::InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      inv_jac{mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jac.get(i, i) = 2.0;
  }
  auto det_inv_jacobian = determinant(inv_jac);
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
  Variables<tmpl::list<Var1, Var2<Dim>>> evolved_vars{
      mesh.number_of_grid_points(), 0.0};

  const TimeDelta time_step{Slab{0.2, 3.4}, {1, 4}};
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {2, 4}}};
  const TimeStepId local_next_time_step_id{true, 3,
                                           Time{Slab{0.2, 3.4}, {3, 4}}};
  const std::vector<TimeStepId> east_id_time_steps{
      {true, 3, Time{Slab{0.2, 3.4}, {0, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {2, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {4, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {5, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {6, 8}}}};
  const std::vector<TimeStepId> east_id_next_time_steps{
      {true, 3, Time{Slab{0.2, 3.4}, {2, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {4, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {5, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {6, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {7, 8}}}};

  register_classes_with_charm<Filters::None<Dim, variables_tags>>();

  ActionTesting::emplace_component_and_initialize<comp>(
      &runner, self_id,
      {10, time_step_id, local_next_time_step_id, time_step,
       std::make_unique<TimeSteppers::AdamsBashforth>(time_stepper),
       dt_evolved_vars, evolved_vars, mesh, element, inertial_coords, inv_jac,
       quadrature, neighbor_mesh,
       std::unique_ptr<Filters::Filter<Dim, variables_tags>>{
           std::make_unique<Filters::None<Dim, variables_tags>>(std::nullopt)},
       static_cast<uint64_t>(0),
       Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>{},
       InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial>{}});

  // Initialize both the mortars
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  // Set the local mortar data
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Start testing the actual dg::ApplyBoundaryCorrections action
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  // Make a copy of the mortar data so we can check against it locally
  auto all_mortar_data =
      get_tag<evolution::dg::Tags::MortarData<Dim>>(runner, self_id);
  typename evolution::dg::Tags::MortarDataHistory<Dim>::type
      mortar_data_history{};
  if (UseLocalTimeStepping) {
    // Copy local mortar data from all_mortar_data to mortar_data_history
    mortar_data_history =
        get_tag<evolution::dg::Tags::MortarDataHistory<Dim>>(runner, self_id);
  }

  // Check that the action for the wrong time-stepping mode runs
  // successfully without any data having been received, and therefore
  // presumably doesn't do anything.
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // "Send" mortar data to element
  const auto& mortar_meshes =
      get_tag<evolution::dg::Tags::MortarMesh<Dim>>(runner, self_id);
  using mortar_tags_list = typename BoundaryTerms<Dim>::dg_package_field_tags;
  constexpr size_t number_of_dg_package_tags_components =
      Variables<mortar_tags_list>::number_of_independent_components;
  typename evolution::dg::subcell::Tags::NeighborTciDecisions<Dim>::type
      neighbor_decision{};
  int decision = 1;
  for (const auto& direction_and_neighbor_id : order_to_send_neighbor_data_in) {
    const auto& direction = direction_and_neighbor_id.direction();
    const auto& neighbor_id = direction_and_neighbor_id.id();
    CAPTURE(direction);
    CAPTURE(neighbor_id);

    size_t count = 0;
    const Mesh<Dim - 1> face_mesh = mesh.slice_away(direction.dimension());
    const auto insert_neighbor_data = [&all_mortar_data, &count, &decision,
                                       &direction, &face_mesh,
                                       &local_next_time_step_id, &mesh,
                                       &mortar_data_history, &mortar_meshes,
                                       &neighbor_decision, &neighbor_id,
                                       &runner, &self_id](
                                          const TimeStepId&
                                              neighbor_time_step_id,
                                          const TimeStepId&
                                              neighbor_next_time_step_id,
                                          const size_t integration_order) {
      CAPTURE(neighbor_next_time_step_id);
      DirectionalId<Dim> mortar_id{direction, neighbor_id};
      const Mesh<Dim - 1>& mortar_mesh = mortar_meshes.at(mortar_id);

      DataVector flux_data{mortar_mesh.number_of_grid_points() *
                               number_of_dg_package_tags_components,
                           0.0};
      alg::iota(flux_data,
                direction.dimension() +
                    10 * static_cast<unsigned long>(direction.side()) +
                    100 * count);
      const evolution::dg::BoundaryData<Dim> data{
          mesh,         std::nullopt,     mortar_mesh,
          std::nullopt, {flux_data},      {neighbor_next_time_step_id},
          decision,     integration_order};
      neighbor_decision.insert(std::pair{mortar_id, decision});
      ++decision;

      runner.template mock_distributed_objects<comp>()
          .at(self_id)
          .template receive_data<
              evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                  Dim, UseNodegroupDgElements>>(
              neighbor_time_step_id,
              std::pair{DirectionalId<Dim>{direction, neighbor_id}, data});
      if (UseLocalTimeStepping) {
        if (neighbor_time_step_id < local_next_time_step_id) {
          evolution::dg::MortarData<Dim> nhbr_mortar_data{};
          nhbr_mortar_data.face_mesh = face_mesh;
          nhbr_mortar_data.mortar_mesh = mortar_mesh;
          nhbr_mortar_data.mortar_data = flux_data;
          mortar_data_history.at(mortar_id).remote().insert(
              neighbor_time_step_id, integration_order,
              std::move(nhbr_mortar_data));
        }
      } else {
        all_mortar_data.at(mortar_id).neighbor().face_mesh = face_mesh;
        all_mortar_data.at(mortar_id).neighbor().mortar_mesh = mortar_mesh;
        all_mortar_data.at(mortar_id).neighbor().mortar_data = flux_data;
      }
      ++count;
    };
    if (neighbor_id == east_id and UseLocalTimeStepping) {
      for (size_t east_id_time_steps_index = 0;
           east_id_time_steps_index < east_id_next_time_steps.size();
           ++east_id_time_steps_index) {
        if (east_id_time_steps_index < east_id_next_time_steps.size() - 1) {
          REQUIRE(not ActionTesting::next_action_if_ready<comp>(
              make_not_null(&runner), self_id));
        }
        insert_neighbor_data(east_id_time_steps[east_id_time_steps_index],
                             east_id_next_time_steps[east_id_time_steps_index],
                             east_integration_order);
      }
    } else {
      // Insert the mortar data (history) running at the same speed as the
      // self_id.
      REQUIRE(not ActionTesting::next_action_if_ready<comp>(
          make_not_null(&runner), self_id));
      if (UseLocalTimeStepping) {
        // Insert the past time, since we are using a 2nd order time stepper.
        const Time prev_time = time_step_id.step_time() - time_step;
        insert_neighbor_data(TimeStepId{time_step_id.time_runs_forward(),
                                        time_step_id.slab_number(), prev_time},
                             time_step_id, common_integration_order);
        REQUIRE(not ActionTesting::next_action_if_ready<comp>(
            make_not_null(&runner), self_id));
      }
      insert_neighbor_data(time_step_id, local_next_time_step_id,
                           common_integration_order);
    }
  }
  // Check expected inboxes
  REQUIRE(
      runner
          .template nonempty_inboxes<
              comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                        Dim, UseNodegroupDgElements>>()
          .size() == 1);

  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Check the inboxes are empty when doing global time stepping
  if (not UseLocalTimeStepping) {
    REQUIRE(
        runner
            .template nonempty_inboxes<
                comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                          Dim, UseNodegroupDgElements>>()
            .empty());
  } else {
    CHECK(
        runner
            .template nonempty_inboxes<
                comp, evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                          Dim, UseNodegroupDgElements>>()
            .size() == 1);
  }

  // Now retrieve dt tag and check that values are correct
  const auto& mortar_infos =
      get_tag<evolution::dg::Tags::MortarInfo<Dim>>(runner, self_id);

  Variables<dt_variables_tags> dt_boundary_correction_on_mortar{};
  Variables<dt_variables_tags> dt_boundary_correction_projected_onto_face{};
  Variables<dt_variables_tags> expected_dt_variables_volume{
      mesh.number_of_grid_points(), 0.0};
  const DirectionalId<Dim>* mortar_id_ptr = nullptr;

  const auto compute_correction_coupling =
      [&det_inv_jacobian, &dg_formulation, &dt_boundary_correction_on_mortar,
       &dt_boundary_correction_projected_onto_face,
       &expected_dt_variables_volume, &mesh, &mortar_id_ptr, &mortar_meshes,
       &mortar_infos, &runner,
       &self_id](const evolution::dg::MortarData<Dim>& local_mortar_data,
                 const evolution::dg::MortarData<Dim>& neighbor_mortar_data)
      -> Variables<db::wrap_tags_in<::Tags::dt, variables_tags>> {
    const auto& mortar_id = *mortar_id_ptr;
    const auto& direction = mortar_id.direction();
    const auto& mortar_mesh = mortar_meshes.at(mortar_id);
    const size_t dimension = direction.dimension();

    const bool using_points_on_face =
        mesh.quadrature(dimension) == Spectral::Quadrature::GaussLobatto or
        mesh.quadrature(dimension) == Spectral::Quadrature::GaussRadauUpper;
    if (UseLocalTimeStepping and not using_points_on_face) {
      // This needs to be updated every call because the Jacobian may be
      // time-dependent. In the case of time-independent maps and local
      // time stepping we could first perform the integral on the
      // boundaries, and then lift to the volume. This is left as a future
      // optimization.
      det_inv_jacobian = local_mortar_data.volume_det_inv_jacobian.value();
    }

    Variables<mortar_tags_list> local_data_on_mortar{
        mortar_mesh.number_of_grid_points()};
    Variables<mortar_tags_list> neighbor_data_on_mortar{
        mortar_mesh.number_of_grid_points()};
    const DataVector& local_data = *local_mortar_data.mortar_data;
    const DataVector& neighbor_data = *neighbor_mortar_data.mortar_data;
    std::copy(local_data.begin(), local_data.end(),
              local_data_on_mortar.data());
    std::copy(neighbor_data.begin(), neighbor_data.end(),
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
        dg_formulation, 10);

    // Project the boundary terms from the mortar to the face
    const std::array<Spectral::SegmentSize, Dim - 1>& mortar_size =
        mortar_infos.at(mortar_id).mortar_size();
    const Mesh<Dim - 1> face_mesh = mesh.slice_away(dimension);

    auto& dt_boundary_correction =
        [&dt_boundary_correction_on_mortar,
         &dt_boundary_correction_projected_onto_face, &face_mesh, &mortar_mesh,
         &mortar_size]() -> Variables<dt_variables_tags>& {
      if (Spectral::needs_projection(face_mesh, mortar_mesh, mortar_size)) {
        dt_boundary_correction_projected_onto_face =
            ::dg::project_from_mortar(dt_boundary_correction_on_mortar,
                                      face_mesh, mortar_mesh, mortar_size);
        return dt_boundary_correction_projected_onto_face;
      }
      return dt_boundary_correction_on_mortar;
    }();

    // Lift the boundary terms from the face into the volume
    Scalar<DataVector> magnitude_of_face_normal{};
    if (UseLocalTimeStepping) {
      magnitude_of_face_normal =
          local_mortar_data.face_normal_magnitude.value();
    } else {
      magnitude_of_face_normal = get<evolution::dg::Tags::MagnitudeOfNormal>(
          *get_tag<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(
               runner, self_id)
               .at(direction));
    }

    if (using_points_on_face) {
      // The lift_flux function lifts only on the slice, it does not add
      // the contribution to the volume.
      ::dg::lift_flux(make_not_null(&dt_boundary_correction),
                      mesh.extents(dimension), magnitude_of_face_normal,
                      mesh.basis(dimension));
      if (UseLocalTimeStepping) {
        return dt_boundary_correction;
      } else {
        // Add the flux contribution to the volume data
        add_slice_to_data(make_not_null(&expected_dt_variables_volume),
                          dt_boundary_correction, mesh.extents(), dimension,
                          index_to_slice_at(mesh.extents(), direction));
      }
    } else {
      if (UseLocalTimeStepping) {
        const Scalar<DataVector>& face_det_jacobian =
            local_mortar_data.face_det_jacobian.value();

        Variables<db::wrap_tags_in<::Tags::dt, variables_tags>>
            volume_dt_correction{mesh.number_of_grid_points(), 0.0};
        ::dg::lift_boundary_terms_gauss_points(
            make_not_null(&volume_dt_correction), det_inv_jacobian, mesh,
            direction, dt_boundary_correction, magnitude_of_face_normal,
            face_det_jacobian);
        return volume_dt_correction;
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
        ::dg::lift_boundary_terms_gauss_points(
            make_not_null(&expected_dt_variables_volume), det_inv_jacobian,
            mesh, direction, dt_boundary_correction, magnitude_of_face_normal,
            face_det_jacobian);
      }
    }

    ASSERT(not UseLocalTimeStepping,
           "We shouldn't be returning empty data when using local time "
           "stepping. Some logic in the lambda this assert is in is bad. Might "
           "be a missing return?");
    return {};
  };

  Variables<variables_tags> expected_evolved_variables{
      mesh.number_of_grid_points(), 0.0};
  if (UseLocalTimeStepping) {
    for (auto& mortar_id_and_data : mortar_data_history) {
      const auto& mortar_id = mortar_id_and_data.first;
      const auto& direction = mortar_id.direction();
      auto& mortar_data_hist = mortar_id_and_data.second;
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      mortar_data_hist.local().for_each(
          [&](const TimeStepId& /*id*/,
              const gsl::not_null<evolution::dg::MortarData<Dim>*> data) {
            return p_project_mortar_data(data, mortar_mesh);
          });
      mortar_data_hist.remote().for_each(
          [&](const TimeStepId& /*id*/,
              const gsl::not_null<evolution::dg::MortarData<Dim>*> data) {
            return p_project_mortar_data(data, mortar_mesh);
          });
      mortar_id_ptr = &mortar_id;
      const bool direction_uses_points_on_face =
          mesh.quadrature(direction.dimension()) ==
              Spectral::Quadrature::GaussLobatto or
          mesh.quadrature(direction.dimension()) ==
              Spectral::Quadrature::GaussRadauUpper;
      Variables<variables_tags> lifted_volume_data{
          direction_uses_points_on_face
              ? mesh.slice_away(direction.dimension()).number_of_grid_points()
              : mesh.number_of_grid_points(),
          0.0};
      time_stepper.add_boundary_delta(&lifted_volume_data, mortar_data_hist,
                                      time_step, compute_correction_coupling);
      if (direction_uses_points_on_face) {
        // Add the flux contribution to the volume data
        add_slice_to_data(make_not_null(&expected_evolved_variables),
                          lifted_volume_data, mesh.extents(),
                          direction.dimension(),
                          index_to_slice_at(mesh.extents(), direction));
      } else {
        expected_evolved_variables += lifted_volume_data;
      }
    }

    // dt_variables should be identically zero in both cases
    CHECK(expected_dt_variables_volume ==
          get_tag<dt_variables_tag>(runner, self_id));
    tmpl::for_each<variables_tags>([&expected_evolved_variables, &runner,
                                    &self_id](auto tag_v) {
      using tag = tmpl::type_from<decltype(tag_v)>;
      CHECK_ITERABLE_APPROX(get<tag>(get_tag<variables_tag>(runner, self_id)),
                            get<tag>(expected_evolved_variables));
    });
  } else {
    for (auto& [mortar_id, mortar_data] : all_mortar_data) {
      if (mortar_id.id() == ElementId<Dim>::external_boundary_id()) {
        continue;
      }
      const auto& mortar_mesh = mortar_meshes.at(mortar_id);
      p_project_mortar_data(make_not_null(&mortar_data.local()), mortar_mesh);
      p_project_mortar_data(make_not_null(&mortar_data.neighbor()),
                            mortar_mesh);
      mortar_id_ptr = &mortar_id;
      compute_correction_coupling(mortar_data.local(), mortar_data.neighbor());
    }
    Approx custom_approx = Approx::custom().epsilon(5.e-11);
    tmpl::for_each<dt_variables_tags>(
        [&custom_approx, &expected_dt_variables_volume, &runner,
         &self_id](auto tag_v) {
          using tag = tmpl::type_from<decltype(tag_v)>;
          CHECK_ITERABLE_CUSTOM_APPROX(
              get<tag>(get_tag<dt_variables_tag>(runner, self_id)),
              get<tag>(expected_dt_variables_volume), custom_approx);
        });
    CHECK(expected_evolved_variables ==
          get_tag<variables_tag>(runner, self_id));
  }

  // Check neighbor meshes
  size_t total_neighbors = 0;
  const auto& neighbor_meshes =
      get_tag<::domain::Tags::NeighborMesh<Dim>>(runner, self_id);
  for (const auto& [direction, neighbors_in_direction] : element.neighbors()) {
    for (const auto& neighbor : neighbors_in_direction) {
      const auto it =
          neighbor_meshes.find(DirectionalId<Dim>{direction, neighbor});
      REQUIRE(it != neighbor_meshes.end());
      CHECK(it->second == mesh);
      ++total_neighbors;
    }
  }
  CHECK(neighbor_meshes.size() == total_neighbors);
}

template <size_t Dim, bool UseLocalTimeStepping,
          TestHelpers::SystemType SystemType>
void test() {
  for (const auto dg_formulation :
       {::dg::Formulation::StrongInertial, ::dg::Formulation::WeakInertial}) {
    for (const auto quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      test_impl<Dim, SystemType, UseLocalTimeStepping, false>(quadrature,
                                                              dg_formulation);
    }
  }
}

template <typename Metavariables>
struct ReceiveOrderComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<1>;

  using simple_tags = tmpl::list<
      VolumeTag, ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>,
      ::Tags::TimeStep, Tags::ConcreteTimeStepper<LtsTimeStepper>,
      typename Metavariables::system::variables_tag, domain::Tags::Mesh<1>,
      domain::Tags::Element<1>, domain::Tags::NeighborMesh<1>,
      Filters::Tags::SpectralFilter<
          1, typename Metavariables::system::variables_tag::tags_list>,
      ::Tags::StepNumberWithinSlab,
      domain::Tags::Jacobian<1, Frame::Grid, Frame::Inertial>,
      domain::Tags::InverseJacobian<1, Frame::Grid, Frame::Inertial>>;
  using compute_tags = tmpl::push_back<time_stepper_ref_tags<LtsTimeStepper>>;

  using phase_dependent_action_list =
      tmpl::list<Parallel::PhaseActions<
                     Parallel::Phase::Initialization,
                     tmpl::list<ActionTesting::InitializeDataBox<simple_tags,
                                                                 compute_tags>,
                                ::evolution::dg::Initialization::Mortars<1>>>,
                 Parallel::PhaseActions<
                     Parallel::Phase::Testing,
                     tmpl::list<::evolution::dg::Actions::
                                    ApplyLtsBoundaryCorrections<1, false>>>>;
};

struct ReceiveOrderMetavariables {
  static constexpr bool local_time_stepping = true;
  using system = System<1, TestHelpers::SystemType::Conservative>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<evolution::BoundaryCorrection,
                                                 tmpl::list<BoundaryTerms<1>>>>;
  };

  using component_list =
      tmpl::list<ReceiveOrderComponent<ReceiveOrderMetavariables>>;
};

void test_receive_order() {
  using metavars = ReceiveOrderMetavariables;
  using comp = ReceiveOrderComponent<metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  register_factory_classes_with_charm<metavars>();

  MAKE_GENERATOR(gen);

  Domain<1> domain{make_vector(Block<1>(nullptr, 0, {}))};
  const Mesh<1> mesh(2, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto);

  const ElementId<1> west_id{0, {{{2, 0}}}};
  const ElementId<1> self_id{0, {{{2, 1}}}};
  const ElementId<1> east_id{0, {{{2, 2}}}};

  DirectionMap<1, Neighbors<1>> neighbors{};
  neighbors[Direction<1>::lower_xi()] =
      Neighbors<1>{{west_id}, OrientationMap<1>::create_aligned()};
  neighbors[Direction<1>::upper_xi()] =
      Neighbors<1>{{east_id}, OrientationMap<1>::create_aligned()};
  const Element<1> element{self_id, std::move(neighbors)};

  const DirectionalId west_mortar{Direction<1>::lower_xi(), west_id};
  const DirectionalId east_mortar{Direction<1>::upper_xi(), east_id};

  domain::Tags::NeighborMesh<1>::type neighbor_mesh{};
  neighbor_mesh[west_mortar] = mesh;
  neighbor_mesh[east_mortar] = mesh;

  MockRuntimeSystem runner{{std::move(domain),
                            std::make_unique<BoundaryTerms<1>>(),
                            dg::Formulation::StrongInertial}};

  const Slab slab(0.0, 1.0);
  const TimeStepId time_step_id(true, 0, slab.start());
  const TimeStepId& next_time_step_id = time_step_id;
  const auto time_step = slab.duration();

  std::vector<std::tuple<DirectionalId<1>, Rational, Rational>> messages{
      {west_mortar, {0, 4}, {1, 4}}, {west_mortar, {1, 4}, {2, 4}},
      {west_mortar, {2, 4}, {3, 4}}, {west_mortar, {3, 4}, {4, 4}},
      {east_mortar, {0, 2}, {1, 2}}, {east_mortar, {1, 2}, {2, 2}}};
  std::shuffle(messages.begin(), messages.end(), gen);

  using variables_tag = metavars::system::variables_tag;
  using variables_tags_1d = typename variables_tag::tags_list;
  variables_tag::type evolved_vars(2, 0.0);

  register_classes_with_charm<Filters::None<1, variables_tags_1d>>();

  ActionTesting::emplace_component_and_initialize<comp>(
      &runner, self_id,
      {10, time_step_id, next_time_step_id, time_step,
       std::make_unique<TimeSteppers::AdamsBashforth>(1), evolved_vars, mesh,
       element, neighbor_mesh,
       std::unique_ptr<Filters::Filter<1, variables_tags_1d>>{
           std::make_unique<Filters::None<1, variables_tags_1d>>(std::nullopt)},
       static_cast<uint64_t>(0),
       Jacobian<DataVector, 1, Frame::Grid, Frame::Inertial>{},
       InverseJacobian<DataVector, 1, Frame::Grid, Frame::Inertial>{}});

  // Initialize the mortars
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  using mortar_tags_list = BoundaryTerms<1>::dg_package_field_tags;
  constexpr size_t mortar_data_size =
      Variables<mortar_tags_list>::number_of_independent_components;

  db::mutate<Tags::Next<Tags::TimeStepId>,
             evolution::dg::Tags::MortarDataHistory<1>>(
      [&](const gsl::not_null<TimeStepId*> id,
          const gsl::not_null<DirectionalIdMap<
              1, TimeSteppers::BoundaryHistory<evolution::dg::MortarData<1>,
                                               evolution::dg::MortarData<1>,
                                               DataVector>>*>
              mortar_history) {
        *id = TimeStepId(true, 0, slab.end());

        evolution::dg::MortarData<1> local_data{};
        local_data.mortar_data.emplace(mortar_data_size, 0.0);
        local_data.face_normal_magnitude.emplace(1_st, 1.0);
        local_data.face_mesh.emplace();
        local_data.mortar_mesh.emplace();

        mortar_history->at(west_mortar)
            .local()
            .insert(time_step_id, 1, local_data);
        mortar_history->at(east_mortar)
            .local()
            .insert(time_step_id, 1, local_data);
      },
      make_not_null(
          &ActionTesting::get_databox<comp>(make_not_null(&runner), self_id)));

  while (not messages.empty()) {
    const auto [mortar_id, send_time, next_time] = messages.back();
    messages.pop_back();

    const Mesh<0> mortar_mesh{};
    const DataVector flux_data{mortar_data_size, 0.0};
    const evolution::dg::BoundaryData<1> data{
        mesh,        std::nullopt,
        mortar_mesh, std::nullopt,
        {flux_data}, TimeStepId(true, 0, Time(slab, next_time)),
        1,           1};

    using inbox =
        evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<1, false>;

    Parallel::receive_data<inbox>(
        runner.mock_distributed_objects<comp>().at(self_id),
        TimeStepId(true, 0, Time(slab, send_time)), std::pair{mortar_id, data});

    REQUIRE(ActionTesting::next_action_if_ready<comp>(
                make_not_null(&runner), self_id) == messages.empty());
  }
}

template <typename Metavariables>
struct DeterministicComponent {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<3>;
  using variables_tag = typename Metavariables::system::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using simple_tags =
      tmpl::list<::domain::Tags::InitialExtents<3>,
                 ::domain::Tags::InitialRefinementLevels<3>,
                 ::evolution::dg::Tags::Quadrature,
                 Tags::ConcreteTimeStepper<TimeStepper>, ::Tags::Time,
                 ::Tags::TimeStep, ::Tags::TimeStepId,
                 ::Tags::Next<::Tags::TimeStepId>, VolumeTag, dt_variables_tag,
                 Filters::Tags::SpectralFilter<3, variables_tags>,
                 ::Tags::StepNumberWithinSlab>;
  using compute_tags = tmpl::push_back<time_stepper_ref_tags<TimeStepper>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          Parallel::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              Initialization::Actions::InitializeItems<
                  ::evolution::dg::Initialization::Domain<Metavariables>>,
              ::evolution::dg::Initialization::Mortars<3>>>,
      Parallel::PhaseActions<
          Parallel::Phase::Testing,
          tmpl::list<::evolution::dg::Actions::
                         ApplyBoundaryCorrectionsToTimeDerivative<3, false>>>>;
};

struct DeterministicMetavariables {
  static constexpr size_t volume_dim = 3;
  static constexpr bool local_time_stepping = false;
  using system = System<3, TestHelpers::SystemType::Conservative>;
  using const_global_cache_tags = tmpl::list<>;
  struct factory_creation
      : tt::ConformsTo<Options::protocols::FactoryCreation> {
    using factory_classes = tmpl::map<tmpl::pair<evolution::BoundaryCorrection,
                                                 tmpl::list<BoundaryTerms<3>>>>;
  };

  using component_list =
      tmpl::list<DeterministicComponent<DeterministicMetavariables>>;
};

void test_deterministic_mortar_interpolation() {
  using metavars = DeterministicMetavariables;
  using component = DeterministicComponent<metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  register_factory_classes_with_charm<metavars>();
  domain::creators::register_derived_with_charm();

  const size_t spherical_harmonic_l = 4;
  const size_t cube_angular_extents = 2;
  const domain::creators::NonconformingSphericalShells creator{
      1.9, 2.4, 2.9, 0, 1, 2, spherical_harmonic_l, cube_angular_extents};
  const Domain<3> domain = creator.create_domain();
  const auto initial_extents = creator.initial_extents();
  const auto initial_refinement = creator.initial_refinement_levels();

  MockRuntimeSystem runner{{creator.create_domain(),
                            std::make_unique<BoundaryTerms<3>>(),
                            dg::Formulation::StrongInertial}};

  const Slab slab(0.0, 1.0);
  const TimeStepId time_step_id(true, 0, slab.start());
  const TimeStepId& next_time_step_id = time_step_id;
  const auto time_step = slab.duration();
  using variables_tag = metavars::system::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  dt_variables_tag::type dt_evolved_vars(
      2 * (spherical_harmonic_l + 1) * (2 * spherical_harmonic_l + 1), 0.0);

  register_classes_with_charm<Filters::None<3, variables_tags>>();

  const ElementId<3> sphere_id{6};
  ActionTesting::emplace_component_and_initialize<component>(
      &runner, sphere_id,
      {initial_extents, initial_refinement, Spectral::Quadrature::GaussLobatto,
       std::make_unique<TimeSteppers::AdamsBashforth>(1), 1.2, time_step,
       time_step_id, next_time_step_id, 10, dt_evolved_vars,
       std::unique_ptr<Filters::Filter<3, variables_tags>>{
           std::make_unique<Filters::None<3, variables_tags>>(std::nullopt)},
       static_cast<uint64_t>(0)});

  // Initialize the domain and mortars
  ActionTesting::next_action<component>(make_not_null(&runner), sphere_id);
  ActionTesting::next_action<component>(make_not_null(&runner), sphere_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);

  const Direction<3> direction_to_sphere = Direction<3>::upper_zeta();
  const DirectionalId<3> sphere_mortar_id{direction_to_sphere, sphere_id};
  const Mesh<2> spherical_face_mesh{
      {{spherical_harmonic_l + 1, 2 * spherical_harmonic_l + 1}},
      {{Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::Gauss, Spectral::Quadrature::Equiangular}}};
  const Mesh<3> cube_volume_mesh{
      {{2, cube_angular_extents, cube_angular_extents}},
      {{Spectral::Basis::Legendre, Spectral::Basis::SphericalHarmonic,
        Spectral::Basis::SphericalHarmonic}},
      {{Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss,
        Spectral::Quadrature::Equiangular}}};
  const Mesh<2> cube_face_mesh{cube_angular_extents, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};

  const size_t n_pts = spherical_face_mesh.number_of_grid_points();
  using mortar_tags_list = BoundaryTerms<3>::dg_package_field_tags;
  constexpr size_t n_components =
      Variables<mortar_tags_list>::number_of_independent_components;
  CHECK(n_components == 9);
  const size_t mortar_data_size = n_pts * n_components;
  auto& sphere_box = get_databox<component>(make_not_null(&runner), sphere_id);
  db::mutate<evolution::dg::Tags::MortarData<3>>(
      [&sphere_id, &mortar_data_size,
       &spherical_face_mesh](const auto mortar_data_ptr) {
        auto& mortar_data =
            mortar_data_ptr
                ->at(DirectionalId<3>{Direction<3>::lower_xi(), sphere_id})
                .local();
        mortar_data.mortar_data = DataVector{mortar_data_size, 0.0};
        mortar_data.mortar_mesh = spherical_face_mesh;
        mortar_data.face_mesh = spherical_face_mesh;
      },
      make_not_null(&sphere_box));

  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist_positive(0.5, 1.);
  using CovectorAndMag =
      Variables<tmpl::list<evolution::dg::Tags::MagnitudeOfNormal,
                           evolution::dg::Tags::NormalCovector<3>>>;
  CovectorAndMag covector_and_mag{spherical_face_mesh.number_of_grid_points()};
  get<evolution::dg::Tags::MagnitudeOfNormal>(covector_and_mag) =
      make_with_random_values<Scalar<DataVector>>(
          make_not_null(&generator), make_not_null(&dist_positive),
          spherical_face_mesh.number_of_grid_points());
  db::mutate<evolution::dg::Tags::NormalCovectorAndMagnitude<3>>(
      [&covector_and_mag](const auto covector_and_mag_ptr,
                          const auto& local_direction) {
        (*covector_and_mag_ptr)[local_direction] = covector_and_mag;
      },
      make_not_null(&sphere_box), Direction<3>::lower_xi());

  std::vector<size_t> contributors(n_pts, 0);
  DataVector expected_interpolated_data{mortar_data_size, 0.0};
  std::optional<evolution::dg::InterpolatedBoundaryData<3>>
      interpolated_boundary_data{std::nullopt};
  double value = 0.0;
  for (size_t b = 0; b < 6; ++b) {
    const auto element_ids =
        initial_element_ids(b, creator.initial_refinement_levels()[b]);
    for (const auto& element_id : element_ids) {
      REQUIRE_FALSE(ActionTesting::next_action_if_ready<component>(
          make_not_null(&runner), sphere_id));
      value += 1.0;
      const ::dg::MortarInterpolator<3> mortar_interpolator{
          element_id, sphere_mortar_id, domain, cube_face_mesh,
          spherical_face_mesh};
      const DataVector face_data{
          n_components * cube_face_mesh.number_of_grid_points(), value};
      interpolated_boundary_data = evolution::dg::InterpolatedBoundaryData<3>{
          {.data = mortar_interpolator.interpolate_to_neighbor(face_data),
           .target_mesh = mortar_interpolator.neighbor_mortar_mesh(),
           .offsets =
               mortar_interpolator.interpolated_neighbor_data_offsets()}};
      for (const auto offset : interpolated_boundary_data.value().offsets()) {
        ++contributors[offset];
        for (size_t c = 0; c < n_components; ++c) {
          expected_interpolated_data[offset + c * n_pts] += value;
        }
      }
      const evolution::dg::BoundaryData<3> data{
          cube_volume_mesh,
          std::nullopt,
          cube_face_mesh,
          std::nullopt,
          {face_data},
          TimeStepId(true, 0, Time(slab, {1, 2})),
          1,
          1,
          interpolated_boundary_data};
      using inbox =
          evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<3, false>;

      Parallel::receive_data<inbox>(
          runner.mock_distributed_objects<component>().at(sphere_id),
          TimeStepId(true, 0, Time(slab, {0, 2})),
          std::pair{DirectionalId<3>{Direction<3>::lower_xi(), element_id},
                    data});
    }
  }
  REQUIRE(ActionTesting::next_action_if_ready<component>(make_not_null(&runner),
                                                         sphere_id));
  for (size_t i = 0; i < n_pts; ++i) {
    for (size_t c = 0; c < n_components; ++c) {
      expected_interpolated_data[i + c * n_pts] /=
          static_cast<double>(contributors[i]);
    }
  }
  const auto& interpolated_data =
      db::get<evolution::dg::Tags::MortarData<3>>(sphere_box)
          .at(DirectionalId<3>{Direction<3>::lower_xi(), sphere_id})
          .neighbor()
          .mortar_data.value();
  CHECK_ITERABLE_APPROX(expected_interpolated_data, interpolated_data);
}

// Concrete mock filter that records boundary-filter invocations.
class MockBoundaryFilter
    : public Filters::Filter<1, tmpl::list<Var1, Var2<1>>> {
 public:
  using Base = Filters::Filter<1, tmpl::list<Var1, Var2<1>>>;
  MockBoundaryFilter() = default;
  MockBoundaryFilter(bool apply_substep, bool apply_this_step, bool need_jacs)
      : apply_substep_(apply_substep),
        apply_this_step_(apply_this_step),
        need_jacs_(need_jacs) {}
  explicit MockBoundaryFilter(CkMigrateMessage* m) : Base(m) {}
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-function"
  // NOLINTNEXTLINE
  WRAPPED_PUPable_decl_base_template(SINGLE_ARG(Base), MockBoundaryFilter);
#pragma GCC diagnostic pop

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p) override {
    Base::pup(p);
    p | apply_substep_;
    p | apply_this_step_;
    p | need_jacs_;
    p | call_count_;
    p | saw_inv_jac_;
    p | saw_jac_;
  }
  std::unique_ptr<Base> get_clone() const override {
    return std::make_unique<MockBoundaryFilter>(*this);
  }
  bool apply_volume_filter_on_substep() const override { return false; }
  bool apply_volume_filter_on_this_step(size_t /*step*/) const override {
    return false;
  }
  bool apply_boundary_filter_on_substep() const override {
    return apply_substep_;
  }
  bool apply_boundary_filter_on_this_step(size_t /*step*/) const override {
    return apply_this_step_;
  }
  bool need_jacobians() const override { return need_jacs_; }
  bool supports_mesh(const Mesh<1>& /*mesh*/) const override { return true; }
  std::string name() const override { return "MockBoundaryFilter"; }
  bool is_equal(const Base& other) const override {
    const auto* rhs = dynamic_cast<const MockBoundaryFilter*>(&other);
    return rhs != nullptr and apply_substep_ == rhs->apply_substep_ and
           apply_this_step_ == rhs->apply_this_step_ and
           need_jacs_ == rhs->need_jacs_;
  }
  const std::optional<std::vector<size_t>>& blocks_to_filter() const override {
    return blocks_to_filter_;
  }
  void set_blocks_to_filter(
      const std::vector<std::string>& /*all_block_names*/,
      const std::unordered_map<std::string, std::unordered_set<std::string>>&
      /*block_groups*/) override {}
  void apply_in_volume(
      gsl::not_null<Variables<tmpl::list<Var1, Var2<1>>>*> /*vars*/,
      const Mesh<1>& /*mesh*/,
      const std::optional<
          InverseJacobian<DataVector, 1, Frame::Grid, Frame::Inertial>>&
      /*inv_jac*/,
      const std::optional<
          Jacobian<DataVector, 1, Frame::Grid, Frame::Inertial>>&
      /*jac*/) const override {}
  void apply_on_boundary(
      gsl::not_null<Variables<tmpl::list<Var1, Var2<1>>>*> /*vars*/,
      const Mesh<0>& /*face_mesh*/,
      const std::optional<InverseJacobian<DataVector, 1, Frame::Grid,
                                          Frame::Inertial>>& inv_jac,
      const std::optional<Jacobian<DataVector, 1, Frame::Grid,
                                   Frame::Inertial>>& jac) const override {
    ++call_count_;
    saw_inv_jac_ = inv_jac.has_value();
    saw_jac_ = jac.has_value();
  }
  size_t call_count() const { return call_count_; }
  bool saw_inv_jac() const { return saw_inv_jac_; }
  bool saw_jac() const { return saw_jac_; }

 private:
  bool apply_substep_{false};
  bool apply_this_step_{false};
  bool need_jacs_{false};
  std::optional<std::vector<size_t>> blocks_to_filter_{};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable size_t call_count_{0};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable bool saw_inv_jac_{false};
  // NOLINTNEXTLINE(spectre-mutable)
  mutable bool saw_jac_{false};
};
// NOLINTNEXTLINE
PUP::able::PUP_ID MockBoundaryFilter::my_PUP_ID = 0;

// Runs a 1D GTS GaussLobatto scenario with the given filter and calls
// callback(runner, self_id) after the GTS action completes.
template <typename Callback>
void run_boundary_filter_test_1d_gts(
    std::unique_ptr<Filters::Filter<1, tmpl::list<Var1, Var2<1>>>> filter_ptr,
    Callback&& callback) {
  constexpr size_t Dim = 1;
  using TagList = tmpl::list<Var1, Var2<Dim>>;
  using metavars =
      Metavariables<Dim, TestHelpers::SystemType::Conservative, false, false>;
  using comp = component<metavars>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;

  register_factory_classes_with_charm<metavars>();

  const ElementId<Dim> self_id{0, {{{1, 0}}}};
  const ElementId<Dim> east_id{0, {{{1, 1}}}};
  DirectionMap<Dim, Neighbors<Dim>> neighbors{};
  neighbors[Direction<Dim>::upper_xi()] =
      Neighbors<Dim>{{east_id}, OrientationMap<Dim>::create_aligned()};
  const Element<Dim> element{self_id, std::move(neighbors)};

  std::vector<Block<Dim>> blocks{1};
  blocks[0] = Block<Dim>(nullptr, 0, {});
  Domain<Dim> domain{std::move(blocks)};

  MockRuntimeSystem runner{{std::move(domain),
                            std::vector<std::array<size_t, Dim>>{
                                make_array<Dim>(2_st), make_array<Dim>(3_st)},
                            std::make_unique<BoundaryTerms<Dim>>(),
                            dg::Formulation::StrongInertial}};

  const Mesh<Dim> mesh{5, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  typename domain::Tags::NeighborMesh<Dim>::type neighbor_mesh{};
  neighbor_mesh[{Direction<Dim>::upper_xi(), east_id}] = mesh;

  InverseJacobian<DataVector, Dim, Frame::ElementLogical, Frame::Inertial>
      el_inv_jac{mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    el_inv_jac.get(i, i) = 2.0;
  }
  tnsr::I<DataVector, Dim, Frame::Inertial> inertial_coords{
      mesh.number_of_grid_points(), 0.0};

  Variables<tmpl::list<::Tags::dt<Var1>, ::Tags::dt<Var2<Dim>>>> dt_vars{
      mesh.number_of_grid_points(), 0.0};
  Variables<TagList> evolved_vars{mesh.number_of_grid_points(), 0.0};

  const TimeDelta time_step{Slab{0.2, 3.4}, {1, 4}};
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {2, 4}}};
  const TimeStepId next_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {3, 4}}};

  // Identity Grid→Inertial Jacobians (Grid == ElementLogical in this test).
  Jacobian<DataVector, Dim, Frame::Grid, Frame::Inertial> grid_jac{
      mesh.number_of_grid_points(), 0.0};
  InverseJacobian<DataVector, Dim, Frame::Grid, Frame::Inertial> grid_inv_jac{
      mesh.number_of_grid_points(), 0.0};
  for (size_t i = 0; i < Dim; ++i) {
    grid_jac.get(i, i) = 1.0;
    grid_inv_jac.get(i, i) = 1.0;
  }

  ActionTesting::emplace_component_and_initialize<comp>(
      &runner, self_id,
      {10, time_step_id, next_time_step_id, time_step,
       std::make_unique<TimeSteppers::AdamsBashforth>(std::nullopt), dt_vars,
       evolved_vars, mesh, element, inertial_coords, el_inv_jac,
       Spectral::Quadrature::GaussLobatto, neighbor_mesh, std::move(filter_ptr),
       static_cast<uint64_t>(0), grid_jac, grid_inv_jac});

  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);
  ActionTesting::set_phase(make_not_null(&runner), Parallel::Phase::Testing);
  // Run the wrong-mode action first (LTS no-op for GTS).
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  // Send one neighbor's boundary data.
  const auto& mortar_meshes =
      ActionTesting::get_databox_tag<comp,
                                     evolution::dg::Tags::MortarMesh<Dim>>(
          runner, self_id);
  using mortar_tags_list_1d =
      typename BoundaryTerms<Dim>::dg_package_field_tags;
  constexpr size_t n_mortar_comps =
      Variables<mortar_tags_list_1d>::number_of_independent_components;
  const DirectionalId<Dim> mortar_id{Direction<Dim>::upper_xi(), east_id};
  const Mesh<Dim - 1>& mortar_mesh = mortar_meshes.at(mortar_id);
  DataVector flux_data{mortar_mesh.number_of_grid_points() * n_mortar_comps,
                       1.0};
  const evolution::dg::BoundaryData<Dim> data{
      mesh,        std::nullopt,      mortar_mesh, std::nullopt,
      {flux_data}, next_time_step_id, 1,           2};
  runner.template mock_distributed_objects<comp>()
      .at(self_id)
      .template receive_data<
          evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim,
                                                                    false>>(
          time_step_id, std::pair{mortar_id, data});

  // Run the GTS action.
  ActionTesting::next_action<comp>(make_not_null(&runner), self_id);

  std::forward<Callback>(callback)(runner, self_id);
}

// Returns a pointer to the MockBoundaryFilter stored in the DataBox, or
// nullptr if the filter is not a MockBoundaryFilter.
template <typename MockRuntimeSystem>
const MockBoundaryFilter* get_mock_boundary_filter(
    const MockRuntimeSystem& runner, const ElementId<1>& self_id) {
  using metavars =
      Metavariables<1, TestHelpers::SystemType::Conservative, false, false>;
  using comp = component<metavars>;
  using FilterTag = Filters::Tags::SpectralFilter<1, tmpl::list<Var1, Var2<1>>>;
  const auto& filter_ref =
      ActionTesting::get_databox_tag<comp, FilterTag>(runner, self_id);
  return dynamic_cast<const MockBoundaryFilter*>(&filter_ref);
}

void test_boundary_filter_no_cadence_skips() {
  run_boundary_filter_test_1d_gts(
      std::make_unique<MockBoundaryFilter>(false, false, false),
      [](const auto& runner, const ElementId<1>& self_id) {
        const MockBoundaryFilter* mock =
            get_mock_boundary_filter(runner, self_id);
        REQUIRE(mock != nullptr);
        CHECK(mock->call_count() == 0);
      });
}

void test_boundary_filter_substep_applies() {
  run_boundary_filter_test_1d_gts(
      std::make_unique<MockBoundaryFilter>(true, false, false),
      [](const auto& runner, const ElementId<1>& self_id) {
        const MockBoundaryFilter* mock =
            get_mock_boundary_filter(runner, self_id);
        REQUIRE(mock != nullptr);
        CHECK(mock->call_count() == 1);
      });
}

void test_boundary_filter_step_applies() {
  run_boundary_filter_test_1d_gts(
      std::make_unique<MockBoundaryFilter>(false, true, false),
      [](const auto& runner, const ElementId<1>& self_id) {
        const MockBoundaryFilter* mock =
            get_mock_boundary_filter(runner, self_id);
        REQUIRE(mock != nullptr);
        CHECK(mock->call_count() == 1);
      });
}

void test_boundary_filter_jacobians_passed_when_needed() {
  run_boundary_filter_test_1d_gts(
      std::make_unique<MockBoundaryFilter>(true, false, true),
      [](const auto& runner, const ElementId<1>& self_id) {
        const MockBoundaryFilter* mock =
            get_mock_boundary_filter(runner, self_id);
        REQUIRE(mock != nullptr);
        CHECK(mock->call_count() == 1);
        CHECK(mock->saw_inv_jac());
        CHECK(mock->saw_jac());
      });
}

void test_boundary_filter_jacobians_not_passed_when_not_needed() {
  run_boundary_filter_test_1d_gts(
      std::make_unique<MockBoundaryFilter>(true, false, false),
      [](const auto& runner, const ElementId<1>& self_id) {
        const MockBoundaryFilter* mock =
            get_mock_boundary_filter(runner, self_id);
        REQUIRE(mock != nullptr);
        CHECK(mock->call_count() == 1);
        CHECK(not mock->saw_inv_jac());
        CHECK(not mock->saw_jac());
      });
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.ApplyBoundaryCorrections",
                  "[Unit][Evolution][Actions]") {
  PUPable_reg(TimeSteppers::AdamsBashforth);
  tmpl::for_each<tmpl::integral_list<size_t, 1, 2, 3>>([](auto dim_v) {
    tmpl::for_each<tmpl::integral_list<bool, false, true>>(
        [&dim_v](auto lts_v) {
          tmpl::for_each<tmpl::integral_list<
              TestHelpers::SystemType, TestHelpers::SystemType::Conservative,
              TestHelpers::SystemType::Nonconservative,
              TestHelpers::SystemType::Mixed>>([&dim_v, &lts_v](auto system_v) {
            (void)dim_v, (void)lts_v;
            test<tmpl::type_from<decltype(dim_v)>::value,
                 tmpl::type_from<decltype(lts_v)>::value,
                 tmpl::type_from<decltype(system_v)>::value>();
          });
        });
  });

  test_receive_order();

  test_deterministic_mortar_interpolation();
  register_classes_with_charm<MockBoundaryFilter>();
  test_boundary_filter_no_cadence_skips();
  test_boundary_filter_substep_applies();
  test_boundary_filter_step_applies();
  test_boundary_filter_jacobians_passed_when_needed();
  test_boundary_filter_jacobians_not_passed_when_not_needed();
}
}  // namespace
