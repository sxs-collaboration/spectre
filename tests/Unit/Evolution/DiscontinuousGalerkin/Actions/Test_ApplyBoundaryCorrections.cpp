// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <functional>
#include <memory>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Actions/ApplyBoundaryCorrections.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/Mortars.hpp"
#include "Evolution/DiscontinuousGalerkin/Initialization/QuadratureTag.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarData.hpp"
#include "Evolution/DiscontinuousGalerkin/MortarTags.hpp"
#include "Evolution/DiscontinuousGalerkin/NormalVectorTags.hpp"
#include "Framework/ActionTesting.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Actions/SystemType.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Formulation.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags/Formulation.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/RegisterDerivedClassesWithCharm.hpp"
#include "Time/BoundaryHistory.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeStepId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Time/TimeSteppers/TimeStepper.hpp"
#include "Utilities/Algorithm.hpp"
#include "Utilities/Gsl.hpp"
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
            Metavariables::volume_dim, Frame::Logical, Frame::Inertial>>(box));

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
          make_not_null(&box),
          [&covector_and_mag](const auto covector_and_mag_ptr,
                              const auto& local_direction) {
            (*covector_and_mag_ptr)[local_direction] = covector_and_mag;
          },
          direction);

      for (const auto& neighbor_id : neighbor_ids) {
        std::pair mortar_id{direction, neighbor_id};
        const Mesh<Metavariables::volume_dim - 1>& mortar_mesh =
            mortar_meshes.at(mortar_id);

        std::vector<double> type_erased_boundary_data_on_mortar(
            mortar_mesh.number_of_grid_points() *
            number_of_dg_package_tags_components);
        alg::iota(type_erased_boundary_data_on_mortar,
                  direction.dimension() +
                      10 * static_cast<unsigned long>(direction.side()) +
                      100 * count + 1000);

        db::mutate<evolution::dg::Tags::MortarData<Metavariables::volume_dim>>(
            make_not_null(&box), [&face_mesh, &mortar_id, &time_step_id,
                                  &type_erased_boundary_data_on_mortar](
                                     const auto mortar_data_ptr) noexcept {
              // when using local time stepping, the ApplyBoundaryCorrections
              // action copies the local data into the MortarDataHistory tag.
              mortar_data_ptr->at(mortar_id).insert_local_mortar_data(
                  time_step_id, face_mesh,
                  std::move(type_erased_boundary_data_on_mortar));
            });
        ++count;
        if (Metavariables::local_time_stepping) {
          const TimeStepId past_time_step_id{true, 3,
                                             Time{Slab{0.2, 3.4}, {1, 4}}};
          // When doing local time stepping we need a past history, starting an
          // 1/4 the slab.
          db::mutate<evolution::dg::Tags::MortarNextTemporalId<
              Metavariables::volume_dim>>(
              make_not_null(&box), [&mortar_id, &past_time_step_id](
                                       const auto mortar_next_temporal_id_ptr) {
                mortar_next_temporal_id_ptr->at(mortar_id) = past_time_step_id;
              });
          // We also need to set the local history one step back to get to 2nd
          // order in time.
          type_erased_boundary_data_on_mortar.resize(
              mortar_mesh.number_of_grid_points() *
              number_of_dg_package_tags_components);
          alg::iota(type_erased_boundary_data_on_mortar,
                    direction.dimension() +
                        10 * static_cast<unsigned long>(direction.side()) +
                        100 * count + 1000);
          count++;
          evolution::dg::MortarData<Metavariables::volume_dim>
              past_mortar_data{};
          past_mortar_data.insert_local_mortar_data(
              past_time_step_id, face_mesh,
              std::move(type_erased_boundary_data_on_mortar));
          Scalar<DataVector> local_face_normal_magnitude{
              face_mesh.number_of_grid_points()};
          alg::iota(get(local_face_normal_magnitude),
                    direction.dimension() +
                        10 * static_cast<unsigned long>(direction.side()) +
                        100 * count + 100000);
          if (volume_mesh.quadrature(0) == Spectral::Quadrature::Gauss) {
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
            past_mortar_data.insert_local_geometric_quantities(
                local_volume_det_inv_jacobian, local_face_det_jacobian,
                local_face_normal_magnitude);
          } else {
            past_mortar_data.insert_local_face_normal_magnitude(
                local_face_normal_magnitude);
          }
          using dt_variables_tag =
              db::add_tag_prefix<::Tags::dt,
                                 typename Metavariables::system::variables_tag>;
          db::mutate<
              evolution::dg::Tags::MortarData<Metavariables::volume_dim>,
              evolution::dg::Tags::MortarDataHistory<
                  Metavariables::volume_dim, typename dt_variables_tag::type>>(
              make_not_null(&box),
              [&det_inv_jacobian, &mortar_id, &past_mortar_data,
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
                mortar_data_history_ptr->at(mortar_id).local_insert(
                    past_time_step_id, past_mortar_data);

                // Now add the current data into the history.
                evolution::dg::MortarData<Metavariables::volume_dim>&
                    local_mortar_data = mortar_data_ptr->at(mortar_id);

                const Scalar<DataVector>& face_normal_magnitude =
                    get<evolution::dg::Tags::MagnitudeOfNormal>(
                        *normal_covector_and_magnitude.at(mortar_id.first));

                const bool using_gauss_points =
                    mesh.quadrature() == make_array<Metavariables::volume_dim>(
                                             Spectral::Quadrature::Gauss);
                if (using_gauss_points) {
                  const Scalar<DataVector> det_jacobian{
                      DataVector{1.0 / get(det_inv_jacobian)}};
                  Scalar<DataVector> face_det_jacobian{
                      mesh.slice_away(mortar_id.first.dimension())
                          .number_of_grid_points()};
                  const Matrix identity{};
                  auto interpolation_matrices =
                      make_array<Metavariables::volume_dim>(
                          std::cref(identity));
                  const std::pair<Matrix, Matrix>& matrices =
                      Spectral::boundary_interpolation_matrices(
                          mesh.slice_through(mortar_id.first.dimension()));
                  gsl::at(interpolation_matrices, mortar_id.first.dimension()) =
                      mortar_id.first.side() == Side::Upper ? matrices.second
                                                            : matrices.first;
                  apply_matrices(make_not_null(&get(face_det_jacobian)),
                                 interpolation_matrices, get(det_jacobian),
                                 mesh.extents());
                  local_mortar_data.insert_local_geometric_quantities(
                      det_inv_jacobian, face_det_jacobian,
                      face_normal_magnitude);
                } else {
                  local_mortar_data.insert_local_face_normal_magnitude(
                      face_normal_magnitude);
                }
                mortar_data_history_ptr->at(mortar_id).local_insert(
                    time_step_id, std::move(local_mortar_data));
              },
              db::get<domain::Tags::Mesh<Metavariables::volume_dim>>(box),
              db::get<evolution::dg::Tags::NormalCovectorAndMagnitude<
                  Metavariables::volume_dim>>(box));
        }
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
      ::Tags::TimeStepId, ::Tags::Next<::Tags::TimeStepId>, ::Tags::TimeStep,
      typename Metavariables::time_stepper_tag,
      db::add_tag_prefix<::Tags::dt,
                         typename Metavariables::system::variables_tag>,
      typename Metavariables::system::variables_tag,
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
                                          Frame::Logical, Frame::Inertial>>;

  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>,
              ::Actions::SetupDataBox,
              ::evolution::dg::Initialization::Mortars<
                  Metavariables::volume_dim, typename Metavariables::system>,
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
  static constexpr bool local_time_stepping = LocalTimeStepping;
  using time_stepper_tag = Tags::TimeStepper<TimeSteppers::AdamsBashforthN>;
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

template <size_t Dim, TestHelpers::SystemType SystemType,
          bool UseLocalTimeStepping>
void test_impl(const Spectral::Quadrature quadrature,
               const ::dg::Formulation dg_formulation) {
  CAPTURE(Dim);
  CAPTURE(SystemType);
  CAPTURE(quadrature);
  Parallel::register_derived_classes_with_charm<BoundaryCorrection<Dim>>();
  using metavars = Metavariables<Dim, SystemType, UseLocalTimeStepping>;
  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavars>;
  using variables_tag = typename metavars::system::variables_tag;
  using variables_tags = typename variables_tag::tags_list;
  using dt_variables_tag = db::add_tag_prefix<::Tags::dt, variables_tag>;
  using dt_variables_tags = db::wrap_tags_in<::Tags::dt, variables_tags>;
  using mortar_tags_list = typename BoundaryTerms<Dim>::dg_package_field_tags;

  // Use a second-order time stepper so that we test the local Jacobian and
  // normal magnitude history is handled correctly.
  const TimeSteppers::AdamsBashforthN time_stepper{2};

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
  std::vector<std::pair<Direction<Dim>, ElementId<Dim>>>
      order_to_send_neighbor_data_in{};

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
  if constexpr (Dim > 1) {
    order_to_send_neighbor_data_in.push_back(
        std::pair{Direction<Dim>::lower_eta(), south_id});
  }
  order_to_send_neighbor_data_in.push_back(
      std::pair{Direction<Dim>::upper_xi(), east_id});

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
  Variables<tmpl::list<Var1, Var2<Dim>>> evolved_vars{
      mesh.number_of_grid_points(), 0.0};

  const TimeDelta time_step{Slab{0.2, 3.4}, {1, 4}};
  const TimeStepId time_step_id{true, 3, Time{Slab{0.2, 3.4}, {2, 4}}};
  const TimeStepId local_next_time_step_id{true, 3,
                                           Time{Slab{0.2, 3.4}, {3, 4}}};
  const std::vector<TimeStepId> east_id_time_steps{
      {true, 3, Time{Slab{0.2, 3.4}, {2, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {3, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {4, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {5, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {6, 8}}}};
  const std::vector<TimeStepId> east_id_next_time_steps{
      {true, 3, Time{Slab{0.2, 3.4}, {3, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {4, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {5, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {6, 8}}},
      {true, 3, Time{Slab{0.2, 3.4}, {7, 8}}}};

  ActionTesting::emplace_component_and_initialize<component<metavars>>(
      &runner, self_id,
      {time_step_id, local_next_time_step_id, time_step,
       std::make_unique<TimeSteppers::AdamsBashforthN>(time_stepper),
       dt_evolved_vars, evolved_vars, mesh, element, inertial_coords, inv_jac,
       quadrature});

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
  typename evolution::dg::Tags::MortarDataHistory<
      Dim, typename dt_variables_tag::type>::type mortar_data_history{};
  if (UseLocalTimeStepping) {
    const TimeStepId past_time_step_id{true, 3, Time{Slab{0.2, 3.4}, {1, 4}}};
    // Copy local mortar data from all_mortar_data to mortar_data_history
    for (const auto& [direction, neighbors_in_direction] :
         element.neighbors()) {
      for (const auto& neighbor : neighbors_in_direction) {
        const std::pair mortar_id{direction, neighbor};
        // Copy past and current mortar data from element's DataBox
        mortar_data_history[mortar_id].local_insert(
            past_time_step_id,
            get_tag<evolution::dg::Tags::MortarDataHistory<
                Dim, typename dt_variables_tag::type>>(runner, self_id)
                .at(mortar_id)
                .local_data(past_time_step_id));
        mortar_data_history[mortar_id].local_insert(
            time_step_id,
            get_tag<evolution::dg::Tags::MortarDataHistory<
                Dim, typename dt_variables_tag::type>>(runner, self_id)
                .at(mortar_id)
                .local_data(time_step_id));
      }
    }
    // If the local history doesn't agree, the rest of the test will fail.
    const auto& element_mortar_data_hist =
        get_tag<evolution::dg::Tags::MortarDataHistory<
            Dim, typename dt_variables_tag::type>>(runner, self_id);
    REQUIRE(mortar_data_history.size() == element_mortar_data_hist.size());
  }

  // "Send" mortar data to element
  const auto& mortar_meshes =
      get_tag<evolution::dg::Tags::MortarMesh<Dim>>(runner, self_id);
  using mortar_tags_list = typename BoundaryTerms<Dim>::dg_package_field_tags;
  constexpr size_t number_of_dg_package_tags_components =
      Variables<mortar_tags_list>::number_of_independent_components;
  for (const auto& direction_and_neighbor_id : order_to_send_neighbor_data_in) {
    const auto& direction = direction_and_neighbor_id.first;
    const auto& neighbor_id = direction_and_neighbor_id.second;
    CAPTURE(direction);
    CAPTURE(neighbor_id);

    size_t count = 0;
    const Mesh<Dim - 1> face_mesh = mesh.slice_away(direction.dimension());
    const auto insert_neighbor_data = [&all_mortar_data, &count, &direction,
                                       &face_mesh, &local_next_time_step_id,
                                       &mortar_data_history, &mortar_meshes,
                                       &neighbor_id, &runner, &self_id](
                                          const TimeStepId&
                                              neighbor_time_step_id,
                                          const TimeStepId&
                                              neighbor_next_time_step_id) {
      CAPTURE(neighbor_next_time_step_id);
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
          data{face_mesh, {}, {flux_data}, {neighbor_next_time_step_id}};

      runner.template mock_distributed_objects<component<metavars>>()
          .at(self_id)
          .template receive_data<
              evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<Dim>>(
              neighbor_time_step_id,
              std::pair{std::pair{direction, neighbor_id}, data});
      if (UseLocalTimeStepping) {
        if (neighbor_time_step_id < local_next_time_step_id) {
          evolution::dg::MortarData<Dim> nhbr_mortar_data{};
          nhbr_mortar_data.insert_neighbor_mortar_data(neighbor_time_step_id,
                                                       face_mesh, flux_data);
          mortar_data_history.at(mortar_id).remote_insert(
              neighbor_time_step_id, std::move(nhbr_mortar_data));
        }
      } else {
        all_mortar_data.at(mortar_id).insert_neighbor_mortar_data(
            neighbor_time_step_id, face_mesh, flux_data);
      }
      ++count;
    };
    if (neighbor_id == east_id and UseLocalTimeStepping) {
      for (size_t east_id_time_steps_index = 0;
           east_id_time_steps_index < east_id_next_time_steps.size();
           ++east_id_time_steps_index) {
        if (east_id_time_steps_index < east_id_next_time_steps.size() - 1) {
          CHECK(not ActionTesting::is_ready<component<metavars>>(runner,
                                                                 self_id));
        } else {
          CHECK(ActionTesting::is_ready<component<metavars>>(runner, self_id));
        }
        insert_neighbor_data(east_id_time_steps[east_id_time_steps_index],
                             east_id_next_time_steps[east_id_time_steps_index]);
      }
    } else {
      // Insert the mortar data (history) running at the same speed as the
      // self_id.
      CHECK(not ActionTesting::is_ready<component<metavars>>(runner, self_id));
      if (UseLocalTimeStepping) {
        // Insert the past time, since we are using a 2nd order time stepper.
        const Time prev_time = time_step_id.step_time() - time_step;
        insert_neighbor_data(TimeStepId{time_step_id.time_runs_forward(),
                                        time_step_id.slab_number(), prev_time},
                             time_step_id);
        CHECK(
            not ActionTesting::is_ready<component<metavars>>(runner, self_id));
      }
      insert_neighbor_data(time_step_id, UseLocalTimeStepping
                                             ? local_next_time_step_id
                                             : time_step_id);
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
  // At this point we've completed testing the `is_ready` part of
  // ApplyBoundaryCorrections

  ActionTesting::next_action<component<metavars>>(make_not_null(&runner),
                                                  self_id);

  // Check the inboxes are empty when doing global time stepping
  if (not UseLocalTimeStepping) {
    REQUIRE(runner
                .template nonempty_inboxes<
                    component<metavars>,
                    evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                        Dim>>()
                .empty());
  } else {
    CHECK(runner
              .template nonempty_inboxes<
                  component<metavars>,
                  evolution::dg::Tags::BoundaryCorrectionAndGhostCellsInbox<
                      Dim>>()
              .size() == 1);
  }

  // Now retrieve dt tag and check that values are correct
  const auto& mortar_sizes =
      get_tag<evolution::dg::Tags::MortarSize<Dim>>(runner, self_id);

  Variables<dt_variables_tags> dt_boundary_correction_on_mortar{};
  Variables<dt_variables_tags> dt_boundary_correction_projected_onto_face{};
  Variables<dt_variables_tags> expected_dt_variables_volume{
      mesh.number_of_grid_points(), 0.0};
  const std::pair<Direction<Dim>, ElementId<Dim>>* mortar_id_ptr = nullptr;

  const auto compute_correction_coupling =
      [&det_inv_jacobian, &dg_formulation, &dt_boundary_correction_on_mortar,
       &dt_boundary_correction_projected_onto_face,
       &expected_dt_variables_volume, &mesh, &mortar_id_ptr, &mortar_meshes,
       &mortar_sizes, &quadrature, &runner, &self_id](
          const evolution::dg::MortarData<Dim>& local_mortar_data,
          const evolution::dg::MortarData<Dim>& neighbor_mortar_data) noexcept
      -> Variables<db::wrap_tags_in<::Tags::dt, variables_tags>> {
    const auto& mortar_id = *mortar_id_ptr;
    const auto& direction = mortar_id.first;
    const auto& mortar_mesh = mortar_meshes.at(mortar_id);
    const size_t dimension = direction.dimension();

    if (UseLocalTimeStepping and quadrature == Spectral::Quadrature::Gauss) {
      // This needs to be updated every call because the Jacobian may be
      // time-dependent. In the case of time-independent maps and local
      // time stepping we could first perform the integral on the
      // boundaries, and then lift to the volume. This is left as a future
      // optimization.
      local_mortar_data.get_local_volume_det_inv_jacobian(make_not_null(
          &const_cast<Scalar<DataVector>&>(det_inv_jacobian)));  // NOLINT
    }

    Variables<mortar_tags_list> local_data_on_mortar{
        mortar_mesh.number_of_grid_points()};
    Variables<mortar_tags_list> neighbor_data_on_mortar{
        mortar_mesh.number_of_grid_points()};
    const std::pair<Mesh<Dim - 1>, std::vector<double>>& local_mesh_and_data =
        *local_mortar_data.local_mortar_data();
    const std::pair<Mesh<Dim - 1>, std::vector<double>>&
        neighbor_mesh_and_data = *neighbor_mortar_data.neighbor_mortar_data();
    std::copy(std::get<1>(local_mesh_and_data).begin(),
              std::get<1>(local_mesh_and_data).end(),
              local_data_on_mortar.data());
    std::copy(std::get<1>(neighbor_mesh_and_data).begin(),
              std::get<1>(neighbor_mesh_and_data).end(),
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
    Scalar<DataVector> magnitude_of_face_normal{};
    if (UseLocalTimeStepping) {
      local_mortar_data.get_local_face_normal_magnitude(
          &magnitude_of_face_normal);
    } else {
      magnitude_of_face_normal = get<evolution::dg::Tags::MagnitudeOfNormal>(
          *get_tag<evolution::dg::Tags::NormalCovectorAndMagnitude<Dim>>(
               runner, self_id)
               .at(direction));
    }

    if (quadrature == Spectral::Quadrature::GaussLobatto) {
      // The lift_flux function lifts only on the slice, it does not add
      // the contribution to the volume.
      ::dg::lift_flux(make_not_null(&dt_boundary_correction),
                      mesh.extents(dimension), magnitude_of_face_normal);
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
        Scalar<DataVector> face_det_jacobian{};
        local_mortar_data.get_local_face_det_jacobian(
            make_not_null(&face_det_jacobian));

        Variables<db::wrap_tags_in<::Tags::dt, variables_tags>>
            volume_dt_correction{mesh.number_of_grid_points(), 0.0};
        evolution::dg::lift_boundary_terms_gauss_points(
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
        evolution::dg::lift_boundary_terms_gauss_points(
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
      auto& mortar_data_hist = mortar_id_and_data.second;
      mortar_id_ptr = &mortar_id;
      auto lifted_volume_data = time_stepper.compute_boundary_delta(
          compute_correction_coupling, make_not_null(&mortar_data_hist),
          time_step);
      if (quadrature == Spectral::Quadrature::GaussLobatto) {
        const auto& direction = mortar_id.first;
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
                                    &self_id](auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      CHECK_ITERABLE_APPROX(get<tag>(get_tag<variables_tag>(runner, self_id)),
                            get<tag>(expected_evolved_variables));
    });
  } else {
    for (auto& [mortar_id, mortar_data] : all_mortar_data) {
      if (mortar_id.second == ElementId<Dim>::external_boundary_id()) {
        continue;
      }
      mortar_id_ptr = &mortar_id;
      compute_correction_coupling(mortar_data, mortar_data);
    }
    tmpl::for_each<dt_variables_tags>([&expected_dt_variables_volume, &runner,
                                       &self_id](auto tag_v) noexcept {
      using tag = tmpl::type_from<decltype(tag_v)>;
      CHECK_ITERABLE_APPROX(
          get<tag>(get_tag<dt_variables_tag>(runner, self_id)),
          get<tag>(expected_dt_variables_volume));
    });
    CHECK(expected_evolved_variables ==
          get_tag<variables_tag>(runner, self_id));
  }
}

template <size_t Dim, bool UseLocalTimeStepping>
void test() {
  for (const auto dg_formulation :
       {::dg::Formulation::StrongInertial, ::dg::Formulation::WeakInertial}) {
    for (const auto quadrature :
         {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
      test_impl<Dim, TestHelpers::SystemType::Conservative,
                UseLocalTimeStepping>(quadrature, dg_formulation);
      test_impl<Dim, TestHelpers::SystemType::Nonconservative,
                UseLocalTimeStepping>(quadrature, dg_formulation);
      test_impl<Dim, TestHelpers::SystemType::Mixed, UseLocalTimeStepping>(
          quadrature, dg_formulation);
    }
  }
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.ApplyBoundaryCorrections",
                  "[Unit][Evolution][Actions]") {
  PUPable_reg(TimeSteppers::AdamsBashforthN);
  test<1, false>();
  test<1, true>();
  test<2, false>();
  test<2, true>();
  test<3, false>();
  test<3, true>();
}
}  // namespace
