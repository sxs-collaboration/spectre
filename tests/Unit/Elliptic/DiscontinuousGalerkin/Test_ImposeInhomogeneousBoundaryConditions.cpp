// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"  // IWYU pragma: keep
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Actions/InitializeDomain.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeBoundaryConditions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/InitializeInterfaces.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "NumericalAlgorithms/LinearSolver/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Projection.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "ParallelAlgorithms/Initialization/MergeIntoDataBox.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestHelpers.hpp"

namespace {

struct ScalarField : db::SimpleTag {
  static std::string name() noexcept { return "ScalarField"; }
  using type = Scalar<DataVector>;
};

using field_tag = LinearSolver::Tags::Operand<ScalarField>;
// The variables that inhomogeneous boundary conditions contribute to
using sources_tag =
    db::add_tag_prefix<::Tags::Source,
                       Tags::Variables<tmpl::list<ScalarField>>>;

struct OtherData : db::SimpleTag {
  static std::string name() noexcept { return "OtherData"; }
  using type = Scalar<DataVector>;
};

template <size_t Dim>
class NumericalFlux {
 public:
  void compute_dirichlet_boundary(
      gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
      const Scalar<DataVector>& field,
      const tnsr::i<DataVector, Dim,
                    Frame::Inertial>& /*interface_unit_normal*/) const
      noexcept {
    numerical_flux_for_field->get() = 2. * get(field);
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct NumericalFluxTag {
  using type = NumericalFlux<Dim>;
};

template <size_t Dim>
struct AnalyticSolution {
  tuples::TaggedTuple<ScalarField> variables(
      const tnsr::I<DataVector, Dim>& x, tmpl::list<ScalarField> /*meta*/) const
      noexcept {
    return {Scalar<DataVector>(get<0>(x))};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct System {
  static constexpr const size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarField>>;
  using variables_tag = fields_tag;
  using impose_boundary_conditions_on_fields = tmpl::list<ScalarField>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

struct AddDependencies {
  template <typename DataBox, typename... InboxTags, typename Metavariables,
            size_t Dim, typename ActionList, typename ParallelComponent>
  static auto apply(DataBox& box,
                    const tuples::TaggedTuple<InboxTags...>& /*inboxes*/,
                    const Parallel::ConstGlobalCache<Metavariables>& /*cache*/,
                    const ElementIndex<Dim>& /*array_index*/,
                    const ActionList /*meta*/,
                    const ParallelComponent* const /*meta*/) noexcept {
    const auto& mesh = get<::Tags::Mesh<Dim>>(box);
    db::item_type<sources_tag> sources{mesh.number_of_grid_points(), 0.};
    return std::make_tuple(
        ::Initialization::merge_into_databox<AddDependencies,
                                             db::AddSimpleTags<sources_tag>>(
            std::move(box), std::move(sources)));
  }
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<
                         tmpl::list<::Tags::Domain<Dim, Frame::Inertial>,
                                    ::Tags::InitialExtents<Dim>>>,
                     domain::Actions::InitializeDomain<Dim>,
                     dg::Actions::InitializeInterfaces<
                         System<Dim>, dg::Initialization::slice_tags_to_face<>,
                         dg::Initialization::slice_tags_to_exterior<>>,
                     AddDependencies>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::dg::Actions::
                         ImposeInhomogeneousBoundaryConditionsOnSource>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<AnalyticSolution<Dim>>;
  using normal_dot_numerical_flux = NumericalFluxTag<Dim>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag, normal_dot_numerical_flux>;
  enum class Phase { Initialization, Testing, Exit };
};

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.ImposeInhomogeneousBoundaryConditions",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    // Reference element:
    //    [X| | | ] -xi->
    //    ^       ^
    // -0.5       1.5
    const ElementId<1> element_id{0, {{{2, 0}}}};
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::Affine>));

    using metavariables = Metavariables<1>;
    using element_array = ElementArray<1, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<1>{}, NumericalFlux<1>{}}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);

    // Expected boundary contribution to source in element X:
    // [ -24 0 0 0 | -xi->
    // -0.5 (field) * 2. (num. flux) * 6. (inverse logical mass) * 4. (jacobian
    // for mass) = -24.
    // This expectation assumes the diagonal mass matrix approximation.
    const DataVector source_expected{-24., 0., 0., 0.};
    CHECK(get_tag(Tags::Source<ScalarField>{}) ==
          Scalar<DataVector>(source_expected));
  }
  {
    INFO("2D");
    // Reference element:
    //   eta ^
    // 1.0 > +-+-+
    //       |X| |
    //       +-+-+
    //       | | |
    // 0.0 > +-+-+> xi
    //       ^   ^
    //    -0.5   1.5
    const ElementId<2> element_id{0, {{{1, 0}, {1, 1}}}};
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    // Register the coordinate map for serialization
    PUPable_reg(
        SINGLE_ARG(domain::CoordinateMap<Frame::Logical, Frame::Inertial,
                                         domain::CoordinateMaps::ProductOf2Maps<
                                             domain::CoordinateMaps::Affine,
                                             domain::CoordinateMaps::Affine>>));

    using metavariables = Metavariables<2>;
    using element_array = ElementArray<2, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<2>{}, NumericalFlux<2>{}}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);

    // Expected boundary contribution to source in element X:
    //   ^ eta
    // -18 0 12
    //  -6 0  0
    //  -6 0  0 > xi
    // This is the sum of contributions from the following faces (see 1D):
    // - upper eta: [ -12 0 12 | -xi->
    // - lower xi: | -6 -6 -6 ] -eta->
    // This expectation assumes the diagonal mass matrix approximation.
    const DataVector source_expected{-6., 0., 0., -6., 0., 0., -18., 0., 12.};
    CHECK(get_tag(Tags::Source<ScalarField>{}) ==
          Scalar<DataVector>(source_expected));
  }
  {
    INFO("3D");
    const ElementId<3> element_id{0, {{{1, 0}, {1, 1}, {1, 0}}}};
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{-0.5, 0., -1.}},
        {{1.5, 1., 3.}},
        {{false, false, false}},
        {{1, 1, 1}},
        {{2, 2, 2}}};
    // Register the coordinate map for serialization
    PUPable_reg(SINGLE_ARG(
        domain::CoordinateMap<
            Frame::Logical, Frame::Inertial,
            domain::CoordinateMaps::ProductOf3Maps<
                domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine,
                domain::CoordinateMaps::Affine>>));

    using metavariables = Metavariables<3>;
    using element_array = ElementArray<3, metavariables>;
    ActionTesting::MockRuntimeSystem<metavariables> runner{
        {AnalyticSolution<3>{}, NumericalFlux<3>{}}};
    ActionTesting::emplace_component_and_initialize<element_array>(
        &runner, element_id,
        {domain_creator.create_domain(), domain_creator.initial_extents()});
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
    runner.set_phase(metavariables::Phase::Testing);
    const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
      using tag = std::decay_t<decltype(tag_v)>;
      return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                                element_id);
    };

    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);

    // Expected boundary contribution to source in reference element (0, 1, 0):
    //                   7 eta
    //         -6 ---- 4
    // zeta ^ / |    / |
    //     -2 ---- 0   |
    //      |  -7 -|-- 5
    //      | /    | /
    //     -3 ---- 1 > xi
    // This is the sum of contributions from the following faces (see 1D):
    // - lower xi:
    //   -2 -2 > zeta
    //   -2 -2
    //    v eta
    // - upper eta:
    //   -4 4 > xi
    //   -4 4
    //    v zeta
    // - lower zeta:
    //   -1 1 > xi
    //   -1 1
    //    v eta
    // This expectation assumes the diagonal mass matrix approximation.
    const DataVector source_expected{-3., 1., -7., 5., -2., 0., -6., 4.};
    CHECK(get_tag(Tags::Source<ScalarField>{}) ==
          Scalar<DataVector>(source_expected));
  }
}
