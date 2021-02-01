// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/ImposeInhomogeneousBoundaryConditionsOnSource.hpp"
#include "Elliptic/Tags.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Parallel/Actions/SetupDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace PUP {
class er;
}  // namespace PUP
// IWYU pragma: no_forward_declare Tensor

namespace {
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Fluxes {
  using argument_tags = tmpl::list<>;
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
  using primal_fields = tmpl::list<ScalarFieldTag>;
  using fluxes_computer = Fluxes<Dim>;
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim>
struct AnalyticSolution {
  tuples::TaggedTuple<ScalarFieldTag> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<ScalarFieldTag> /*meta*/) const noexcept {
    Scalar<DataVector> solution{get<0>(x)};
    for (size_t d = 1; d < Dim; d++) {
      get(solution) += x.get(d);
    }
    return {std::move(solution)};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct NumericalFlux {
  void compute_dirichlet_boundary(
      const gsl::not_null<Scalar<DataVector>*> numerical_flux_for_field,
      const Scalar<DataVector>& field, const Mesh<Dim>& /*volume_mesh*/,
      const Direction<Dim>& /*direction*/,
      const tnsr::i<DataVector, Dim,
                    Frame::Inertial>& /*interface_unit_normal*/,
      const Scalar<DataVector>& /*face_normal_magnitude*/,
      const Fluxes<Dim>& /*flux_computer*/) const noexcept {
    numerical_flux_for_field->get() = 2. * get(field);
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
                         domain::Tags::Domain<Dim>,
                         domain::Tags::InitialRefinementLevels<Dim>,
                         domain::Tags::InitialExtents<Dim>,
                         db::add_tag_prefix<
                             ::Tags::FixedSource,
                             typename Metavariables::system::fields_tag>>>,
                     Actions::SetupDataBox, dg::Actions::InitializeDomain<Dim>,
                     dg::Actions::InitializeInterfaces<
                         typename Metavariables::system,
                         dg::Initialization::slice_tags_to_face<>,
                         dg::Initialization::slice_tags_to_exterior<>,
                         dg::Initialization::face_compute_tags<>,
                         dg::Initialization::exterior_compute_tags<>, false>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::dg::Actions::
                         ImposeInhomogeneousBoundaryConditionsOnSource<
                             Metavariables>>>>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using analytic_solution_tag = Tags::AnalyticSolution<AnalyticSolution<Dim>>;
  using normal_dot_numerical_flux = Tags::NumericalFlux<NumericalFlux<Dim>>;
  using component_list = tmpl::list<ElementArray<Dim, Metavariables>>;
  using const_global_cache_tags =
      tmpl::list<elliptic::Tags::FluxesComputer<Fluxes<Dim>>,
                 analytic_solution_tag, normal_dot_numerical_flux>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim, typename DomainCreator>
void test_impose_inhomogeneous_boundary_conditions_on_source(
    const DomainCreator& domain_creator, const ElementId<Dim>& element_id,
    const DataVector& source_expected) {
  using metavariables = Metavariables<Dim>;
  using system = typename metavariables::system;
  using element_array = ElementArray<Dim, metavariables>;

  typename db::add_tag_prefix<Tags::FixedSource,
                              typename system::fields_tag>::type source_vars{
      source_expected.size(), 0.};

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {Fluxes<Dim>{}, AnalyticSolution<Dim>{}, NumericalFlux<Dim>{}}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id,
      {domain_creator.create_domain(),
       domain_creator.initial_refinement_levels(),
       domain_creator.initial_extents(), std::move(source_vars)});
  for (size_t i = 0; i < 3; ++i) {
    ActionTesting::next_action<element_array>(make_not_null(&runner),
                                              element_id);
  }
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  const auto get_tag = [&runner, &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::get_databox_tag<element_array, tag>(runner,
                                                              element_id);
  };

  CHECK(get_tag(Tags::FixedSource<ScalarFieldTag>{}) ==
        Scalar<DataVector>(source_expected));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Actions.InhomogeneousBoundaryConditions",
                  "[Unit][Elliptic][Actions]") {
  domain::creators::register_derived_with_charm();
  {
    INFO("1D");
    // Reference element:
    //    [X| | | ] -> xi
    //    ^       ^
    // -0.5       1.5
    const ElementId<1> element_id{0, {{SegmentId{2, 0}}}};
    const domain::creators::Interval domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};

    // Expected boundary contribution to source in element X:
    // [ -24 0 0 0 | -> xi
    // -0.5 (field) * 2. (num. flux) * 6. (inverse logical mass) * 4.
    // (jacobian for mass) = -24. This expectation assumes the diagonal mass
    // matrix approximation.
    const DataVector source_expected{-24., 0., 0., 0.};
    test_impose_inhomogeneous_boundary_conditions_on_source(
        domain_creator, element_id, source_expected);
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
    const ElementId<2> element_id{0, {{SegmentId{1, 0}, SegmentId{1, 1}}}};
    const domain::creators::Rectangle domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 3}}};

    // Expected boundary contribution to source in element X:
    //  ^ eta
    // 18 24 36
    //  3  0  0
    //  0  0  0 > xi
    // This is the sum of contributions from the following faces (see 1D):
    // - upper eta: [ 12 24 36 | -> xi
    // - lower xi: | 0 3 6 ] -> eta
    // This expectation assumes the diagonal mass matrix approximation.
    const DataVector source_expected{0., 0., 0., 3., 0., 0., 18., 24., 36.};
    test_impose_inhomogeneous_boundary_conditions_on_source(
        domain_creator, element_id, source_expected);
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}};
    const domain::creators::Brick domain_creator{{{-0.5, 0., -1.}},
                                                 {{1.5, 1., 3.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{2, 2, 2}}};

    // Expected boundary contribution to source in reference element (0, 1, 0):
    //                   7 eta
    //         18 ---- 20
    // zeta ^ / |    / |
    //      4 ---- 0   |
    //      |  -7 -|-- 5
    //      | /    | /
    //     -6 ---- 0 > xi
    // This is the sum of contributions from the following faces (see 1D):
    // - lower xi:
    //   -4 4 > zeta
    //   -2 6
    //    v eta
    // - upper eta:
    //   -4  4 > xi
    //   12 20
    //    v zeta
    // - lower zeta:
    //   -2 0 > xi
    //   -1 1
    //    v eta
    // This expectation assumes the diagonal mass matrix approximation.
    const DataVector source_expected{-6., 0., -7., 5., 4., 0., 18., 20.};
    test_impose_inhomogeneous_boundary_conditions_on_source(
        domain_creator, element_id, source_expected);
  }
}
