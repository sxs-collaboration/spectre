// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Domain.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/Initialization/BoundaryConditions.hpp"
#include "Elliptic/Initialization/Domain.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

namespace PUP {
class er;
}  // namespace PUP
// IWYU pragma: no_forward_declare Tensor

namespace {
struct ScalarFieldTag : db::SimpleTag {
  static std::string name() noexcept { return "ScalarFieldTag"; };
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  using fields_tag = Tags::Variables<tmpl::list<ScalarFieldTag>>;
  using impose_boundary_conditions_on_fields = tmpl::list<ScalarFieldTag>;
};

template <size_t Dim>
struct AnalyticSolution {
  tuples::TaggedTuple<ScalarFieldTag> variables(
      const tnsr::I<DataVector, Dim>& x,
      tmpl::list<ScalarFieldTag> /*meta*/) const noexcept {
    return {Scalar<DataVector>(get<0>(x))};
  }
  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
struct NumericalFlux {
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
struct Metavariables {
  using system = System<Dim>;
  using analytic_solution_tag =
      OptionTags::AnalyticSolution<AnalyticSolution<Dim>>;
  using normal_dot_numerical_flux =
      OptionTags::NumericalFluxParams<NumericalFlux<Dim>>;
  using component_list = tmpl::list<>;
  using const_global_cache_tag_list =
      tmpl::list<analytic_solution_tag, normal_dot_numerical_flux>;
};

template <size_t Dim>
using arguments_compute_tags = db::AddComputeTags<
    Tags::BoundaryDirectionsInterior<Dim>,
    Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<Dim>,
                               Tags::Direction<Dim>>,
    Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<Dim>,
                               Tags::InterfaceMesh<Dim>>,
    Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<Dim>,
                               Tags::BoundaryCoordinates<Dim, Frame::Inertial>>,
    Tags::InterfaceComputeItem<Tags::BoundaryDirectionsInterior<Dim>,
                               Tags::UnnormalizedFaceNormal<Dim>>,
    Tags::InterfaceComputeItem<
        Tags::BoundaryDirectionsInterior<Dim>,
        Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<Dim>>>,
    Tags::InterfaceComputeItem<
        Tags::BoundaryDirectionsInterior<Dim>,
        Tags::Normalized<Tags::UnnormalizedFaceNormal<Dim>>>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.Initialization.BoundaryConditions",
                  "[Unit][Elliptic][Actions]") {
  {
    INFO("1D");
    // Reference element:
    //    [X| | | ] -xi->
    //    ^       ^
    // -0.5       1.5
    const ElementId<1> element_id{0, {{SegmentId{2, 0}}}};
    const domain::creators::Interval<Frame::Inertial> domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};
    auto domain_box = Elliptic::Initialization::Domain<1>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<1>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    db::item_type<
        db::add_tag_prefix<Tags::Source, typename System<1>::fields_tag>>
        sources{4, 0.};
    auto arguments_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<db::add_tag_prefix<
                            Tags::Source, typename System<1>::fields_tag>>,
                        arguments_compute_tags<1>>(std::move(domain_box),
                                                   std::move(sources));

    ActionTesting::MockRuntimeSystem<Metavariables<1>> runner{
        {AnalyticSolution<1>{}, NumericalFlux<1>{}}, {}};

    const auto box = Elliptic::Initialization::BoundaryConditions<
        Metavariables<1>>::initialize(std::move(arguments_box), runner.cache());

    // Expected boundary contribution to source in element X:
    // [ -24 0 0 0 | -xi->
    // -0.5 (field) * 2. (num. flux) * 6. (inverse logical mass) * 4. (jacobian
    // for mass) = -24.
    // This expectation assumes the diagonal mass matrix approximation.
    const DataVector source_expected{-24., 0., 0., 0.};
    CHECK(get<Tags::Source<ScalarFieldTag>>(box) ==
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
    const ElementId<2> element_id{0, {{SegmentId{1, 0}, SegmentId{1, 1}}}};
    const domain::creators::Rectangle<Frame::Inertial> domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 3}}};
    auto domain_box = Elliptic::Initialization::Domain<2>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<2>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    db::item_type<
        db::add_tag_prefix<Tags::Source, typename System<3>::fields_tag>>
        sources{9, 0.};
    auto arguments_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<db::add_tag_prefix<
                            Tags::Source, typename System<2>::fields_tag>>,
                        arguments_compute_tags<2>>(std::move(domain_box),
                                                   std::move(sources));

    ActionTesting::MockRuntimeSystem<Metavariables<2>> runner{
        {AnalyticSolution<2>{}, NumericalFlux<2>{}}, {}};

    const auto box = Elliptic::Initialization::BoundaryConditions<
        Metavariables<2>>::initialize(std::move(arguments_box), runner.cache());

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
    CHECK(get<Tags::Source<ScalarFieldTag>>(box) ==
          Scalar<DataVector>(source_expected));
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}};
    const domain::creators::Brick<Frame::Inertial> domain_creator{
        {{-0.5, 0., -1.}},
        {{1.5, 1., 3.}},
        {{false, false, false}},
        {{1, 1, 1}},
        {{2, 2, 2}}};
    auto domain_box = Elliptic::Initialization::Domain<3>::initialize(
        db::DataBox<tmpl::list<>>{}, ElementIndex<3>{element_id},
        domain_creator.initial_extents(), domain_creator.create_domain());

    db::item_type<
        db::add_tag_prefix<Tags::Source, typename System<3>::fields_tag>>
        sources{8, 0.};
    auto arguments_box =
        db::create_from<db::RemoveTags<>,
                        db::AddSimpleTags<db::add_tag_prefix<
                            Tags::Source, typename System<3>::fields_tag>>,
                        arguments_compute_tags<3>>(std::move(domain_box),
                                                   std::move(sources));

    ActionTesting::MockRuntimeSystem<Metavariables<3>> runner{
        {AnalyticSolution<3>{}, NumericalFlux<3>{}}, {}};

    const auto box = Elliptic::Initialization::BoundaryConditions<
        Metavariables<3>>::initialize(std::move(arguments_box), runner.cache());

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
    CHECK(get<Tags::Source<ScalarFieldTag>>(box) ==
          Scalar<DataVector>(source_expected));
  }
}
