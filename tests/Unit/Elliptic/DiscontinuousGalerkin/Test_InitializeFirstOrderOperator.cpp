// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <pup.h>
#include <string>

#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/PrefixHelpers.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Creators/Brick.hpp"
#include "Domain/Creators/Interval.hpp"
#include "Domain/Creators/Rectangle.hpp"
#include "Domain/Creators/RegisterDerivedWithCharm.hpp"
#include "Domain/Domain.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Elliptic/DiscontinuousGalerkin/InitializeFirstOrderOperator.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeDomain.hpp"
#include "ParallelAlgorithms/DiscontinuousGalerkin/InitializeInterfaces.hpp"
#include "ParallelAlgorithms/Initialization/Actions/AddComputeTags.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct ScalarFieldTag : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct AuxiliaryFieldTag : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

template <size_t Dim>
using vars_tag =
    Tags::Variables<tmpl::list<ScalarFieldTag, AuxiliaryFieldTag<Dim>>>;
template <size_t Dim>
using fluxes_tag = db::add_tag_prefix<::Tags::Flux, vars_tag<Dim>,
                                      tmpl::size_t<Dim>, Frame::Inertial>;
template <size_t Dim>
using div_fluxes_tag = db::add_tag_prefix<::Tags::div, fluxes_tag<Dim>>;
template <size_t Dim>
using n_dot_fluxes_tag =
    db::add_tag_prefix<::Tags::NormalDotFlux, vars_tag<Dim>>;
template <size_t Dim>
using sources_tag = db::add_tag_prefix<::Tags::Source, vars_tag<Dim>>;
template <size_t Dim>
using inv_jacobian_tag = domain::Tags::InverseJacobianCompute<
    domain::Tags::ElementMap<Dim>,
    domain::Tags::Coordinates<Dim, Frame::Logical>>;

template <size_t Dim>
struct Fluxes {
  using argument_tags = tmpl::list<>;
  static void apply(
      const gsl::not_null<tnsr::I<DataVector, Dim>*> flux_for_field,
      const tnsr::i<DataVector, Dim>& auxiliary_field) {
    for (size_t d = 0; d < Dim; d++) {
      flux_for_field->get(d) = auxiliary_field.get(d);
    }
  }
  static void apply(
      const gsl::not_null<tnsr::Ij<DataVector, Dim>*> flux_for_aux_field,
      const Scalar<DataVector>& field) {
    for (size_t d = 0; d < Dim; d++) {
      flux_for_aux_field->get(d, d) = get(field);
    }
  }
  // clang-tidy: no runtime references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

struct Sources {
  using argument_tags = tmpl::list<>;
  static void apply(const gsl::not_null<Scalar<DataVector>*> source_for_field,
                    const Scalar<DataVector>& field) {
    get(*source_for_field) = get(field);
  }
};

template <size_t Dim, typename Metavariables>
struct ElementArray {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<domain::Tags::Domain<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<ActionTesting::InitializeDataBox<tmpl::list<
                         domain::Tags::InitialRefinementLevels<Dim>,
                         domain::Tags::InitialExtents<Dim>, vars_tag<Dim>>>,
                     dg::Actions::InitializeDomain<Dim>,
                     dg::Actions::InitializeInterfaces<
                         typename Metavariables::system,
                         dg::Initialization::slice_tags_to_face<>,
                         dg::Initialization::slice_tags_to_exterior<>,
                         dg::Initialization::face_compute_tags<>,
                         dg::Initialization::exterior_compute_tags<>, false>>>,

      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<elliptic::dg::Actions::InitializeFirstOrderOperator<
              Dim, Fluxes<Dim>, Sources, vars_tag<Dim>,
              tmpl::list<ScalarFieldTag>,
              tmpl::list<AuxiliaryFieldTag<Dim>>>>>>;
};

template <size_t Dim>
struct System {
  static constexpr size_t volume_dim = Dim;
  template <typename Tag>
  using magnitude_tag = ::Tags::EuclideanMagnitude<Tag>;
};

template <size_t Dim>
struct Metavariables {
  using system = System<Dim>;
  using element_array = ElementArray<Dim, Metavariables>;
  using component_list = tmpl::list<element_array>;
  using const_global_cache_tags =
      tmpl::list<elliptic::Tags::FluxesComputer<Fluxes<Dim>>>;
  enum class Phase { Initialization, Testing, Exit };
};

template <size_t Dim>
void check_compute_items(
    const ActionTesting::MockRuntimeSystem<Metavariables<Dim>>& runner,
    const ElementId<Dim>& element_id) {
  // The compute items themselves are tested elsewhere, so just check if they
  // were indeed added by the initializer
  const auto tag_is_retrievable = [&runner,
                                   &element_id](auto tag_v) -> decltype(auto) {
    using tag = std::decay_t<decltype(tag_v)>;
    return ActionTesting::tag_is_retrievable<
        ElementArray<Dim, Metavariables<Dim>>, tag>(runner, element_id);
  };
  // Volume items
  using volume_tags =
      tmpl::list<fluxes_tag<Dim>, div_fluxes_tag<Dim>, sources_tag<Dim>>;
  tmpl::for_each<volume_tags>([&tag_is_retrievable](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CAPTURE(db::tag_name<tag>());
    CHECK(tag_is_retrievable(tag{}));
  });
  // Interface items
  using face_tags = tmpl::list<vars_tag<Dim>, fluxes_tag<Dim>,
                               div_fluxes_tag<Dim>, n_dot_fluxes_tag<Dim>>;
  tmpl::for_each<face_tags>([&tag_is_retrievable](auto tag_v) {
    using tag = tmpl::type_from<decltype(tag_v)>;
    CAPTURE(db::tag_name<tag>());
    CHECK(tag_is_retrievable(
        domain::Tags::Interface<domain::Tags::InternalDirections<Dim>, tag>{}));
    CHECK(tag_is_retrievable(
        domain::Tags::Interface<domain::Tags::BoundaryDirectionsInterior<Dim>,
                                tag>{}));
    CHECK(tag_is_retrievable(
        domain::Tags::Interface<domain::Tags::BoundaryDirectionsExterior<Dim>,
                                tag>{}));
  });
}

template <size_t Dim>
void test_initialize_fluxes(const DomainCreator<Dim>& domain_creator,
                            const ElementId<Dim>& element_id) {
  using metavariables = Metavariables<Dim>;
  using element_array = typename metavariables::element_array;

  auto initial_extents = domain_creator.initial_extents();
  const size_t num_points =
      alg::accumulate(gsl::at(initial_extents, element_id.block_id()),
                      size_t{1}, funcl::Multiplies<>{});
  db::item_type<vars_tag<Dim>> vars{num_points, 0.};

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {Fluxes<Dim>{}, domain_creator.create_domain()}};
  ActionTesting::emplace_component_and_initialize<element_array>(
      &runner, element_id,
      {domain_creator.initial_refinement_levels(), std::move(initial_extents),
       std::move(vars)});
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);
  ActionTesting::next_action<element_array>(make_not_null(&runner), element_id);

  check_compute_items(runner, element_id);

  // Check the variables on exterior boundaries exist (for imposing boundary
  // conditions)
  const auto& exterior_boundary_vars = ActionTesting::get_databox_tag<
      element_array,
      domain::Tags::Interface<domain::Tags::BoundaryDirectionsExterior<Dim>,
                              vars_tag<Dim>>>(runner, element_id);
  const auto& exterior_directions = ActionTesting::get_databox_tag<
      element_array, domain::Tags::BoundaryDirectionsExterior<Dim>>(runner,
                                                                    element_id);
  for (const auto& direction : exterior_directions) {
    CHECK(exterior_boundary_vars.find(direction) !=
          exterior_boundary_vars.end());
  }
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Elliptic.DG.Actions.InitializeFirstOrderOperator",
                  "[Unit][Elliptic][Actions]") {
  domain::creators::register_derived_with_charm();
  {
    INFO("1D");
    // Reference element:
    // [X| | | ]-> xi
    const ElementId<1> element_id{0, {{{2, 0}}}};
    const domain::creators::Interval domain_creator{
        {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}};

    test_initialize_fluxes(domain_creator, element_id);
  }
  {
    INFO("2D");
    // Reference element:
    // ^ eta
    // +-+-+> xi
    // |X| |
    // +-+-+
    // | | |
    // +-+-+
    const ElementId<2> element_id{0, {{{1, 0}, {1, 1}}}};
    const domain::creators::Rectangle domain_creator{
        {{-0.5, 0.}}, {{1.5, 1.}}, {{false, false}}, {{1, 1}}, {{3, 2}}};

    test_initialize_fluxes(domain_creator, element_id);
  }
  {
    INFO("3D");
    const ElementId<3> element_id{
        0, {{SegmentId{1, 0}, SegmentId{1, 1}, SegmentId{1, 0}}}};
    const domain::creators::Brick domain_creator{{{-0.5, 0., -1.}},
                                                 {{1.5, 1., 3.}},
                                                 {{false, false, false}},
                                                 {{1, 1, 1}},
                                                 {{2, 3, 4}}};

    test_initialize_fluxes(domain_creator, element_id);
  }
}
