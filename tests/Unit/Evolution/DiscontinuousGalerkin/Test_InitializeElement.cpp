// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <memory>
// IWYU pragma: no_include <pup.h>
#include <string>
#include <tuple>
#include <unordered_map>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"  // IWYU pragma: keep
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/Identity.hpp"
#include "Domain/CreateInitialElement.hpp"  // IWYU pragma: keep
#include "Domain/Direction.hpp"
#include "Domain/Domain.hpp"
#include "Domain/DomainCreators/Brick.hpp"
#include "Domain/DomainCreators/Interval.hpp"
#include "Domain/DomainCreators/Rectangle.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"  // IWYU pragma: keep
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SegmentId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/InitializeElement.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/FluxCommunicationTypes.hpp"
#include "NumericalAlgorithms/LinearOperators/Divergence.tpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.tpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "PointwiseFunctions/AnalyticSolutions/Tags.hpp"  // IWYU pragma: keep
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"  // IWYU pragma: keep
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Time/TimeSteppers/AdamsBashforthN.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare ElementIndex
// IWYU pragma: no_forward_declare Variables
namespace PUP {
class er;  // IWYU pragma: keep
}  // namespace PUP
namespace Tags {
template <typename Tag, size_t VolumeDim, typename Fr>
struct ComputeNormalDotFlux;
template <size_t Dim>
struct MortarSize;
template <typename Tag, size_t VolumeDim>
struct Mortars;
}  // namespace Tags
// IWYU pragma: no_forward_declare db::DataBox

namespace {
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Var"; }
  static constexpr bool should_be_sliced_to_boundary = true;
};

struct SystemAnalyticSolution {
  template <size_t Dim>
  tuples::TaggedTuple<Var> variables(const tnsr::I<DataVector, Dim>& x,
                                     double t, tmpl::list<Var> /*meta*/) const
      noexcept {
    tuples::TaggedTuple<Var> vars(x.get(0) + t);
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Var>(vars)) += x.get(d) + t;
    }
    return vars;
  }

  template <size_t Dim>
  tuples::TaggedTuple<Tags::dt<Var>> variables(
      const tnsr::I<DataVector, Dim>& x, double t,
      tmpl::list<Tags::dt<Var>> /*meta*/) const noexcept {
    tuples::TaggedTuple<Tags::dt<Var>> vars{
        Scalar<DataVector>(2.0 * x.get(0) + t)};
    for (size_t d = 1; d < Dim; ++d) {
      get(get<Tags::dt<Var>>(vars)) += 2.0 * x.get(d) + t;
    }
    return vars;
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim, bool IsConservative>
struct System {
  static constexpr bool is_conservative = IsConservative;
  static constexpr size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  using gradients_tags = tmpl::list<Var>;
  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

struct NormalDotNumericalFluxTag {
  using type = struct { using package_tags = tmpl::list<Var>; };
};

template <size_t Dim, bool IsConservative>
struct Metavariables;

template <size_t Dim, bool IsConservative>
using component = ActionTesting::MockArrayComponent<
    Metavariables<Dim, IsConservative>, ElementIndex<Dim>,
    tmpl::list<CacheTags::TimeStepper,
               CacheTags::AnalyticSolution<SystemAnalyticSolution>>,
    tmpl::list<dg::Actions::InitializeElement<Dim>>>;

template <size_t Dim, bool IsConservative>
struct Metavariables {
  using component_list = tmpl::list<component<Dim, IsConservative>>;
  using system = System<Dim, IsConservative>;
  using temporal_id = Tags::TimeId;
  using normal_dot_numerical_flux = NormalDotNumericalFluxTag;
  using const_global_cache_tag_list = tmpl::list<>;
};

template <typename Tag, typename Box>
bool box_contains(const Box& /*box*/) noexcept {
  return tmpl::list_contains_v<typename Box::tags_list, Tag>;
}

template <bool IsConservative>
struct TestConservativeOrNonconservativeParts {
  template <typename Metavariables, typename DbTags>
  static void apply(const gsl::not_null<db::DataBox<DbTags>*> box) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t dim = system::volume_dim;

    CHECK(box_contains<
          Tags::deriv<typename system::variables_tag::tags_list,
                      typename system::gradients_tags,
                      Tags::InverseJacobian<Tags::ElementMap<dim>,
                                            Tags::LogicalCoordinates<dim>>>>(
        *box));
  }
};

template <>
struct TestConservativeOrNonconservativeParts<true> {
  template <typename Metavariables, typename DbTags>
  static void apply(const gsl::not_null<db::DataBox<DbTags>*> box) noexcept {
    using system = typename Metavariables::system;
    constexpr size_t dim = system::volume_dim;
    using variables_tag = typename system::variables_tag;

    const size_t number_of_grid_points =
        get<Tags::Mesh<dim>>(*box).number_of_grid_points();

    CHECK(
        db::get<db::add_tag_prefix<Tags::Flux, Tags::Variables<tmpl::list<Var>>,
                                   tmpl::size_t<dim>, Frame::Inertial>>(*box)
            .number_of_grid_points() == number_of_grid_points);
    CHECK(db::get<db::add_tag_prefix<Tags::Source, variables_tag>>(*box)
              .number_of_grid_points() == number_of_grid_points);

    CHECK(box_contains<Tags::ComputeDiv<
              db::add_tag_prefix<Tags::Flux, variables_tag, tmpl::size_t<dim>,
                                 Frame::Inertial>,
              Tags::InverseJacobian<Tags::ElementMap<dim>,
                                    Tags::LogicalCoordinates<dim>>>>(*box));

    CHECK(box_contains<Tags::Interface<
              Tags::InternalDirections<dim>,
              Tags::ComputeNormalDotFlux<variables_tag, dim, Frame::Inertial>>>(
        *box));
  }
};

template <typename Metavariables, typename DomainCreatorType>
void test_initialize_element(
    const ElementId<Metavariables::system::volume_dim>& element_id,
    const DomainCreatorType& domain_creator) noexcept {
  using system = typename Metavariables::system;
  constexpr size_t dim = system::volume_dim;

  ActionTesting::ActionRunner<Metavariables> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(4, false),
       SystemAnalyticSolution{}}};

  const Slab slab = Slab::with_duration_from_start(0.3, 0.01);

  const auto domain = domain_creator.create_domain();

  db::DataBox<tmpl::list<>> empty_box{};
  auto box =
      std::get<0>(runner.template apply<component<dim, system::is_conservative>,
                                        dg::Actions::InitializeElement<dim>>(
          empty_box, element_id, domain_creator.initial_extents(),
          domain_creator.create_domain(), slab.start(), slab.duration()));
  CHECK(db::get<Tags::Next<Tags::TimeId>>(box) ==
        TimeId(true, 0, slab.start()));
  // The TimeId is uninitialized and is updated immediately by the
  // algorithm loop.
  CHECK(box_contains<Tags::TimeId>(box));
  CHECK(box_contains<Tags::Time>(box));
  CHECK(db::get<Tags::TimeStep>(box) == slab.duration());

  const auto& my_block = domain.blocks()[element_id.block_id()];
  ElementMap<dim, Frame::Inertial> map{element_id,
                                       my_block.coordinate_map().get_clone()};
  Element<dim> element = create_initial_element(element_id, my_block);
  Mesh<dim> mesh{domain_creator.initial_extents()[element_id.block_id()],
                 Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  auto logical_coords = logical_coordinates(mesh);
  auto inertial_coords = map(logical_coords);
  CHECK(db::get<Tags::LogicalCoordinates<dim>>(box) == logical_coords);
  CHECK(db::get<Tags::Mesh<dim>>(box) == mesh);
  CHECK(db::get<Tags::Element<dim>>(box) == element);
  CHECK(box_contains<Tags::ElementMap<dim>>(box));
  CHECK(db::get<Var>(box) == ([&inertial_coords, &slab]() {
          double time = slab.start().value();
          Scalar<DataVector> var{inertial_coords.get(0) + time};
          for (size_t d = 1; d < dim; ++d) {
            get(var) += inertial_coords.get(d) + time;
          }
          return var;
        }()));
  {
    const auto& history = db::get<Tags::HistoryEvolvedVariables<
        typename system::variables_tag,
        db::add_tag_prefix<Tags::dt, typename system::variables_tag>>>(
        box);
    TimeSteppers::AdamsBashforthN stepper(4, false);
    CHECK(history.size() == stepper.number_of_past_steps());
    const SystemAnalyticSolution solution{};
    Time past_t{slab.start()};
    TimeDelta past_dt{slab.duration()};
    for (size_t i = stepper.number_of_past_steps(); i > 0; --i) {
      const auto entry = history.begin() + static_cast<ssize_t>(i - 1);
      past_dt = past_dt.with_slab(past_dt.slab().advance_towards(-past_dt));
      past_t -= past_dt;

      CHECK(*entry == past_t);
      tmpl::for_each<tmpl::list<Var>>([&solution, &entry, &inertial_coords,
                                       &past_t](auto type_wrapped_tag) {
        using tag = tmpl::type_from<decltype(type_wrapped_tag)>;
        CHECK(get<tag>(entry.value()) ==
              get<tag>(solution.variables(inertial_coords, past_t.value(),
                                          tmpl::list<Var>{})));
        CHECK(
            get<Tags::dt<tag>>(entry.derivative()) ==
            get<Tags::dt<tag>>(solution.variables(
                inertial_coords, past_t.value(), tmpl::list<Tags::dt<Var>>{})));
      });
    }
  }
  CHECK((db::get<Tags::MappedCoordinates<Tags::ElementMap<dim>,
                                         Tags::LogicalCoordinates<dim>>>(
            box)) == inertial_coords);
  CHECK((db::get<Tags::InverseJacobian<Tags::ElementMap<dim>,
                                       Tags::LogicalCoordinates<dim>>>(box)) ==
        map.inv_jacobian(logical_coords));
  CHECK(db::get<
            db::add_tag_prefix<Tags::dt, typename system::variables_tag>>(
            box)
            .size() == mesh.number_of_grid_points());

  CHECK(db::get<typename dg::FluxCommunicationTypes<
            Metavariables>::simple_mortar_data_tag>(box)
            .size() == element.number_of_neighbors());
  CHECK(db::get<Tags::Mortars<Tags::Next<Tags::TimeId>, dim>>(box).size() ==
        element.number_of_neighbors());
  CHECK(db::get<Tags::Mortars<Tags::Mesh<dim - 1>, dim>>(box).size() ==
        element.number_of_neighbors());
  CHECK(db::get<Tags::Mortars<Tags::MortarSize<dim - 1>, dim>>(box).size() ==
        element.number_of_neighbors());

  CHECK(box_contains<Tags::Interface<Tags::InternalDirections<dim>,
                                     Tags::UnnormalizedFaceNormal<dim>>>(box));
  using magnitude_tag =
      Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<dim>>;
  CHECK(box_contains<
        Tags::Interface<Tags::InternalDirections<dim>, magnitude_tag>>(box));
  CHECK(box_contains<
        Tags::Interface<Tags::InternalDirections<dim>,
                        Tags::Normalized<Tags::UnnormalizedFaceNormal<dim>>>>(
      box));
  CHECK(box_contains<Tags::Interface<Tags::InternalDirections<dim>,
                                     typename system::variables_tag>>(box));
  CHECK(box_contains<Tags::Interface<Tags::BoundaryDirections<dim>,
                                     Tags::UnnormalizedFaceNormal<dim>>>(box));
  CHECK(box_contains<
        Tags::Interface<Tags::BoundaryDirections<dim>, magnitude_tag>>(box));
  CHECK(box_contains<
        Tags::Interface<Tags::BoundaryDirections<dim>,
                        Tags::Normalized<Tags::UnnormalizedFaceNormal<dim>>>>(
      box));
  CHECK(box_contains<Tags::Interface<Tags::BoundaryDirections<dim>,
                                     typename system::variables_tag>>(box));

  TestConservativeOrNonconservativeParts<system::is_conservative>::
      template apply<Metavariables>(make_not_null(&box));
}

void test_mortar_orientation() noexcept {
  ActionTesting::ActionRunner<Metavariables<3, false>> runner{
      {std::make_unique<TimeSteppers::AdamsBashforthN>(4, false),
       SystemAnalyticSolution{}}};
  const Slab slab(0., 1.);

  // This is the domain from the OrientationMap and corner numbering
  // tutorial.
  Domain<3, Frame::Inertial> domain(
      make_vector_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::Identity<3>{}, CoordinateMaps::Identity<3>{}),
      {{{0, 1, 3, 4, 6, 7, 9, 10}}, {{1, 4, 7, 10, 2, 5, 8, 11}}});
  const auto neighbor_direction = Direction<3>::upper_xi();
  const auto mortar_id = std::make_pair(neighbor_direction, ElementId<3>(1));
  const std::vector<std::array<size_t, 3>> extents{{{2, 2, 2}}, {{3, 4, 5}}};

  db::DataBox<tmpl::list<>> empty_box{};
  const auto box = std::get<0>(
      runner.apply<component<3, false>, dg::Actions::InitializeElement<3>>(
          empty_box, ElementId<3>(0), extents, std::move(domain), slab.start(),
          slab.duration()));

  CHECK(db::get<Tags::Mortars<Tags::Mesh<2>, 3>>(box).at(mortar_id).extents() ==
        Index<2>{{{3, 4}}});
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.dG.InitializeElement",
                  "[Unit][Evolution][Actions]") {
  test_initialize_element<Metavariables<1, false>>(
      ElementId<1>{0, {{SegmentId{2, 1}}}},
      DomainCreators::Interval<Frame::Inertial>{
          {{-0.5}}, {{1.5}}, {{false}}, {{2}}, {{4}}});

  test_initialize_element<Metavariables<2, false>>(
      ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 2}}}},
      DomainCreators::Rectangle<Frame::Inertial>{
          {{-0.5, -0.75}}, {{1.5, 2.4}}, {{false, false}}, {{2, 3}}, {{4, 5}}});

  test_initialize_element<Metavariables<3, false>>(
      ElementId<3>{0, {{SegmentId{2, 1}, SegmentId{3, 2}, SegmentId{1, 0}}}},
      DomainCreators::Brick<Frame::Inertial>{{{-0.5, -0.75, -1.2}},
                                             {{1.5, 2.4, 1.2}},
                                             {{false, false, true}},
                                             {{2, 3, 1}},
                                             {{4, 5, 3}}});

  test_initialize_element<Metavariables<2, true>>(
      ElementId<2>{0, {{SegmentId{2, 1}, SegmentId{3, 2}}}},
      DomainCreators::Rectangle<Frame::Inertial>{
          {{-0.5, -0.75}}, {{1.5, 2.4}}, {{false, false}}, {{2, 3}}, {{4, 5}}});

  test_mortar_orientation();
}
