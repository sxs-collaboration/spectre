// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/SizeOfElement.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.tpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/AddOptionsToDataBox.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {
struct TemporalId : db::SimpleTag {
  static std::string name() noexcept { return "TemporalId"; }
  using type = int;
};

struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr const size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
};

struct LimiterTag : db::SimpleTag {
  using type = Limiters::Minmod<2, tmpl::list<Var>>;
  using container_tag = LimiterTag;
  static std::string name() noexcept { return "LimiterTag"; }
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<LimiterTag>;
  using add_options_to_databox = Parallel::AddNoOptionsToDataBox;
  using simple_tags =
      db::AddSimpleTags<TemporalId, Tags::Mesh<Dim>, Tags::Element<Dim>,
                        Tags::ElementMap<Dim>,
                        Tags::Coordinates<Dim, Frame::Logical>,
                        Tags::Coordinates<Dim, Frame::Inertial>, Var>;
  using compute_tags = db::AddComputeTags<Tags::SizeOfElement<Dim>>;
  using phase_dependent_action_list = tmpl::list<
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Initialization,
          tmpl::list<
              ActionTesting::InitializeDataBox<simple_tags, compute_tags>>>,
      Parallel::PhaseActions<
          typename Metavariables::Phase, Metavariables::Phase::Testing,
          tmpl::list<Limiters::Actions::SendData<Metavariables>,
                     Limiters::Actions::Limit<Metavariables>>>>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using limiter = LimiterTag;
  using system = System<Dim>;
  using temporal_id = TemporalId;
  static constexpr bool local_time_stepping = false;
  enum class Phase { Initialization, Testing, Exit };
};
}  // namespace

// This test checks that the Minmod limiter's interfaces and type aliases
// succesfully integrate with the limiter actions. It does this by compiling
// together the Minmod limiter and the actions, then making calls to the
// SendData and the Limit actions. No checks are performed here that the limiter
// and/or actions produce correct output: that is done in other tests.
SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.LimiterActions.Minmod",
                  "[Unit][NumericalAlgorithms][Actions]") {
  using metavariables = Metavariables<2>;
  using my_component = component<2, metavariables>;

  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};
  const ElementId<2> self_id(1, {{{2, 0}, {1, 0}}});
  const Element<2> element(self_id, {});

  using Affine = domain::CoordinateMaps::Affine;
  using Affine2D = domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>;
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Logical, Frame::Inertial, Affine2D>));
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};
  auto map = ElementMap<2, Frame::Inertial>(
      self_id,
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          Affine2D(xi_map, eta_map)));

  auto logical_coords = logical_coordinates(mesh);
  auto inertial_coords = map(logical_coords);
  auto var = Scalar<DataVector>(mesh.number_of_grid_points(), 1234.);

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      Limiters::Minmod<2, tmpl::list<Var>>(Limiters::MinmodType::LambdaPi1)};
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {0, mesh, element, std::move(map), std::move(logical_coords),
       std::move(inertial_coords), std::move(var)});
  runner.set_phase(metavariables::Phase::Testing);

  // SendData
  runner.next_action<my_component>(self_id);
  CHECK(runner.is_ready<my_component>(self_id));
  // Limit
  runner.next_action<my_component>(self_id);
}
