// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <memory>
#include <pup.h>
#include <string>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/FunctionsOfTime/RegisterDerivedWithCharm.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/SizeOfElement.hpp"  // IWYU pragma: keep
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/LimiterActions.hpp"  // IWYU pragma: keep
#include "Evolution/DiscontinuousGalerkin/Limiters/Minmod.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/MinmodType.hpp"
#include "Framework/ActionTesting.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Parallel/PhaseDependentActionList.hpp"  // IWYU pragma: keep
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare ActionTesting::InitializeDataBox

namespace {
struct TemporalId : db::SimpleTag {
  using type = int;
};

struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct System {
  static constexpr const size_t volume_dim = Dim;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
};

struct LimiterTag : db::SimpleTag {
  using type = Limiters::Minmod<2, tmpl::list<Var>>;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementId<Dim>;
  using const_global_cache_tags = tmpl::list<LimiterTag>;
  using simple_tags = db::AddSimpleTags<
      TemporalId, domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
      domain::Tags::ElementMap<Dim, Frame::Grid>,
      domain::CoordinateMaps::Tags::CoordinateMap<2, Frame::Grid,
                                                  Frame::Inertial>,
      ::Tags::Time, domain::Tags::FunctionsOfTime, Var>;
  using compute_tags = db::AddComputeTags<
      ::domain::Tags::LogicalCoordinates<Dim>,
      ::domain::Tags::MappedCoordinates<
          ::domain::Tags::ElementMap<Dim, Frame::Grid>,
          ::domain::Tags::Coordinates<Dim, Frame::ElementLogical>>,
      domain::Tags::SizeOfElementCompute<Dim>>;
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
  using CubicScaleMap = domain::CoordinateMaps::TimeDependent::CubicScale<2>;
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::BlockLogical, Frame::Grid, Affine2D>));
  PUPable_reg(SINGLE_ARG(
      domain::CoordinateMap<Frame::Grid, Frame::Inertial, CubicScaleMap>));
  domain::FunctionsOfTime::register_derived_with_charm();

  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};
  auto logical_to_grid_map = ElementMap<2, Frame::Grid>(
      self_id,
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
          Affine2D(xi_map, eta_map)));
  std::unique_ptr<domain::CoordinateMapBase<Frame::Grid, Frame::Inertial, 2>>
      grid_to_inertial_map =
          domain::make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
              CubicScaleMap{10.0, "Expansion", "Expansion"});

  const double initial_time = 0.0;
  const double expiration_time = 2.5;
  std::unordered_map<std::string,
                     std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>
      functions_of_time{};
  functions_of_time.insert(std::make_pair(
      "Expansion",
      std::make_unique<domain::FunctionsOfTime::PiecewisePolynomial<2>>(
          initial_time, std::array<DataVector, 3>{{{0.0}, {1.0}, {0.0}}},
          expiration_time)));

  auto var = Scalar<DataVector>(mesh.number_of_grid_points(), 1234.);

  const double tvb_constant = 0.0;
  ActionTesting::MockRuntimeSystem<metavariables> runner{
      Limiters::Minmod<2, tmpl::list<Var>>(Limiters::MinmodType::LambdaPi1,
                                           tvb_constant)};
  ActionTesting::emplace_component_and_initialize<my_component>(
      &runner, self_id,
      {0, mesh, element, std::move(logical_to_grid_map),
       std::move(grid_to_inertial_map), 1.0, std::move(functions_of_time),
       std::move(var)});
  ActionTesting::set_phase(make_not_null(&runner),
                           metavariables::Phase::Testing);

  // SendData
  runner.next_action<my_component>(self_id);
  // Limit
  CHECK(runner.next_action_if_ready<my_component>(self_id));
}
