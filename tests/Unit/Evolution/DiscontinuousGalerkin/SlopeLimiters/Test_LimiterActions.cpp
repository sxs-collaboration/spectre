// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>  // IWYU pragma: keep
#include <cstddef>
#include <functional>
#include <map>
#include <memory>
#include <pup.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "Evolution/DiscontinuousGalerkin/SlopeLimiters/LimiterActions.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare Tensor

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

class DummyLimiterForTest {
 public:
  // Data sent by the limiter to its neighbors
  struct PackagedData {
    double mean_;
    Mesh<2> mesh_;
  };
  using package_argument_tags = tmpl::list<Var, Tags::Mesh<2>>;
  void package_data(const gsl::not_null<PackagedData*> packaged_data,
                    const Scalar<DataVector>& var, const Mesh<2>& mesh,
                    const OrientationMap<2>& orientation_map) const noexcept {
    packaged_data->mean_ = mean_value(get(var), mesh);
    packaged_data->mesh_ = orientation_map(mesh);
  }

  using limit_tags = tmpl::list<Var>;
  using limit_argument_tags = tmpl::list<Tags::Mesh<2>, Tags::Element<2>>;
  void operator()(const gsl::not_null<db::item_type<Var>*> var,
                  const Mesh<2>& /*mesh*/, const Element<2>& /*element*/,
                  const std::unordered_map<
                      std::pair<Direction<2>, ElementId<2>>,
                      DummyLimiterForTest::PackagedData,
                      boost::hash<std::pair<Direction<2>, ElementId<2>>>>&
                      neighbor_packaged_data) const noexcept {
    // Zero the data as an easy check that the limiter got called
    get(*var) = 0.;
    for (const auto& data : neighbor_packaged_data) {
      get(*var) += data.second.mean_;
    }
  }

  void pup(const PUP::er& /*p*/) const noexcept {}
};

struct LimiterTag {
  using type = DummyLimiterForTest;
};

template <size_t Dim, typename Metavariables>
struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<Dim>;
  using const_global_cache_tag_list = tmpl::list<LimiterTag>;
  using action_list =
      tmpl::list<SlopeLimiters::Actions::SendData<Metavariables>,
                 SlopeLimiters::Actions::Limit<Metavariables>>;
  using simple_tags =
      db::AddSimpleTags<TemporalId, Tags::Mesh<Dim>, Tags::Element<Dim>,
                        Tags::ElementMap<Dim>, Var>;
  using initial_databox = db::compute_databox_type<simple_tags>;
};

template <size_t Dim>
struct Metavariables {
  using component_list = tmpl::list<component<Dim, Metavariables>>;
  using const_global_cache_tag_list = tmpl::list<>;
  using limiter = LimiterTag;
  using system = System<Dim>;
  using temporal_id = TemporalId;
  static constexpr bool local_time_stepping = false;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.LimiterActions.Generic",
                  "[Unit][NumericalAlgorithms][Actions]") {
  using metavariables = Metavariables<2>;
  using my_component = component<2, metavariables>;
  using limiter_comm_tag =
      SlopeLimiters::Tags::LimiterCommunicationTag<metavariables>;

  const Mesh<2> mesh{
      {{3, 4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

  //      xi      Block       +- xi
  //      |     0   |   1     |
  // eta -+ +-------+-+-+---+ eta
  //        |       |X| |   |
  //        |       +-+-+   |
  //        |       | | |   |
  //        +-------+-+-+---+
  // We run the actions on the indicated element.  The blocks are square.
  const ElementId<2> self_id(1, {{{2, 0}, {1, 0}}});
  const ElementId<2> west_id(0);
  const ElementId<2> east_id(1, {{{2, 1}, {1, 0}}});
  const ElementId<2> south_id(1, {{{2, 0}, {1, 1}}});

  // OrientationMap from block 1 to block 0
  const OrientationMap<2> block_orientation(
      {{Direction<2>::upper_xi(), Direction<2>::upper_eta()}},
      {{Direction<2>::lower_eta(), Direction<2>::lower_xi()}});

  // Since we're lazy and use the same map for both blocks (the
  // actions are only sensitive to the ElementMap, which does differ),
  // we need to make the xi and eta maps line up along the block
  // interface.
  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};

  const auto coordmap =
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>(xi_map,
                                                                 eta_map));

  const struct {
    std::unordered_map<Direction<2>, Scalar<DataVector>> var;
  } test_data{
      {{Direction<2>::lower_xi(),
        Scalar<DataVector>(mesh.number_of_grid_points(), 5.)},
       {Direction<2>::upper_xi(),
        Scalar<DataVector>(mesh.number_of_grid_points(), 6.)},
       {Direction<2>::upper_eta(),
        Scalar<DataVector>(mesh.number_of_grid_points(), 7.)}},
  };

  auto start_box = [
    &self_id, &west_id, &east_id, &south_id, &block_orientation, &coordmap,
    &mesh
  ]() noexcept {
    const Element<2> element(
        self_id, {{Direction<2>::lower_xi(), {{west_id}, block_orientation}},
                  {Direction<2>::upper_xi(), {{east_id}, {}}},
                  {Direction<2>::upper_eta(), {{south_id}, {}}}});
    auto map = ElementMap<2, Frame::Inertial>(self_id, coordmap->get_clone());
    auto var = Scalar<DataVector>(mesh.number_of_grid_points(), 1234.);
    return db::create<my_component::simple_tags>(
        0, mesh, element, std::move(map), std::move(var));
  }
  ();

  const auto create_neighbor_databox = [&mesh, &self_id, &coordmap ](
      const ElementId<2>& id, const Direction<2>& direction,
      const OrientationMap<2>& orientation,
      const Scalar<DataVector>& var) noexcept {
    const Element<2> element(id, {{direction, {{self_id}, orientation}}});
    auto map = ElementMap<2, Frame::Inertial>(id, coordmap->get_clone());
    return db::create<my_component::simple_tags>(0, mesh, element,
                                                 std::move(map), var);
  };

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<my_component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id, std::move(start_box));
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(south_id, create_neighbor_databox(
                             south_id, Direction<2>::lower_eta(), {},
                             test_data.var.at(Direction<2>::upper_eta())));
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(east_id, create_neighbor_databox(
                            east_id, Direction<2>::lower_xi(), {},
                            test_data.var.at(Direction<2>::upper_xi())));
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(west_id, create_neighbor_databox(
                            west_id, Direction<2>::lower_eta(),
                            block_orientation.inverse_map(),
                            test_data.var.at(Direction<2>::lower_xi())));

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {DummyLimiterForTest{}}, std::move(dist_objects)};

  // Call SendDataForLimiter on self, sending data to neighbors
  runner.next_action<my_component>(self_id);

  // Here, we just check that messages are sent to the correct places.
  // We do not check the contents. We will check the contents of received
  // messages on self later
  {
    CHECK(runner.nonempty_inboxes<my_component, limiter_comm_tag>() ==
          std::unordered_set<ElementIndex<2>>{west_id, east_id, south_id});
    const auto check_sent_data = [&runner, &self_id ](
        const ElementId<2>& id, const Direction<2>& direction) noexcept {
      const auto& inboxes = runner.inboxes<my_component>();
      const auto& inbox = tuples::get<limiter_comm_tag>(inboxes.at(id));
      CHECK(inbox.size() == 1);
      CHECK(inbox.count(0) == 1);
      const auto& inbox_at_time = inbox.at(0);
      CHECK(inbox_at_time.size() == 1);
      CHECK(inbox_at_time.count({direction, self_id}) == 1);
    };
    check_sent_data(west_id, Direction<2>::lower_eta());
    check_sent_data(east_id, Direction<2>::lower_xi());
    check_sent_data(south_id, Direction<2>::lower_eta());
  }

  // Now we check ApplyLimiter
  CHECK_FALSE(runner.is_ready<my_component>(self_id));
  runner.next_action<my_component>(south_id);
  CHECK_FALSE(runner.is_ready<my_component>(self_id));
  runner.next_action<my_component>(east_id);
  CHECK_FALSE(runner.is_ready<my_component>(self_id));
  runner.next_action<my_component>(west_id);
  CHECK(runner.is_ready<my_component>(self_id));

  // ApplyLimiter::is_ready reports true, so here we check that the inbox is
  // correctly filled with information from neighbors.
  {
    const auto check_inbox = [&runner, &self_id ](
        const ElementId<2>& id, const Direction<2>& direction,
        const double expected_mean_data,
        const Mesh<2>& expected_mesh) noexcept {
      const auto received_package =
          tuples::get<limiter_comm_tag>(runner.inboxes<my_component>()[self_id])
              .at(0)
              .at(std::make_pair(direction, id));
      CHECK(received_package.mean_ == approx(expected_mean_data));
      CHECK(received_package.mesh_ == expected_mesh);
    };

    const Mesh<2> rotated_mesh{{{4, 3}},
                               Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
    check_inbox(west_id, Direction<2>::lower_xi(), 5., rotated_mesh);
    check_inbox(east_id, Direction<2>::upper_xi(), 6., mesh);
    check_inbox(south_id, Direction<2>::upper_eta(), 7., mesh);
  }

  // Now we run the ApplyLimiter action. We verify the pre- and post-limiting
  // state of the variable being limited.
  const auto& var_to_limit =
      db::get<Var>(runner.algorithms<my_component>()
                       .at(self_id)
                       .get_databox<my_component::initial_databox>());
  CHECK_ITERABLE_APPROX(
      var_to_limit, Scalar<DataVector>(mesh.number_of_grid_points(), 1234.));

  runner.next_action<my_component>(self_id);

  // Expected value: 18 = 5 + 6 + 7 from the three neighbors.
  CHECK_ITERABLE_APPROX(var_to_limit,
                        Scalar<DataVector>(mesh.number_of_grid_points(), 18.));

  // Check that data for this time (t=0) was deleted from the inbox after the
  // limiter call. Note that we only put in t=0 data, so the whole inbox should
  // be empty.
  CHECK(tuples::get<limiter_comm_tag>(runner.inboxes<my_component>()[self_id])
            .empty());
}

SPECTRE_TEST_CASE("Unit.Evolution.DG.SlopeLimiters.LimiterActions.NoNeighbors",
                  "[Unit][NumericalAlgorithms][Actions]") {
  using metavariables = Metavariables<2>;
  using my_component = component<2, metavariables>;
  using limiter_comm_tag =
      SlopeLimiters::Tags::LimiterCommunicationTag<metavariables>;

  const Mesh<2> mesh{
      {{3, 4}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const ElementId<2> self_id(1, {{{2, 0}, {1, 0}}});
  const Element<2> element(self_id, {});

  auto input_var = Scalar<DataVector>(mesh.number_of_grid_points(), 1234.);

  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., 7., 3.};
  auto map = ElementMap<2, Frame::Inertial>(
      self_id,
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>(xi_map,
                                                                 eta_map)));

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<my_component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(self_id,
               db::create<my_component::simple_tags>(
                   0, mesh, element, std::move(map), std::move(input_var)));

  ActionTesting::MockRuntimeSystem<metavariables> runner{
      {DummyLimiterForTest{}}, std::move(dist_objects)};

  // Call SendDataForLimiter on self. Expect empty inboxes all around.
  runner.next_action<my_component>(self_id);

  CHECK(runner.nonempty_inboxes<my_component, limiter_comm_tag>().empty());
  CHECK(runner.is_ready<my_component>(self_id));
  CHECK(tuples::get<limiter_comm_tag>(runner.inboxes<my_component>()[self_id])
            .empty());

  // Now we run the ApplyLimiter action, checking pre and post values.
  const auto& var_to_limit =
      db::get<Var>(runner.algorithms<my_component>()
                       .at(self_id)
                       .get_databox<my_component::initial_databox>());
  CHECK_ITERABLE_APPROX(
      var_to_limit, Scalar<DataVector>(mesh.number_of_grid_points(), 1234.));

  runner.next_action<my_component>(self_id);

  CHECK_ITERABLE_APPROX(var_to_limit,
                        Scalar<DataVector>(mesh.number_of_grid_points(), 0.));
}
