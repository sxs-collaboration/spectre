// Distributed under the MIT License.
// See LICENSE.txt for details.

#include <catch.hpp>
#include <memory>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"
#include "tests/Unit/TestingFramework.hpp"

namespace {
struct Var : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Var";
  using type = Scalar<DataVector>;
};

struct System {
  static constexpr const size_t volume_dim = 2;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;
  static constexpr const size_t number_of_independent_components =
      db::item_type<variables_tag>::number_of_independent_components;

  struct compute_flux {
    static auto apply(const Variables<tmpl::list<Var>>& value) noexcept {
      using flux_tag = Tags::Flux<Var, tmpl::size_t<2>, Frame::Inertial>;
      Variables<tmpl::list<flux_tag>> result(value.number_of_grid_points());
      get<0>(get<flux_tag>(result)) = 10. * value;
      get<1>(get<flux_tag>(result)) = 20. * value;
      return result;
    }
  };
};

using dt_variables_tag = Tags::dt<Tags::Variables<tmpl::list<Tags::dt<Var>>>>;

class NumericalFlux {
 public:
  using flux_tag =
      db::add_tag_prefix<Tags::NormalDotFlux, System::variables_tag>;
  using argument_tags = tmpl::list<flux_tag>;
  db::item_type<flux_tag> operator()(
      const db::item_type<flux_tag>& self_fluxes,
      const db::item_type<flux_tag>& neighbor_fluxes) const noexcept {
    return 11. * self_fluxes + 1000. * neighbor_fluxes;
  }
};

struct NumericalFluxTag {
  using type = NumericalFlux;
};

using fluxes_tag = Actions::ComputeBoundaryFlux<System>::FluxesTag;
using history_tag =
    Tags::HistoryBoundaryVariables<Direction<2>, System::variables_tag>;
using Index_t = ElementIndex<2>;

struct Metavariables;
using component = ActionTesting::MockArrayComponent<
    Metavariables, Index_t, tmpl::list<NumericalFluxTag>,
    tmpl::list<Actions::ComputeBoundaryFlux<System>>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using numerical_flux = NumericalFluxTag;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication",
                  "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};

  const Slab slab(1., 3.);
  const TimeId time_id{8, slab.start(), 0};

  const Index<2> extents{{{3, 3}}};

  //      xi  Block   +- xi
  //      |   0   1   |
  // eta -+ +---+-+-+ eta
  //        |   |X| |
  //        |   +++-+
  //        |   ||| |
  //        +---+++-+
  // We run the actions on the indicated element.
  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});
  const ElementId<2> west_id(0);
  const ElementId<2> east_id(1, {{{1, 1}, {1, 0}}});
  // These are the halves of the box to the south, not elements in
  // diagonal directions.
  const ElementId<2> south_west_id(1, {{{2, 0}, {1, 1}}});
  const ElementId<2> south_east_id(1, {{{2, 1}, {1, 1}}});

  // OrientationMap from block 1 to block 0
  const OrientationMap<2> block_orientation(
      {{Direction<2>::upper_xi(), Direction<2>::upper_eta()}},
      {{Direction<2>::lower_eta(), Direction<2>::lower_xi()}});

  const CoordinateMaps::Affine xi_map{-1., 1., 3., 7.};
  const CoordinateMaps::Affine eta_map{-1., 1., -2., 4.};

  auto start_box =
      [&extents, &time_id, &self_id, &west_id, &east_id, &south_west_id,
       &south_east_id, &block_orientation, &xi_map, &eta_map]() {
    const Element<2> element(
        self_id,
       {{Direction<2>::lower_xi(), {{west_id}, block_orientation}},
        {Direction<2>::upper_xi(), {{east_id}, {}}},
        {Direction<2>::upper_eta(), {{south_west_id, south_east_id}, {}}}});

    auto map = ElementMap<2, Frame::Inertial>(
        ElementId<2>{0},
        make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
            CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                           CoordinateMaps::Affine>(xi_map,
                                                                   eta_map)));

    Variables<tmpl::list<Var>> variables(extents.product());
    get<Var>(variables).get() = DataVector{1., 2., 3., 4., 5., 6., 7., 8., 9.};
    db::item_type<db::add_tag_prefix<Tags::dt, System::variables_tag>>
        dt_variables(extents.product(), 0.0);

    return db::create<
        db::AddTags<Tags::TimeId, Tags::Extents<2>, Tags::Element<2>,
                    Tags::ElementMap<2>, System::variables_tag,
                    db::add_tag_prefix<Tags::dt, System::variables_tag>>,
        db::AddComputeItemsTags<Tags::UnnormalizedFaceNormal<2>>>(
        time_id, extents, element, std::move(map), std::move(variables),
        std::move(dt_variables));
  }();

  auto sent_box = std::get<0>(
      runner.apply<component, Actions::SendDataForFluxes>(start_box, self_id));

  // Check local state
  auto local_boundaries = db::get<history_tag>(sent_box);
  CHECK(local_boundaries.size() == 3);
  CHECK(get<Var>(local_boundaries[Direction<2>::lower_xi()]).get() ==
        (DataVector{1., 4., 7.}));
  CHECK(get<Var>(local_boundaries[Direction<2>::upper_xi()]).get() ==
        (DataVector{3., 6., 9.}));
  CHECK(get<Var>(local_boundaries[Direction<2>::upper_eta()]).get() ==
        (DataVector{7., 8., 9.}));

  // Check sent data
  CHECK((runner.nonempty_inboxes<component, fluxes_tag>()) ==
        (std::unordered_set<Index_t>{west_id, east_id, south_west_id,
                                     south_east_id}));
  auto& inboxes = runner.inboxes<component>();
  const auto flux_inbox =
      [&inboxes, &time_id](const ElementId<2>& id) noexcept {
    return tuples::get<fluxes_tag>(inboxes[id])[time_id];
  };
  CHECK(flux_inbox(west_id).size() == 1);
  CHECK(get<Var>(flux_inbox(west_id)[{Direction<2>::lower_eta(), self_id}])
        .get()
        == (DataVector{7., 4., 1.}));
  CHECK(flux_inbox(east_id).size() == 1);
  CHECK(get<Var>(flux_inbox(east_id)[{Direction<2>::lower_xi(), self_id}]).get()
        == (DataVector{3., 6., 9.}));
  CHECK(flux_inbox(south_west_id).size() == 1);
  CHECK(
      get<Var>(flux_inbox(south_west_id)[{Direction<2>::lower_eta(), self_id}])
      .get()
      == (DataVector{7., 8., 9.}));
  CHECK(flux_inbox(south_east_id).size() == 1);
  CHECK(
      get<Var>(flux_inbox(south_east_id)[{Direction<2>::lower_eta(), self_id}])
      .get()
      == (DataVector{7., 8., 9.}));

  // Now check ComputeBoundaryFlux
  // We can't handle complex boundaries yet, but the actions currently
  // ignore the complexity and just pretend the elements are
  // conforming, so we can use that for the test until h-refinement is
  // implemented.  We do need to get rid of the "extra" southern
  // neighbor, though.
  db::mutate<Tags::Element<2>>(
      sent_box,
      [&south_west_id](auto& element) {
        auto neighbors = element.neighbors();
        neighbors[Direction<2>::upper_eta()].set_ids_to({south_west_id});
        element = Element<2>(element.id(), std::move(neighbors));
      });

  CHECK_FALSE((runner.is_ready<component, Actions::ComputeBoundaryFlux<System>>(
      sent_box, self_id)));

  const auto send_data = [&extents, &runner, &self_id, &time_id](
      const ElementId<2>& id, const Direction<2>& direction,
      const OrientationMap<2>& orientation, DataVector volume_data) noexcept {
    const Element<2> element(id, {{direction, {{self_id}, orientation}}});

    Variables<tmpl::list<Var>> variables(extents.product());
    get<Var>(variables).get() = std::move(volume_data);

    auto box = db::create<db::AddTags<Tags::TimeId, Tags::Extents<2>,
                                      Tags::Element<2>, System::variables_tag>>(
        time_id, extents, element, std::move(variables));
    runner.apply<component, Actions::SendDataForFluxes>(box, id);
  };
  send_data(south_west_id, Direction<2>::lower_eta(), {},
            DataVector{11., 12., 13., 14., 15., 16., 17., 18., 19.});
  CHECK_FALSE((runner.is_ready<component, Actions::ComputeBoundaryFlux<System>>(
      sent_box, self_id)));
  send_data(east_id, Direction<2>::lower_xi(), {},
            DataVector{21., 22., 23., 24., 25., 26., 27., 28., 29.});
  CHECK_FALSE((runner.is_ready<component, Actions::ComputeBoundaryFlux<System>>(
      sent_box, self_id)));
  send_data(west_id, Direction<2>::lower_eta(), block_orientation.inverse_map(),
            DataVector{31., 32., 33., 34., 35., 36., 37., 38., 39.});
  CHECK((runner.is_ready<component, Actions::ComputeBoundaryFlux<System>>(
      sent_box, self_id)));

  auto received_box = std::get<0>(
      runner.apply<component, Actions::ComputeBoundaryFlux<System>>(
          sent_box, self_id));

  CHECK(tuples::get<fluxes_tag>(runner.inboxes<component>()[self_id])
            .empty());

  const double element_length_xi = (xi_map(std::array<double, 1>{{1.}}) -
                                    xi_map(std::array<double, 1>{{-1.}}))[0];
  const double element_length_eta = (eta_map(std::array<double, 1>{{1.}}) -
                                     eta_map(std::array<double, 1>{{-1.}}))[0];
  // Prefactor and weight as in Kopriva 8.42.  Equal to
  // -2/(element length)/w_0  with  w_0 = 2/(N(N-1))  where here N=3.
  const double xi_lift = -6./element_length_xi;
  const double eta_lift = -6./element_length_eta;
  // n.(F* - F)
  // For the test we set n.F* = 11 n.F + 1000 n.F_nbr, so
  //      n.(F* - F) = 10 n.F + 1000 n.F_nbr
  // F_xi = 10 U    F_eta = 20 U
  const DataVector xi_boundaries{-330100., 0., 210300.,
                                 -320400., 0., 240600.,
                                 -310700., 0., 270900.};
  const DataVector eta_boundaries{0., 0., 0.,
                                  0., 0., 0.,
                                  221400., 241600., 261800.};

  CHECK_ITERABLE_APPROX(
      get<Tags::dt<Var>>(db::get<dt_variables_tag>(received_box)).get(),
      xi_lift * xi_boundaries + eta_lift * eta_boundaries);
}

SPECTRE_TEST_CASE(
    "Unit.DiscontinuousGalerkin.Actions.FluxCommunication.NoNeighbors",
    "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};

  const Slab slab(1., 3.);
  const TimeId time_id{8, slab.start(), 0};

  const Index<2> extents{{{3, 3}}};

  const ElementId<2> self_id(1, {{{1, 0}, {1, 0}}});

  const Element<2> element(self_id, {});

  auto map = ElementMap<2, Frame::Inertial>(
      self_id, make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                   CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                  CoordinateMaps::Affine>(
                       {-1., 1., 3., 7.}, {-1., 1., -2., 4.})));

  Variables<tmpl::list<Var>> variables(extents.product());
  get<Var>(variables).get() = DataVector{1., 2., 3., 4., 5., 6., 7., 8., 9.};
  db::item_type<db::add_tag_prefix<Tags::dt, System::variables_tag>>
      dt_variables(extents.product(), 0.0);
  auto start_box = db::create<
      db::AddTags<Tags::TimeId, Tags::Extents<2>, Tags::Element<2>,
                  Tags::ElementMap<2>, System::variables_tag,
                  db::add_tag_prefix<Tags::dt, System::variables_tag>>,
      db::AddComputeItemsTags<Tags::UnnormalizedFaceNormal<2>>>(
      time_id, extents, element, std::move(map), std::move(variables),
      std::move(dt_variables));

  auto sent_box =
      std::get<0>(runner.apply<component, Actions::SendDataForFluxes>(
          start_box, self_id));

  CHECK(db::get<history_tag>(sent_box).empty());
  CHECK((runner.nonempty_inboxes<component, fluxes_tag>().empty()));

  CHECK((runner.is_ready<component, Actions::ComputeBoundaryFlux<System>>(
      sent_box, self_id)));

  auto received_box =
      std::get<0>(runner.apply<component, Actions::ComputeBoundaryFlux<System>>(
          sent_box, self_id));

  CHECK(get<Tags::dt<Var>>(db::get<dt_variables_tag>(received_box))
            .get() == (DataVector{0., 0., 0., 0., 0., 0., 0., 0., 0.}));
}
