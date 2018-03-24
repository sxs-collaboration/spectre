// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <tuple>
#include <unordered_map>
#include <unordered_set>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
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
#include "Domain/FaceNormal.hpp"
#include "Domain/Neighbors.hpp"
#include "Domain/OrientationMap.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/FluxCommunication.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Tags.hpp"
#include "Time/Slab.hpp"
#include "Time/Tags.hpp"
#include "Time/Time.hpp"
#include "Time/TimeId.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "Utilities/TypeTraits.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
struct Var : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Var";
  using type = Scalar<DataVector>;
};

struct OtherData : db::DataBoxTag {
  static constexpr db::DataBoxString label = "OtherData";
  using type = Scalar<DataVector>;
  static constexpr bool should_be_sliced_to_boundary = false;
};

class NumericalFlux {
 public:
  struct ExtraData : db::DataBoxTag {
    static constexpr db::DataBoxString label = "ExtraTag";
    using type = tnsr::I<DataVector, 1>;
  };

  using package_tags = tmpl::list<ExtraData, Var>;
  // This is a silly set of things to request, but it tests not
  // requesting the evolved variables and requesting multiple other
  // things.
  using slice_tags = tmpl::list<Tags::NormalDotFlux<Var>, OtherData>;
  void package_data(const gsl::not_null<Variables<package_tags>*> packaged_data,
                    const Scalar<DataVector>& var_flux,
                    const Scalar<DataVector>& var_flux2,
                    const Scalar<DataVector>& other_data,
                    const tnsr::i<DataVector, 2, Frame::Inertial>&
                        interface_unit_normal) const noexcept {
    CHECK(var_flux == var_flux2);
    get(get<Var>(*packaged_data)) = 10. * get(var_flux);
    get<0>(get<ExtraData>(*packaged_data)) =
        get(other_data) + 2. * get<0>(interface_unit_normal) +
        3. * get<1>(interface_unit_normal);
  }

  void operator()(
      const gsl::not_null<Scalar<DataVector>*> normal_dot_numerical_flux,
      const tnsr::I<DataVector, 1>& extra_data_interior,
      const Scalar<DataVector>& packaged_var_interior,
      const tnsr::I<DataVector, 1>& extra_data_exterior,
      const Scalar<DataVector>& packaged_var_exterior) const noexcept {
    get(*normal_dot_numerical_flux) =
        1.1 * get(packaged_var_interior) + 100. * get(packaged_var_exterior);
    // We can't easily get an expected value in here, so this
    // expression is chosen so similar errors on the two interfaces
    // are unlikely to cancel out.  We tune the exterior data we feed
    // in to make it pass.
    CHECK(get<0>(extra_data_exterior) == 2. * get<0>(extra_data_interior) + 1.);
  }

  // clang-tidy: do not use references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

struct NumericalFluxTag {
  using type = NumericalFlux;
};

struct System {
  static constexpr const size_t volume_dim = 2;
  using variables_tag = Tags::Variables<tmpl::list<Var>>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;
};

struct Metavariables;
using send_data_for_fluxes = dg::Actions::SendDataForFluxes;
using compute_boundary_flux = dg::Actions::ComputeBoundaryFlux<Metavariables>;

using component =
    ActionTesting::MockArrayComponent<Metavariables, ElementIndex<2>,
                                      tmpl::list<NumericalFluxTag>,
                                      tmpl::list<compute_boundary_flux>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;

  using normal_dot_numerical_flux = NumericalFluxTag;
};

using dt_variables_tag = Tags::dt<Tags::Variables<tmpl::list<Tags::dt<Var>>>>;
using fluxes_tag = compute_boundary_flux::FluxesTag;
using history_tag = Tags::Mortars<System::variables_tag, 2>;

template <typename Tag>
using interface_tag = Tags::Interface<Tags::InternalDirections<2>, Tag>;

using normal_dot_fluxes_tag = interface_tag<
    db::add_tag_prefix<Tags::NormalDotFlux, System::variables_tag>>;
using other_data_tag = interface_tag<Tags::Variables<tmpl::list<OtherData>>>;

using compute_items = db::AddComputeTags<
    Tags::InternalDirections<2>, interface_tag<Tags::Direction<2>>,
    interface_tag<Tags::Extents<1>>,
    interface_tag<Tags::UnnormalizedFaceNormal<2>>,
    interface_tag<Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>,
    interface_tag<Tags::Normalized<
        Tags::UnnormalizedFaceNormal<2>,
        Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.Actions.FluxCommunication",
                  "[Unit][NumericalAlgorithms][Actions]") {
  ActionTesting::ActionRunner<Metavariables> runner{{}};

  const Slab slab(1., 3.);
  const TimeId time_id{8, slab.start(), 0};

  const Index<2> extents{{{3, 3}}};

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
  const CoordinateMaps::Affine xi_map{-1., 1., 3., 7.};
  const CoordinateMaps::Affine eta_map{-1., 1., 7., 3.};

  const auto coordmap =
      make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                         CoordinateMaps::Affine>(xi_map,
                                                                 eta_map));

  auto start_box = [
    &extents, &time_id, &self_id, &west_id, &east_id, &south_id,
    &block_orientation, &coordmap
  ]() noexcept {
    const Element<2> element(
        self_id, {{Direction<2>::lower_xi(), {{west_id}, block_orientation}},
                  {Direction<2>::upper_xi(), {{east_id}, {}}},
                  {Direction<2>::upper_eta(), {{south_id}, {}}}});

    auto map = ElementMap<2, Frame::Inertial>(self_id, coordmap->get_clone());

    // The initial value for dt_variables checks that it isn't
    // improperly zeroed.
    db::item_type<dt_variables_tag> dt_variables(extents.product(), 3.0);
    db::item_type<normal_dot_fluxes_tag> normal_dot_fluxes;
    {
      const auto set_flux_in_direction = [&normal_dot_fluxes](
          const Direction<2>& direction, const DataVector& flux) noexcept {
        auto& flux_vars = normal_dot_fluxes[direction];
        flux_vars.initialize(3);
        get(get<Tags::NormalDotFlux<Var>>(flux_vars)) = flux;
      };
      set_flux_in_direction(Direction<2>::lower_xi(), {1., 2., 3.});
      set_flux_in_direction(Direction<2>::upper_xi(), {4., 5., 6.});
      set_flux_in_direction(Direction<2>::upper_eta(), {7., 8., 9.});
    }

    db::item_type<other_data_tag> other_data;
    {
      const auto set_other_data_in_direction = [&other_data](
          const Direction<2>& direction, const DataVector& data) noexcept {
        auto& other_data_vars = other_data[direction];
        other_data_vars.initialize(3);
        get(get<OtherData>(other_data_vars)) = data;
      };
      set_other_data_in_direction(Direction<2>::lower_xi(), {15., 25., 35.});
      set_other_data_in_direction(Direction<2>::upper_xi(), {45., 55., 65.});
      set_other_data_in_direction(Direction<2>::upper_eta(), {75., 85., 95.});
    }

    return db::create<
        db::AddSimpleTags<Tags::TimeId, Tags::Extents<2>, Tags::Element<2>,
                          Tags::ElementMap<2>, dt_variables_tag,
                          normal_dot_fluxes_tag, other_data_tag>,
        compute_items>(time_id, extents, element, std::move(map),
                       std::move(dt_variables), std::move(normal_dot_fluxes),
                       std::move(other_data));
  }();

  auto sent_box = std::get<0>(
      runner.apply<component, send_data_for_fluxes>(start_box, self_id));

  // The check in NumericalFlux::operator() (and the final dt check)
  // verify that the action sends the correct values, but we need to
  // check that it sends the correct number of messages to the correct
  // places.
  {
    CHECK((runner.nonempty_inboxes<component, fluxes_tag>()) ==
          (std::unordered_set<ElementIndex<2>>{west_id, east_id, south_id}));
    const auto check_sent_data = [&runner, &time_id, &self_id](
        const ElementId<2>& id, const Direction<2>& direction) noexcept {
      const auto& inboxes = runner.inboxes<component>();
      const auto& flux_inbox = tuples::get<fluxes_tag>(inboxes.at(id));
      CHECK(flux_inbox.size() == 1);
      CHECK(flux_inbox.count(time_id) == 1);
      const auto& flux_inbox_at_time = flux_inbox.at(time_id);
      CHECK(flux_inbox_at_time.size() == 1);
      CHECK(flux_inbox_at_time.count({direction, self_id}) == 1);
    };
    check_sent_data(west_id, Direction<2>::lower_eta());
    check_sent_data(east_id, Direction<2>::lower_xi());
    check_sent_data(south_id, Direction<2>::lower_eta());
  }

  // Now check ComputeBoundaryFlux
  const auto send_data = [&extents, &runner, &self_id, &time_id, &coordmap](
      const ElementId<2>& id, const Direction<2>& direction,
      const OrientationMap<2>& orientation, const DataVector& normal_dot_fluxes,
      const DataVector& other_data) noexcept {
    const Element<2> element(id, {{direction, {{self_id}, orientation}}});
    auto map = ElementMap<2, Frame::Inertial>(id, coordmap->get_clone());

    db::item_type<normal_dot_fluxes_tag> normal_dot_fluxes_map{};
    normal_dot_fluxes_map[direction].initialize(normal_dot_fluxes.size());
    get(get<Tags::NormalDotFlux<Var>>(normal_dot_fluxes_map[direction])) =
        normal_dot_fluxes;

    db::item_type<other_data_tag> other_data_map{};
    other_data_map[direction].initialize(other_data.size());
    get(get<OtherData>(other_data_map[direction])) = other_data;

    auto box =
        db::create<db::AddSimpleTags<Tags::TimeId, Tags::Extents<2>,
                                     Tags::Element<2>, Tags::ElementMap<2>,
                                     normal_dot_fluxes_tag, other_data_tag>,
                   compute_items>(time_id, extents, element, std::move(map),
                                  std::move(normal_dot_fluxes_map),
                                  std::move(other_data_map));

    runner.apply<component, send_data_for_fluxes>(box, id);
  };

  // The `other_data` (second DataVector argument) for each send is
  // chosen to satisfy the check in NumericalFlux::operator().  For
  // that, we want (with I<> and E<> indicating interior and exterior)
  // E<other_data> = 1 + 2 I<other_data>
  //    + 4 I<normal>_0 - 2 E<normal>_0 + 6 I<normal>_1 - 3 E<normal>_1
  // Note that these are unit normals, and remember the eta map is
  // reversing so the normals point the wrong way with respect to the
  // logical coordinates.
  CHECK_FALSE(
      (runner.is_ready<component, compute_boundary_flux>(sent_box, self_id)));
  // 0 = I<normal>_0 = E<normal>_0, 1 = E<normal>_1 = - I<normal>_1
  // => E<other_data> = 2 I<other_data> - 8
  send_data(south_id, Direction<2>::lower_eta(), {}, {11., 12., 13.},
            {142., 162., 182.});
  CHECK_FALSE(
      (runner.is_ready<component, compute_boundary_flux>(sent_box, self_id)));
  // 0 = I<normal>_1 = E<normal>_1, 1 = I<normal>_0 = - E<normal>_0
  // => E<other_data> = 2 I<other_data> + 7
  send_data(east_id, Direction<2>::lower_xi(), {}, {21., 22., 23.},
            {97., 117., 137.});
  CHECK_FALSE(
      (runner.is_ready<component, compute_boundary_flux>(sent_box, self_id)));
  // 0 = I<normal>_1 = E<normal>_0, 1 = E<normal>_1 = - I<normal>_0
  // => E<other_data> = 2 I<other_data> - 6
  // And the data order is reversed due to the block alignment.
  send_data(west_id, Direction<2>::lower_eta(), block_orientation.inverse_map(),
            {31., 32., 33.}, {64., 44., 24.});
  CHECK((runner.is_ready<component, compute_boundary_flux>(sent_box, self_id)));

  auto received_box = std::get<0>(
      runner.apply<component, compute_boundary_flux>(sent_box, self_id));

  CHECK(tuples::get<fluxes_tag>(runner.inboxes<component>()[self_id]).empty());

  const double element_length_xi =
      0.25 * abs(xi_map(std::array<double, 1>{{1.}})[0] -
                 xi_map(std::array<double, 1>{{-1.}})[0]);
  const double element_length_eta =
      0.5 * abs(eta_map(std::array<double, 1>{{1.}})[0] -
                eta_map(std::array<double, 1>{{-1.}})[0]);

  // Prefactor and weight as in Kopriva 8.42.  Equal to
  // -2/(element length)/w_0  with  w_0 = 2/(N(N-1))  where here N=3.
  const double xi_lift = -6. / element_length_xi;
  const double eta_lift = -6. / element_length_eta;
  // n.(F* - F)
  // For the test we set n.F* = 11 n.F + 1000 n.F_nbr, so
  //      n.(F* - F) = 10 n.F + 1000 n.F_nbr
  const DataVector xi_boundaries{33010., 0., 21040.,
                                 32020., 0., 22050.,
                                 31030., 0., 23060.};
  const DataVector eta_boundaries{0., 0., 0.,
                                  0., 0., 0.,
                                  11070., 12080., 13090.};

  // 3.0 is the time derivative put in at the start.
  CHECK_ITERABLE_APPROX(
      get(get<Tags::dt<Var>>(db::get<dt_variables_tag>(received_box))),
      xi_lift * xi_boundaries + eta_lift * eta_boundaries + 3.0);
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

  db::item_type<dt_variables_tag> dt_variables(extents.product(), 3.0);
  auto start_box = db::create<
      db::AddSimpleTags<Tags::TimeId, Tags::Extents<2>, Tags::Element<2>,
                        Tags::ElementMap<2>, dt_variables_tag,
                        normal_dot_fluxes_tag, other_data_tag>,
      compute_items>(
      time_id, extents, element, std::move(map), std::move(dt_variables),
      db::item_type<normal_dot_fluxes_tag>{}, db::item_type<other_data_tag>{});

  auto sent_box = std::get<0>(
      runner.apply<component, send_data_for_fluxes>(start_box, self_id));

  CHECK((runner.nonempty_inboxes<component, fluxes_tag>().empty()));

  CHECK((runner.is_ready<component, compute_boundary_flux>(sent_box, self_id)));

  const auto received_box = std::get<0>(
      runner.apply<component, compute_boundary_flux>(sent_box, self_id));

  CHECK(db::get<dt_variables_tag>(received_box) ==
        db::item_type<dt_variables_tag>(extents.product(), 3.0));
}
