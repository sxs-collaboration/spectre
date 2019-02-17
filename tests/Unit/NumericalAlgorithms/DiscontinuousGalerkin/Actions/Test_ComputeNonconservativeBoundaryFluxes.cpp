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
#include "DataStructures/DataBox/Prefixes.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/EagerMath/Magnitude.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/ElementIndex.hpp"  // IWYU pragma: keep
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"  // IWYU pragma: keep
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare db::DataBox
// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

// IWYU pragma: no_include <boost/variant/get.hpp>

namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  static std::string name() noexcept { return "Var2"; }
  using type = tnsr::ii<DataVector, 2>;
};

struct OtherArg : db::SimpleTag {
  static std::string name() noexcept { return "OtherArg"; }
  using type = double;
};

struct System {
  using variables_tag = Tags::Variables<tmpl::list<Var, Var2>>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;

  struct normal_dot_fluxes {
    using argument_tags =
        tmpl::list<Var, OtherArg, Var2,
                   Tags::Normalized<Tags::UnnormalizedFaceNormal<2>>>;
    static void apply(
        const gsl::not_null<Scalar<DataVector>*> var_normal_dot_flux,
        const gsl::not_null<tnsr::ii<DataVector, 2>*> var2_normal_dot_flux,
        const Scalar<DataVector>& var, const double other_arg,
        const tnsr::ii<DataVector, 2>& var2,
        const tnsr::i<DataVector, 2>& unit_face_normal) noexcept {
      get(*var_normal_dot_flux) =
          get(var) + other_arg * get<0>(unit_face_normal);
      for (size_t i = 0; i < 2; ++i) {
        for (size_t j = i; j < 2; ++j) {
          var2_normal_dot_flux->get(i, j) =
              var2.get(i, j) + other_arg * get<1>(unit_face_normal);
        }
      }
    }
  };
};

template <typename Tag>
using interface_tag = Tags::Interface<Tags::InternalDirections<2>, Tag>;
template <typename Tag>
using interface_compute_tag =
    Tags::InterfaceComputeItem<Tags::InternalDirections<2>, Tag>;

using n_dot_f_tag = interface_tag<Tags::NormalDotFlux<Tags::Variables<
    tmpl::list<Tags::NormalDotFlux<Var>, Tags::NormalDotFlux<Var2>>>>>;

using VarsType = Variables<tmpl::list<Var, Var2>>;

struct Metavariables;

struct component {
  using metavariables = Metavariables;
  using chare_type = ActionTesting::MockArrayChare;
  using array_index = ElementIndex<2>;
  using const_global_cache_tag_list = tmpl::list<>;
  using action_list =
      tmpl::list<dg::Actions::ComputeNonconservativeBoundaryFluxes<
          Tags::InternalDirections<2>>>;
  using simple_tags =
      db::AddSimpleTags<Tags::Element<2>, Tags::Mesh<2>, Tags::ElementMap<2>,
                        interface_tag<Tags::Variables<tmpl::list<Var, Var2>>>,
                        interface_tag<OtherArg>, n_dot_f_tag>;
  using compute_tags = db::AddComputeTags<
      Tags::InternalDirections<2>, interface_compute_tag<Tags::Direction<2>>,
      interface_compute_tag<Tags::InterfaceMesh<2>>,
      interface_compute_tag<Tags::UnnormalizedFaceNormal<2>>,
      interface_compute_tag<
          Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>,
      interface_compute_tag<Tags::Normalized<Tags::UnnormalizedFaceNormal<2>>>>;
  using initial_databox =
      db::compute_databox_type<tmpl::append<simple_tags, compute_tags>>;
};

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};

auto run_action(
    const Element<2>& element,
    const std::unordered_map<Direction<2>, VarsType>& vars,
    const std::unordered_map<Direction<2>, double>& other_arg) noexcept {
  const Mesh<2> mesh{3, Spectral::Basis::Legendre,
                     Spectral::Quadrature::GaussLobatto};

  using Affine = domain::CoordinateMaps::Affine;
  const Affine xi_map{-1., 1., 3., 7.};
  const Affine eta_map{-1., 1., -2., 4.};

  auto element_map = ElementMap<2, Frame::Inertial>(
      element.id(),
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::ProductOf2Maps<Affine, Affine>(xi_map,
                                                                 eta_map)));

  n_dot_f_tag::type n_dot_f_storage{};
  for (const auto& direction_neighbors : element.neighbors()) {
    n_dot_f_storage[direction_neighbors.first].initialize(3);
  }

  using simple_tags = typename component::simple_tags;
  using compute_tags = typename component::compute_tags;

  using MockRuntimeSystem = ActionTesting::MockRuntimeSystem<Metavariables>;
  using MockDistributedObjectsTag =
      MockRuntimeSystem::MockDistributedObjectsTag<component>;
  MockRuntimeSystem::TupleOfMockDistributedObjects dist_objects{};
  tuples::get<MockDistributedObjectsTag>(dist_objects)
      .emplace(ElementIndex<2>{element.id()},
               ActionTesting::MockDistributedObject<component>{
                   db::create<simple_tags, compute_tags>(
                       element, mesh, std::move(element_map), vars, other_arg,
                       std::move(n_dot_f_storage))});
  MockRuntimeSystem runner{{}, std::move(dist_objects)};
  runner.next_action<component>(element.id());
  // std::move call on returned value is intentional.
  return std::move(runner.algorithms<component>()
                       .at(element.id())
                       .get_databox<db::compute_databox_type<
                           tmpl::append<simple_tags, compute_tags>>>());
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Actions.ComputeNonconservativeBoundaryFluxes",
                  "[Unit][NumericalAlgorithms][Actions]") {
  const Element<2> element(
      ElementId<2>(0), {{Direction<2>::upper_xi(), {{ElementId<2>(1)}, {}}},
                        {Direction<2>::lower_eta(), {{ElementId<2>(1)}, {}}}});

  std::unordered_map<Direction<2>, VarsType> vars;
  vars[Direction<2>::upper_xi()] = VarsType(3);
  get(get<Var>(vars[Direction<2>::upper_xi()])) = DataVector{0., 1., 2.};
  {
    auto& var2 = get<Var2>(vars[Direction<2>::upper_xi()]);
    get<0, 0>(var2) = DataVector{3., 4., 5.};
    get<0, 1>(var2) = DataVector{6., 7., 8.};
    get<1, 1>(var2) = DataVector{9., 10., 11.};
  }
  vars[Direction<2>::lower_eta()] = -10. * vars[Direction<2>::upper_xi()];

  const std::unordered_map<Direction<2>, double> other_arg{
      {Direction<2>::upper_xi(), 5.}, {Direction<2>::lower_eta(), 7.}};

  auto box = run_action(element, vars, other_arg);

  const auto& unit_face_normal = db::get<interface_tag<Tags::Normalized<
      Tags::UnnormalizedFaceNormal<2>>>>(box);
  const auto& n_dot_f = db::get<n_dot_f_tag>(box);

  std::unordered_map<Direction<2>,
                     Variables<tmpl::list<Tags::NormalDotFlux<Var>,
                                          Tags::NormalDotFlux<Var2>>>>
      expected;
  for (const auto& direction :
       {Direction<2>::upper_xi(), Direction<2>::lower_eta()}) {
    expected[direction].initialize(3);
    System::normal_dot_fluxes::apply(
        &get<Tags::NormalDotFlux<Var>>(expected[direction]),
        &get<Tags::NormalDotFlux<Var2>>(expected[direction]),
        get<Var>(vars.at(direction)), other_arg.at(direction),
        get<Var2>(vars.at(direction)), unit_face_normal.at(direction));
  }

  CHECK(n_dot_f == expected);
}

SPECTRE_TEST_CASE(
    "Unit.DG.Actions.ComputeNonconservativeBoundaryFluxes.NoNeighbors",
    "[Unit][NumericalAlgorithms][Actions]") {
  const Element<2> element(ElementId<2>(0), {});

  const std::unordered_map<Direction<2>, VarsType> vars{};
  const std::unordered_map<Direction<2>, double> other_arg{};

  auto box = run_action(element, vars, other_arg);

  const auto& n_dot_f = db::get<n_dot_f_tag>(box);

  CHECK(n_dot_f.empty());
}
