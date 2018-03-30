// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <tuple>
#include <unordered_map>
#include <utility>

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
#include "Domain/ElementIndex.hpp"  // IWYU pragma: keep
#include "Domain/ElementMap.hpp"
#include "Domain/FaceNormal.hpp"
#include "Domain/Neighbors.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
struct Var : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Var";
  using type = Scalar<DataVector>;
  static constexpr bool should_be_sliced_to_boundary = false;
};

struct Var2 : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Var2";
  using type = tnsr::ii<DataVector, 2>;
  static constexpr bool should_be_sliced_to_boundary = false;
};

struct OtherArg : db::DataBoxTag {
  static constexpr db::DataBoxString label = "OtherArg";
  using type = double;
};

struct System {
  static constexpr const size_t volume_dim = 2;
  using variables_tag = Tags::Variables<tmpl::list<Var, Var2>>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;

  struct normal_dot_fluxes {
    using argument_tags = tmpl::list<Var, OtherArg, Var2>;
    static void apply(
        const gsl::not_null<Scalar<DataVector>*> var_normal_dot_flux,
        const gsl::not_null<tnsr::ii<DataVector, 2>*> var2_normal_dot_flux,
        const Scalar<DataVector>& var,
        const double other_arg,
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

struct Metavariables;

using component =
    ActionTesting::MockArrayComponent<Metavariables, ElementIndex<2>,
                                      tmpl::list<>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
};

template <typename Tag>
using interface_tag = Tags::Interface<Tags::InternalDirections<2>, Tag>;

using VarsType = Variables<tmpl::list<Var, Var2>>;
auto run_action(
    const Element<2>& element,
    const std::unordered_map<Direction<2>, VarsType>& vars,
    const std::unordered_map<Direction<2>, double>& other_arg) noexcept {
  ActionTesting::ActionRunner<Metavariables> runner{{}};

  const Index<2> extents{{{3, 3}}};

  const CoordinateMaps::Affine xi_map{-1., 1., 3., 7.};
  const CoordinateMaps::Affine eta_map{-1., 1., -2., 4.};

  auto element_map = ElementMap<2, Frame::Inertial>(
      element.id(), make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
                        CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                                       CoordinateMaps::Affine>(
                            xi_map, eta_map)));

  auto start_box = db::create<
      db::AddSimpleTags<Tags::Element<2>, Tags::Extents<2>, Tags::ElementMap<2>,
                        interface_tag<Tags::Variables<tmpl::list<Var, Var2>>>,
                        interface_tag<OtherArg>>,
      db::AddComputeTags<
          Tags::InternalDirections<2>, interface_tag<Tags::Direction<2>>,
          interface_tag<Tags::Extents<1>>,
          interface_tag<Tags::UnnormalizedFaceNormal<2>>,
          interface_tag<
              Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>,
          interface_tag<Tags::Normalized<
              Tags::UnnormalizedFaceNormal<2>,
              Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>>>>(
      element, extents, std::move(element_map), vars, other_arg);

  return std::get<0>(
      runner
          .apply<component, dg::Actions::ComputeNonconservativeBoundaryFluxes>(
              start_box, element.id()));
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
      Tags::UnnormalizedFaceNormal<2>,
      Tags::EuclideanMagnitude<Tags::UnnormalizedFaceNormal<2>>>>>(box);
  const auto& n_dot_f =
      db::get<interface_tag<Tags::NormalDotFlux<Tags::Variables<
          tmpl::list<Tags::NormalDotFlux<Var>, Tags::NormalDotFlux<Var2>>>>>>(
          box);

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

  const auto& n_dot_f =
      db::get<interface_tag<Tags::NormalDotFlux<Tags::Variables<
          tmpl::list<Tags::NormalDotFlux<Var>, Tags::NormalDotFlux<Var2>>>>>>(
          box);

  CHECK(n_dot_f.empty());
}
