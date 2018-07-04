// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <tuple>
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
#include "NumericalAlgorithms/DiscontinuousGalerkin/Actions/ComputeNonconservativeBoundaryFluxes.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "tests/Unit/ActionTesting.hpp"

// IWYU pragma: no_forward_declare Tensor
// IWYU pragma: no_forward_declare Variables

namespace {
struct Var : db::SimpleTag {
  static std::string name() noexcept { return "Var"; }
  using type = Scalar<DataVector>;
  static constexpr bool should_be_sliced_to_boundary = false;
};

struct Var2 : db::SimpleTag {
  static std::string name() noexcept { return "Var2"; }
  using type = tnsr::ii<DataVector, 2>;
  static constexpr bool should_be_sliced_to_boundary = false;
};

struct OtherArg : db::SimpleTag {
  static std::string name() noexcept { return "OtherArg"; }
  using type = double;
};

struct System {
  static constexpr const size_t volume_dim = 2;
  using variables_tag = Tags::Variables<tmpl::list<Var, Var2>>;

  template <typename Tag>
  using magnitude_tag = Tags::EuclideanMagnitude<Tag>;

  struct normal_dot_fluxes {
    using argument_tags =
        tmpl::list<Var, OtherArg, Var2,
                   Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<2>>>;
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

struct Metavariables;

using component =
    ActionTesting::MockArrayComponent<Metavariables, domain::ElementIndex<2>,
                                      tmpl::list<>>;

struct Metavariables {
  using system = System;
  using component_list = tmpl::list<component>;
  using const_global_cache_tag_list = tmpl::list<>;
};

template <typename Tag>
using interface_tag =
    domain::Tags::Interface<domain::Tags::InternalDirections<2>, Tag>;

using n_dot_f_tag = interface_tag<Tags::NormalDotFlux<Tags::Variables<
    tmpl::list<Tags::NormalDotFlux<Var>, Tags::NormalDotFlux<Var2>>>>>;

using VarsType = Variables<tmpl::list<Var, Var2>>;
auto run_action(const domain::Element<2>& element,
                const std::unordered_map<domain::Direction<2>, VarsType>& vars,
                const std::unordered_map<domain::Direction<2>, double>&
                    other_arg) noexcept {
  ActionTesting::ActionRunner<Metavariables> runner{{}};

  const domain::Mesh<2> mesh{3, Spectral::Basis::Legendre,
                             Spectral::Quadrature::GaussLobatto};

  const domain::CoordinateMaps::Affine xi_map{-1., 1., 3., 7.};
  const domain::CoordinateMaps::Affine eta_map{-1., 1., -2., 4.};

  auto element_map = domain::ElementMap<2, Frame::Inertial>(
      element.id(),
      domain::make_coordinate_map_base<Frame::Logical, Frame::Inertial>(
          domain::CoordinateMaps::ProductOf2Maps<
              domain::CoordinateMaps::Affine, domain::CoordinateMaps::Affine>(
              xi_map, eta_map)));

  n_dot_f_tag::type n_dot_f_storage{};
  for (const auto& direction_neighbors : element.neighbors()) {
    n_dot_f_storage[direction_neighbors.first].initialize(3);
  }

  auto start_box = db::create<
      db::AddSimpleTags<domain::Tags::Element<2>, domain::Tags::Mesh<2>,
                        domain::Tags::ElementMap<2>,
                        interface_tag<Tags::Variables<tmpl::list<Var, Var2>>>,
                        interface_tag<OtherArg>, n_dot_f_tag>,
      db::AddComputeTags<domain::Tags::InternalDirections<2>,
                         interface_tag<domain::Tags::Direction<2>>,
                         interface_tag<domain::Tags::Mesh<1>>,
                         interface_tag<domain::Tags::UnnormalizedFaceNormal<2>>,
                         interface_tag<Tags::EuclideanMagnitude<
                             domain::Tags::UnnormalizedFaceNormal<2>>>,
                         interface_tag<Tags::Normalized<
                             domain::Tags::UnnormalizedFaceNormal<2>>>>>(
      element, mesh, std::move(element_map), vars, other_arg,
      std::move(n_dot_f_storage));

  return std::get<0>(
      runner
          .apply<component, dg::Actions::ComputeNonconservativeBoundaryFluxes>(
              start_box, element.id()));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DG.Actions.ComputeNonconservativeBoundaryFluxes",
                  "[Unit][NumericalAlgorithms][Actions]") {
  const domain::Element<2> element(
      domain::ElementId<2>(0),
      {{domain::Direction<2>::upper_xi(), {{domain::ElementId<2>(1)}, {}}},
       {domain::Direction<2>::lower_eta(), {{domain::ElementId<2>(1)}, {}}}});

  std::unordered_map<domain::Direction<2>, VarsType> vars;
  vars[domain::Direction<2>::upper_xi()] = VarsType(3);
  get(get<Var>(vars[domain::Direction<2>::upper_xi()])) =
      DataVector{0., 1., 2.};
  {
    auto& var2 = get<Var2>(vars[domain::Direction<2>::upper_xi()]);
    get<0, 0>(var2) = DataVector{3., 4., 5.};
    get<0, 1>(var2) = DataVector{6., 7., 8.};
    get<1, 1>(var2) = DataVector{9., 10., 11.};
  }
  vars[domain::Direction<2>::lower_eta()] =
      -10. * vars[domain::Direction<2>::upper_xi()];

  const std::unordered_map<domain::Direction<2>, double> other_arg{
      {domain::Direction<2>::upper_xi(), 5.},
      {domain::Direction<2>::lower_eta(), 7.}};

  auto box = run_action(element, vars, other_arg);

  const auto& unit_face_normal = db::get<
      interface_tag<Tags::Normalized<domain::Tags::UnnormalizedFaceNormal<2>>>>(
      box);
  const auto& n_dot_f = db::get<n_dot_f_tag>(box);

  std::unordered_map<domain::Direction<2>,
                     Variables<tmpl::list<Tags::NormalDotFlux<Var>,
                                          Tags::NormalDotFlux<Var2>>>>
      expected;
  for (const auto& direction :
       {domain::Direction<2>::upper_xi(), domain::Direction<2>::lower_eta()}) {
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
  const domain::Element<2> element(domain::ElementId<2>(0), {});

  const std::unordered_map<domain::Direction<2>, VarsType> vars{};
  const std::unordered_map<domain::Direction<2>, double> other_arg{};

  auto box = run_action(element, vars, other_arg);

  const auto& n_dot_f = db::get<n_dot_f_tag>(box);

  CHECK(n_dot_f.empty());
}
