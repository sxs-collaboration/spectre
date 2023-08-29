// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <functional>
#include <memory>
#include <pup.h>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesTag.hpp"
#include "Domain/CoordinateMaps/Affine.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.hpp"
#include "Domain/CoordinateMaps/CoordinateMap.tpp"
#include "Domain/CoordinateMaps/ProductMaps.hpp"
#include "Domain/CoordinateMaps/ProductMaps.tpp"
#include "Domain/CoordinateMaps/Rotation.hpp"
#include "Domain/CoordinateMaps/TimeDependent/CubicScale.hpp"
#include "Domain/Creators/Tags/FunctionsOfTime.hpp"
#include "Domain/ElementMap.hpp"
#include "Domain/FunctionsOfTime/FunctionOfTime.hpp"
#include "Domain/FunctionsOfTime/PiecewisePolynomial.hpp"
#include "Domain/InterfaceComputeTags.hpp"
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/DataBox/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/SpatialDiscretization/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Time/Tags/Time.hpp"
#include "Utilities/CloneUniquePtrs.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

namespace domain {
namespace {
template <size_t N>
struct NoCopy {
  NoCopy() = default;
  NoCopy(const NoCopy&) = delete;
  NoCopy(NoCopy&&) = default;
  NoCopy& operator=(const NoCopy&) = delete;
  NoCopy& operator=(NoCopy&&) = default;
  ~NoCopy() = default;
};

namespace TestTags {
struct BaseInt : db::BaseTag {};

struct Int : db::SimpleTag, BaseInt {
  using type = int;
};

struct Double : db::SimpleTag {
  using type = double;
};

template <size_t N>
struct NoCopy : db::SimpleTag {
  using type = domain::NoCopy<N>;
};

template <typename Tag>
struct Negate : db::PrefixTag, db::SimpleTag {
  using tag = Tag;
  using type = typename Tag::type;
};

template <typename Tag>
struct NegateCompute : Negate<Tag>, db::ComputeTag {
  using base = Negate<Tag>;
  static constexpr void function(
      const gsl::not_null<typename Tag::type*> result,
      const typename Tag::type& x) {
    *result = -x;
  }
  using argument_tags = tmpl::list<Tag>;
  using return_type = typename Tag::type;
};

struct NegateDoubleAddInt : db::SimpleTag {
  using type = double;
};

struct NegateDoubleAddIntCompute : NegateDoubleAddInt, db::ComputeTag {
  using base = NegateDoubleAddInt;
  using return_type = double;
  static constexpr void function(const gsl::not_null<double*> result,
                                 const double x, const int y) {
    *result = -x + y;
  }
  using argument_tags = tmpl::list<Double, BaseInt>;
  using volume_tags = tmpl::list<BaseInt>;
};

struct IntCompute : db::ComputeTag, Int {
  static constexpr void function(const gsl::not_null<int*> result,
                                 const int x) {
    *result = x + 3;
  }
  using argument_tags = tmpl::list<Int>;
  using volume_tags = tmpl::list<Int>;
  using base = Int;
  using return_type = int;
};

template <size_t VolumeDim>
struct ComplexItem : db::SimpleTag {
  using type = std::pair<int, double>;
};

template <size_t VolumeDim>
struct ComplexItemCompute : ComplexItem<VolumeDim>, db::ComputeTag {
  static constexpr void function(
      const gsl::not_null<std::pair<int, double>*> result, const int i,
      const double d, const domain::NoCopy<1>& /*unused*/,
      const domain::NoCopy<2>& /*unused*/) {
    *result = std::make_pair(i, d);
  }
  using argument_tags = tmpl::list<Int, Double, NoCopy<1>, NoCopy<2>>;
  using volume_tags = tmpl::list<Int, NoCopy<1>>;
  using base = ComplexItem<VolumeDim>;
  using return_type = std::pair<int, double>;
};

template <typename>
struct TemplatedDirections : db::SimpleTag {
  static constexpr size_t volume_dim = 3;
  using type = std::unordered_set<Direction<3>>;
};
}  // namespace TestTags

void test_interface_items() {
  constexpr size_t dim = 3;
  using internal_directions = Tags::InternalDirections<dim>;
  using boundary_directions_interior = Tags::BoundaryDirectionsInterior<dim>;
  using templated_directions = TestTags::TemplatedDirections<int>;

  Element<dim> element{ElementId<3>(0),
                       {{Direction<dim>::lower_xi(), {}},
                        {Direction<dim>::upper_xi(), {}},
                        {Direction<dim>::upper_zeta(), {}}}};

  std::unordered_map<Direction<dim>, NoCopy<2>> internal_nocopy_map_item;
  internal_nocopy_map_item.emplace(Direction<dim>::lower_xi(), NoCopy<2>{});
  internal_nocopy_map_item.emplace(Direction<dim>::upper_xi(), NoCopy<2>{});
  internal_nocopy_map_item.emplace(Direction<dim>::upper_zeta(), NoCopy<2>{});
  std::unordered_map<Direction<dim>, NoCopy<2>> external_nocopy_map_item;
  external_nocopy_map_item.emplace(Direction<dim>::lower_eta(), NoCopy<2>{});
  external_nocopy_map_item.emplace(Direction<dim>::upper_eta(), NoCopy<2>{});
  external_nocopy_map_item.emplace(Direction<dim>::lower_zeta(), NoCopy<2>{});
  const auto box = db::create<
      db::AddSimpleTags<
          Tags::Element<dim>, TestTags::Int,
          Tags::Interface<internal_directions, TestTags::Double>,
          Tags::Interface<boundary_directions_interior, TestTags::Double>,
          TestTags::NoCopy<1>,
          Tags::Interface<internal_directions, TestTags::NoCopy<2>>,
          Tags::Interface<boundary_directions_interior, TestTags::NoCopy<2>>,
          templated_directions,
          Tags::Interface<templated_directions, TestTags::Double>>,
      db::AddComputeTags<
          Tags::InternalDirectionsCompute<dim>,
          Tags::InterfaceCompute<internal_directions, Tags::Direction<dim>>,
          TestTags::NegateCompute<TestTags::Int>,
          Tags::InterfaceCompute<internal_directions, TestTags::IntCompute>,
          Tags::InterfaceCompute<internal_directions,
                                 TestTags::NegateCompute<TestTags::Double>>,
          Tags::InterfaceCompute<internal_directions,
                                 TestTags::NegateDoubleAddIntCompute>,
          Tags::InterfaceCompute<internal_directions,
                                 TestTags::ComplexItemCompute<dim>>,
          Tags::InterfaceCompute<templated_directions, Tags::Direction<dim>>,
          Tags::InterfaceCompute<templated_directions,
                                 TestTags::NegateCompute<TestTags::Double>>,
          Tags::InterfaceCompute<templated_directions,
                                 TestTags::NegateDoubleAddIntCompute>,
          Tags::BoundaryDirectionsInteriorCompute<dim>,
          Tags::InterfaceCompute<boundary_directions_interior,
                                 Tags::Direction<dim>>,
          Tags::InterfaceCompute<boundary_directions_interior,
                                 TestTags::IntCompute>,
          Tags::InterfaceCompute<boundary_directions_interior,
                                 TestTags::NegateCompute<TestTags::Double>>,
          Tags::InterfaceCompute<boundary_directions_interior,
                                 TestTags::NegateDoubleAddIntCompute>,
          Tags::InterfaceCompute<boundary_directions_interior,
                                 TestTags::ComplexItemCompute<dim>>,
          Tags::BoundaryDirectionsExteriorCompute<dim>>>(
      std::move(element), 5,
      std::unordered_map<Direction<dim>, double>{
          {Direction<dim>::lower_xi(), 1.5},
          {Direction<dim>::upper_xi(), 2.5},
          {Direction<dim>::upper_zeta(), 3.5}},
      std::unordered_map<Direction<dim>, double>{
          {Direction<dim>::lower_eta(), 10.5},
          {Direction<dim>::upper_eta(), 20.5},
          {Direction<dim>::lower_zeta(), 30.5}},
      NoCopy<1>{}, std::move(internal_nocopy_map_item),
      std::move(external_nocopy_map_item),
      std::unordered_set<Direction<dim>>{Direction<dim>::upper_xi()},
      std::unordered_map<Direction<dim>, double>{
          {Direction<dim>::upper_xi(), 4.5}});

  CHECK(get<Tags::BoundaryDirectionsInterior<dim>>(box) ==
        std::unordered_set<Direction<dim>>{Direction<dim>::lower_eta(),
                                           Direction<dim>::upper_eta(),
                                           Direction<dim>::lower_zeta()});

  CHECK(
      (get<Tags::Interface<internal_directions, Tags::Direction<dim>>>(box)) ==
      (std::unordered_map<Direction<dim>, Direction<dim>>{
          {Direction<dim>::lower_xi(), Direction<dim>::lower_xi()},
          {Direction<dim>::upper_xi(), Direction<dim>::upper_xi()},
          {Direction<dim>::upper_zeta(), Direction<dim>::upper_zeta()}}));

  CHECK(get<Tags::BoundaryDirectionsInterior<dim>>(box) ==
        get<Tags::BoundaryDirectionsExterior<dim>>(box));

  CHECK(get<TestTags::Negate<TestTags::Int>>(box) == -5);

  CHECK((get<Tags::Interface<templated_directions, TestTags::Double>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::upper_xi(), 4.5}}));

  CHECK((get<Tags::Interface<internal_directions,
                             TestTags::Negate<TestTags::Double>>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::lower_xi(), -1.5},
            {Direction<dim>::upper_xi(), -2.5},
            {Direction<dim>::upper_zeta(), -3.5}}));

  CHECK(
      (get<Tags::Interface<internal_directions, TestTags::NegateDoubleAddInt>>(
          box)) == (std::unordered_map<Direction<dim>, double>{
                       {Direction<dim>::lower_xi(), -1.5 + 5.0},
                       {Direction<dim>::upper_xi(), -2.5 + 5.0},
                       {Direction<dim>::upper_zeta(), -3.5 + 5.0}}));

  CHECK((get<Tags::Interface<boundary_directions_interior,
                             TestTags::Negate<TestTags::Double>>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::lower_eta(), -10.5},
            {Direction<dim>::upper_eta(), -20.5},
            {Direction<dim>::lower_zeta(), -30.5}}));

  CHECK((get<Tags::Interface<boundary_directions_interior,
                             TestTags::NegateDoubleAddInt>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::lower_eta(), -10.5 + 5.0},
            {Direction<dim>::upper_eta(), -20.5 + 5.0},
            {Direction<dim>::lower_zeta(), -30.5 + 5.0}}));

  CHECK((get<Tags::Interface<internal_directions, TestTags::Int>>(box)) ==
        (std::unordered_map<Direction<dim>, int>{
            {Direction<dim>::lower_xi(), 8},
            {Direction<dim>::upper_xi(), 8},
            {Direction<dim>::upper_zeta(), 8}}));

  CHECK((get<Tags::Interface<boundary_directions_interior, TestTags::Int>>(
            box)) == (std::unordered_map<Direction<dim>, int>{
                         {Direction<dim>::lower_eta(), 8},
                         {Direction<dim>::upper_eta(), 8},
                         {Direction<dim>::lower_zeta(), 8}}));

  CHECK((get<Tags::Interface<internal_directions, TestTags::ComplexItem<dim>>>(
            box)) ==
        (std::unordered_map<Direction<dim>, std::pair<int, double>>{
            {Direction<dim>::lower_xi(), {5, 1.5}},
            {Direction<dim>::upper_xi(), {5, 2.5}},
            {Direction<dim>::upper_zeta(), {5, 3.5}}}));

  CHECK((get<Tags::InterfaceCompute<boundary_directions_interior,
                                    TestTags::ComplexItemCompute<dim>>>(box)) ==
        (std::unordered_map<Direction<dim>, std::pair<int, double>>{
            {Direction<dim>::lower_eta(), {5, 10.5}},
            {Direction<dim>::upper_eta(), {5, 20.5}},
            {Direction<dim>::lower_zeta(), {5, 30.5}}}));

  CHECK((get<Tags::Interface<templated_directions,
                             TestTags::Negate<TestTags::Double>>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::upper_xi(), -4.5}}));

  CHECK(
      (get<Tags::Interface<templated_directions, TestTags::NegateDoubleAddInt>>(
          box)) == (std::unordered_map<Direction<dim>, double>{
                       {Direction<dim>::upper_xi(), -4.5 + 5.0}}));
}

constexpr size_t dim = 2;

struct Dirs : db::SimpleTag {
  static constexpr size_t volume_dim = dim;
  using type = std::unordered_set<Direction<dim>>;
};

struct DirsCompute : Dirs, db::ComputeTag {
  static void function(
      const gsl::not_null<std::unordered_set<Direction<dim>>*> result) {
    *result = std::unordered_set<Direction<dim>>{Direction<dim>::lower_xi(),
                                                 Direction<dim>::upper_eta()};
  }
  using argument_tags = tmpl::list<>;
  using base = Dirs;
  using return_type = std::unordered_set<Direction<dim>>;
};

template <size_t N>
struct Var : db::SimpleTag {
  using type = Scalar<DataVector>;
  static constexpr bool should_be_sliced_to_boundary =
      N == 3 or N == 30 or  // sliced_simple_item_tag below
      N == 2 or N == 20;    // sliced_compute_item_tag below
};

template <size_t N>
struct VarPlusFive : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t N>
struct VarPlusFiveCompute : VarPlusFive<N>, db::ComputeTag {
  static void function(const gsl::not_null<Scalar<DataVector>*> result,
                       const Scalar<DataVector>& var) {
    *result = Scalar<DataVector>{get(var) + 5.0};
  }
  using argument_tags = tmpl::list<Var<N>>;
  using base = VarPlusFive<N>;
  using return_type = Scalar<DataVector>;
};

template <size_t VolumeDim>
struct Compute : db::SimpleTag {
  using type = Variables<tmpl::list<Var<VolumeDim>, Var<10 * VolumeDim>>>;
};

template <size_t VolumeDim>
struct ComputeCompute : Compute<VolumeDim>, db::ComputeTag {
  using base = Compute<VolumeDim>;
  using return_type =
      Variables<tmpl::list<Var<VolumeDim>, Var<10 * VolumeDim>>>;
  static void function(const gsl::not_null<return_type*> result,
                       const Mesh<VolumeDim>& mesh) {
    *result = Variables<tmpl::list<Var<VolumeDim>, Var<10 * VolumeDim>>>(
        mesh.number_of_grid_points(), VolumeDim);
    get(get<Var<10 * VolumeDim>>(*result)) *= 5.0;
  }
  using argument_tags = tmpl::list<Tags::Mesh<VolumeDim>>;
};

template <size_t N>
auto make_interface_variables(const DataVector& value_xi,
                              const DataVector& value_eta) {
  const auto make = [](const DataVector& value) {
    Variables<tmpl::list<Var<N>>> v(value.size());
    get(get<Var<N>>(v)) = value;
    return v;
  };
  std::unordered_map<Direction<dim>, Variables<tmpl::list<Var<N>>>> ret;
  ret.emplace(Direction<dim>::lower_xi(), make(value_xi));
  ret.emplace(Direction<dim>::upper_eta(), make(value_eta));
  return ret;
}

template <size_t N0, size_t N1>
auto make_interface_variables(const DataVector& value_xi0,
                              const DataVector& value_xi1,
                              const DataVector& value_eta0,
                              const DataVector& value_eta1) {
  const auto make = [](const DataVector& value0, const DataVector& value1) {
    Variables<tmpl::list<Var<N0>, Var<N1>>> v(value0.size());
    get(get<Var<N0>>(v)) = value0;
    get(get<Var<N1>>(v)) = value1;
    return v;
  };
  std::unordered_map<Direction<dim>, Variables<tmpl::list<Var<N0>, Var<N1>>>>
      ret;
  ret.emplace(Direction<dim>::lower_xi(), make(value_xi0, value_xi1));
  ret.emplace(Direction<dim>::upper_eta(), make(value_eta0, value_eta1));
  return ret;
}

auto make_interface_tensor(DataVector value_xi, DataVector value_eta) {
  std::unordered_map<Direction<dim>, Scalar<DataVector>> ret;
  ret.emplace(Direction<dim>::lower_xi(),
              Scalar<DataVector>(std::move(value_xi)));
  ret.emplace(Direction<dim>::upper_eta(),
              Scalar<DataVector>(std::move(value_eta)));
  return ret;
}

void test_interface_subitems() {
  const Mesh<dim> mesh{
      {{4, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

  const DataVector boundary_vars_xi{10., 11., 12.};
  const DataVector boundary_vars_eta{20., 21., 22., 23.};

  const auto volume_var = [&mesh]() {
    Variables<tmpl::list<Var<2>>> volume_data(mesh.number_of_grid_points());
    for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
      get(get<Var<2>>(volume_data))[i] = i;
    }
    return volume_data;
  }();

  const auto volume_tensor = [&mesh]() {
    DataVector result(mesh.number_of_grid_points());
    for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
      result[i] = i;
    }
    return Scalar<DataVector>{result};
  }();

  auto box = db::create<
      db::AddSimpleTags<
          Tags::Mesh<dim>, ::Tags::Variables<tmpl::list<Var<2>>>, Var<3>,
          Tags::Interface<Dirs, ::Tags::Variables<tmpl::list<Var<0>>>>>,
      db::AddComputeTags<
          DirsCompute, VarPlusFiveCompute<3>,
          Tags::InterfaceCompute<Dirs, Tags::Direction<dim>>,
          Tags::InterfaceCompute<Dirs, Tags::InterfaceMesh<dim>>,
          Tags::InterfaceCompute<Dirs, ComputeCompute<1>>,
          Tags::Slice<Dirs, ::Tags::Variables<tmpl::list<Var<2>>>>,
          Tags::Slice<Dirs, Var<3>>, Tags::Slice<Dirs, VarPlusFive<3>>>>(
      mesh, volume_var, volume_tensor,
      make_interface_variables<0>(boundary_vars_xi, boundary_vars_eta));

  CHECK((db::get<Tags::Interface<Dirs, Var<0>>>(box)) ==
        make_interface_tensor(boundary_vars_xi, boundary_vars_eta));
  CHECK((db::get<Tags::Interface<Dirs, Var<1>>>(box)) ==
        make_interface_tensor({1., 1., 1.}, {1., 1., 1., 1.}));
  CHECK((db::get<Tags::Interface<Dirs, Var<10>>>(box)) ==
        make_interface_tensor({5., 5., 5.}, {5., 5., 5., 5.}));
  CHECK((db::get<Tags::Interface<Dirs, Var<2>>>(box)) ==
        make_interface_tensor({0., 4., 8.}, {8., 9., 10., 11.}));
  CHECK((db::get<Tags::Interface<Dirs, Var<3>>>(box)) ==
        make_interface_tensor({0., 4., 8.}, {8., 9., 10., 11.}));
  CHECK((db::get<Tags::Interface<Dirs, VarPlusFive<3>>>(box)) ==
        make_interface_tensor({5., 9., 13.}, {13., 14., 15., 16.}));

  TestHelpers::db::test_compute_tag<::Tags::Subitem<
      Tags::Interface<Dirs, Var<2>>,
      Tags::Slice<Dirs, ::Tags::Variables<tmpl::list<Var<2>>>>>>(
      "Interface<Dirs, Var>");

  TestHelpers::db::test_compute_tag<
      ::Tags::Subitem<Tags::Interface<Dirs, Var<1>>,
                      Tags::InterfaceCompute<Dirs, ComputeCompute<1>>>>(
      "Interface<Dirs, Var>");

  db::mutate<Tags::Interface<Dirs, Var<0>>>(
      [](const auto boundary_tensor) {
        get(boundary_tensor->at(Direction<dim>::lower_xi())) *= 3.;
      },
      make_not_null(&box));
  CHECK((db::get<Tags::Interface<Dirs, Var<0>>>(box)) ==
        make_interface_tensor(3. * boundary_vars_xi, boundary_vars_eta));
}

using simple_item_tag = ::Tags::Variables<tmpl::list<Var<0>>>;
using compute_item_tag = ComputeCompute<1>;
using sliced_simple_item_tag = ::Tags::Variables<tmpl::list<Var<3>>>;

void test_interface_slice(){
  const Mesh<dim> mesh{
      {{4, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};

  auto volume_tensor = [&mesh]() {
    DataVector result(mesh.number_of_grid_points());
    for (size_t i = 0; i < mesh.number_of_grid_points(); ++i) {
      result[i] = 2.0 * static_cast<double>(i);
    }
    return Scalar<DataVector>{result};
  }();
  typename sliced_simple_item_tag::type volume_vars(
      mesh.number_of_grid_points());
  get<Var<3>>(volume_vars).get() =
      DataVector{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
  const DataVector boundary_vars_xi{10., 11., 12.};
  const DataVector boundary_vars_eta{20., 21., 22., 23.};

  ElementMap<2, Frame::Inertial> element_map(
      ElementId<2>(0),
      domain::make_coordinate_map_base<Frame::BlockLogical, Frame::Inertial>(
          domain::CoordinateMaps::Rotation<2>(atan2(4., 3.))));

  const std::unordered_map<Direction<dim>, Mesh<dim - 1>>
      expected_interface_mesh{
          {Direction<dim>::lower_xi(),
           Mesh<dim - 1>{3, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto}},
          {Direction<dim>::upper_eta(),
           Mesh<dim - 1>{4, Spectral::Basis::Legendre,
                         Spectral::Quadrature::GaussLobatto}}};

  const auto expected_boundary_coords = [&expected_interface_mesh,
                                         &element_map]() {
    std::unordered_map<Direction<dim>, tnsr::I<DataVector, dim>> coords{};
    typename DirsCompute::return_type directions;
    DirsCompute::function(make_not_null(&directions));
    for (const auto& direction : directions) {
      coords[direction] = element_map(interface_logical_coordinates(
          expected_interface_mesh.at(direction), direction));
    }
    return coords;
  }();

  auto box = db::create<
      db::AddSimpleTags<Tags::Mesh<dim>, Tags::ElementMap<dim>,
                        sliced_simple_item_tag, Var<4>,
                        Tags::Interface<Dirs, simple_item_tag>>,
      db::AddComputeTags<
          DirsCompute, ComputeCompute<2>, VarPlusFiveCompute<4>,
          Tags::InterfaceCompute<Dirs, Tags::Direction<dim>>,
          Tags::InterfaceCompute<Dirs, Tags::InterfaceMesh<dim>>,
          Tags::InterfaceCompute<Dirs, compute_item_tag>,
          Tags::InterfaceCompute<Dirs, Tags::BoundaryCoordinates<dim>>,
          Tags::Slice<Dirs, Compute<2>>,
          Tags::Slice<Dirs, sliced_simple_item_tag>, Tags::Slice<Dirs, Var<4>>,
          Tags::Slice<Dirs, VarPlusFive<4>>>>(
      mesh, std::move(element_map), std::move(volume_vars),
      std::move(volume_tensor),
      make_interface_variables<0>(boundary_vars_xi, boundary_vars_eta));

  TestHelpers::db::test_simple_tag<Tags::Interface<Dirs, simple_item_tag>>(
      "Interface<Dirs, Variables(Var)>");
  TestHelpers::db::test_compute_tag<
      Tags::InterfaceCompute<Dirs, Tags::InterfaceMesh<dim>>>(
      "Interface<Dirs, Mesh>");
  TestHelpers::db::test_compute_tag<
      Tags::InterfaceCompute<Dirs, Tags::BoundaryCoordinates<dim>>>(
      "Interface<Dirs, InertialCoordinates>");
  TestHelpers::db::test_compute_tag<Tags::Slice<Dirs, Var<4>>>(
      "Interface<Dirs, Var>");
  TestHelpers::db::test_compute_tag<
      Tags::InterfaceCompute<Dirs, Tags::Direction<dim>>>(
      "Interface<Dirs, Direction>");
  TestHelpers::db::test_compute_tag<Tags::InterfaceMesh<dim>>("Mesh");
  TestHelpers::db::test_compute_tag<Tags::BoundaryCoordinates<dim>>(
      "InertialCoordinates");

  CHECK((db::get<Tags::Interface<Dirs, simple_item_tag>>(box)) ==
        make_interface_variables<0>(boundary_vars_xi, boundary_vars_eta));
  CHECK((db::get<Tags::Interface<Dirs, Tags::Mesh<dim - 1>>>(box)) ==
        expected_interface_mesh);
  CHECK(db::get<Tags::Interface<Dirs, Tags::Coordinates<dim, Frame::Inertial>>>(
            box) == expected_boundary_coords);
  CHECK((db::get<Tags::InterfaceCompute<Dirs, compute_item_tag>>(box)) ==
        (make_interface_variables<1, 10>({1., 1., 1.}, {5., 5., 5.},
                                         {1., 1., 1., 1.}, {5., 5., 5., 5.})));
  CHECK((db::get<Tags::Interface<Dirs, Compute<2>>>(box)) ==
        (make_interface_variables<2, 20>({2., 2., 2.}, {10., 10., 10.},
                                         {2., 2., 2., 2.},
                                         {10., 10., 10., 10.})));
  CHECK((db::get<Tags::Interface<Dirs, sliced_simple_item_tag>>(box)) ==
        make_interface_variables<3>({0., 4., 8.}, {8., 9., 10., 11.}));
  CHECK((db::get<Tags::Interface<Dirs, Var<4>>>(box)) ==
        make_interface_tensor({0., 8., 16.}, {16., 18., 20., 22.}));
  CHECK((db::get<Tags::Interface<Dirs, VarPlusFive<4>>>(box)) ==
        make_interface_tensor({5., 13., 21.}, {21., 23., 25., 27.}));
}

template <size_t Dim>
struct Directions : db::SimpleTag {
  static constexpr size_t volume_dim = Dim;
  static std::string name() { return "Directions"; }
  using type = std::unordered_set<Direction<Dim>>;
};

template <size_t Dim>
std::unordered_set<Direction<Dim>> get_directions();

template <>
std::unordered_set<Direction<1>> get_directions<1>() {
  return std::unordered_set<Direction<1>>{Direction<1>::upper_xi()};
}

template <>
std::unordered_set<Direction<2>> get_directions<2>() {
  return std::unordered_set<Direction<2>>{Direction<2>::upper_xi(),
                                          Direction<2>::lower_eta()};
}

template <>
std::unordered_set<Direction<3>> get_directions<3>() {
  return std::unordered_set<Direction<3>>{Direction<3>::upper_xi(),
                                          Direction<3>::lower_eta(),
                                          Direction<3>::lower_zeta()};
}

template <size_t Dim>
void test_boundary_coordinates_moving_mesh_impl(
    const ElementMap<Dim, Frame::Grid>& logical_to_grid_map,
    const std::unique_ptr<CoordinateMapBase<Frame::Grid, Frame::Inertial, Dim>>&
        grid_to_inertial_map,
    const std::unique_ptr<
        CoordinateMapBase<Frame::ElementLogical, Frame::Inertial, Dim>>&
        logical_to_inertial_map,
    const double time,
    const std::unordered_map<
        std::string, std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>>&
        functions_of_time) {
  CAPTURE(Dim);
  const Mesh<Dim> mesh{3, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  auto box = db::create<
      db::AddSimpleTags<Tags::Mesh<Dim>, Directions<Dim>,
                        Tags::ElementMap<Dim, Frame::Grid>,
                        CoordinateMaps::Tags::CoordinateMap<Dim, Frame::Grid,
                                                            Frame::Inertial>,
                        ::Tags::Time, Tags::FunctionsOfTimeInitialize>,
      db::AddComputeTags<
          Tags::InterfaceCompute<Directions<Dim>, Tags::Direction<Dim>>,
          Tags::InterfaceCompute<Directions<Dim>, Tags::InterfaceMesh<Dim>>,
          Tags::InterfaceCompute<Directions<Dim>,
                                 Tags::BoundaryCoordinates<Dim, true>>>>(
      mesh, get_directions<Dim>(),
      ElementMap<Dim, Frame::Grid>(logical_to_grid_map.element_id(),
                                   logical_to_grid_map.block_map().get_clone()),
      grid_to_inertial_map->get_clone(), time,
      clone_unique_ptrs(functions_of_time));
  for (const auto& direction_and_coordinates :
       db::get<Tags::Interface<Directions<Dim>,
                               Tags::Coordinates<Dim, Frame::Inertial>>>(box)) {
    const auto& direction = direction_and_coordinates.first;
    CHECK_ITERABLE_APPROX(
        (*logical_to_inertial_map)(
            interface_logical_coordinates(
                mesh.slice_away(direction.dimension()), direction),
            time, functions_of_time),
        direction_and_coordinates.second);
  }
}

void test_boundary_coordinates_moving_mesh() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> dist(-1., 1.);

  const std::array<double, 4> times_to_check{{0.0, 0.3, 1.1, 7.8}};
  const double outer_boundary = 10.0;
  std::array<std::string, 2> functions_of_time_names{
      {"ExpansionA", "ExpansionB"}};
  using Polynomial = domain::FunctionsOfTime::PiecewisePolynomial<2>;
  using FoftPtr = std::unique_ptr<domain::FunctionsOfTime::FunctionOfTime>;
  std::unordered_map<std::string, FoftPtr> functions_of_time{};
  const std::array<DataVector, 3> init_func_a{{{1.0}, {-0.1}, {0.0}}};
  const std::array<DataVector, 3> init_func_b{{{1.0}, {0.0}, {0.0}}};
  const double initial_time = 0.0;
  const double expiration_time = 10.0;
  functions_of_time["ExpansionA"] =
      std::make_unique<Polynomial>(initial_time, init_func_a, expiration_time);
  functions_of_time["ExpansionB"] =
      std::make_unique<Polynomial>(initial_time, init_func_b, expiration_time);

  const auto perform_checks = [&functions_of_time, &times_to_check](
                                  const auto& element_id,
                                  const auto& time_independent_map,
                                  const auto& time_dependent_map) {
    INFO(std::decay_t<decltype(element_id)>::volume_dim);
    const ElementMap<std::decay_t<decltype(element_id)>::volume_dim,
                     Frame::Grid>
        logical_to_grid_map(
            element_id,
            make_coordinate_map_base<Frame::BlockLogical, Frame::Grid>(
                time_independent_map));
    const auto grid_to_inertial_map =
        make_coordinate_map_base<Frame::Grid, Frame::Inertial>(
            time_dependent_map);
    const auto logical_to_inertial_map =
        make_coordinate_map_base<Frame::ElementLogical, Frame::Inertial>(
            time_independent_map, time_dependent_map);

    for (const double time : times_to_check) {
      test_boundary_coordinates_moving_mesh_impl(
          logical_to_grid_map, grid_to_inertial_map, logical_to_inertial_map,
          time, functions_of_time);
    }
  };

  perform_checks(ElementId<1>(0), CoordinateMaps::Affine{-1.0, 1.0, 2.0, 7.8},
                 CoordinateMaps::TimeDependent::CubicScale<1>{
                     outer_boundary, functions_of_time_names[0],
                     functions_of_time_names[1]});
  perform_checks(ElementId<2>(0), CoordinateMaps::Rotation<2>{atan2(4., 3.)},
                 CoordinateMaps::TimeDependent::CubicScale<2>{
                     outer_boundary, functions_of_time_names[0],
                     functions_of_time_names[1]});
  perform_checks(
      ElementId<3>(0),
      CoordinateMaps::ProductOf2Maps<CoordinateMaps::Affine,
                                     CoordinateMaps::Rotation<2>>{
          {-1., 1., 2., 7.}, CoordinateMaps::Rotation<2>(atan2(4., 3.))},
      CoordinateMaps::TimeDependent::CubicScale<3>{outer_boundary,
                                                   functions_of_time_names[0],
                                                   functions_of_time_names[1]});
}

struct SimpleBase : db::SimpleTag {
  using type = int;
};

struct ComputeBase : db::SimpleTag {
  using type = double;
};

struct ComputeDerived : ComputeBase, db::ComputeTag {
  using base = ComputeBase;
  using return_type = double;
  static void function(const gsl::not_null<double*> result, const int& arg) {
    *result = arg + 1.5;
  }
  using argument_tags = tmpl::list<SimpleBase>;
};

void test_interface_base_tags() {
  const auto interface = [](const auto xi_value, const auto eta_value) {
    return std::unordered_map<Direction<2>, std::decay_t<decltype(xi_value)>>{
        {Direction<2>::lower_xi(), xi_value},
        {Direction<2>::upper_eta(), eta_value}};
  };
  const auto box = db::create<
      db::AddSimpleTags<Tags::Interface<Dirs, SimpleBase>>,
      db::AddComputeTags<DirsCompute,
                         Tags::InterfaceCompute<Dirs, ComputeDerived>>>(
      interface(4, 5));
  CHECK(get<Tags::Interface<Dirs, SimpleBase>>(box) == interface(4, 5));
  CHECK(get<Tags::Interface<Dirs, ComputeBase>>(box) == interface(5.5, 6.5));
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.InterfaceItems", "[Unit][Domain]") {
  test_interface_items();
  test_interface_subitems();
  test_interface_slice();
  test_boundary_coordinates_moving_mesh();
  test_interface_base_tags();
}
}  // namespace domain
