// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <functional>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/DataBoxTag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VariablesHelpers.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/Neighbors.hpp"  // IWYU pragma: keep
#include "Domain/Tags.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"

// IWYU pragma: no_forward_declare Tensor

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
struct Int : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Int";
  using type = int;
};

struct Double : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Double";
  using type = double;
};

template <size_t N>
struct NoCopy : db::DataBoxTag {
  static constexpr db::DataBoxString label = "NoCopy";
  using type = ::NoCopy<N>;
};

template <typename Tag>
struct Negate : db::DataBoxPrefix, db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Negate";
  using tag = Tag;
  static constexpr auto function(const db::item_type<Tag>& x) noexcept {
    return -x;
  }
  using argument_tags = tmpl::list<Tag>;
};

struct AddThree : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "AddThree";
  static constexpr auto function(const int x) noexcept { return x + 3; }
  using argument_tags = tmpl::list<Int>;
  using volume_tags = tmpl::list<Int>;
};

template <size_t VolumeDim>
struct ComplexComputeItem : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ComplexComputeItem";
  static constexpr auto function(const int i, const double d,
                                 const ::NoCopy<1>& /*unused*/,
                                 const ::NoCopy<2>& /*unused*/) noexcept {
    return std::make_pair(i, d);
  }
  using argument_tags = tmpl::list<Int, Double, NoCopy<1>, NoCopy<2>>;
  using volume_tags = tmpl::list<Int, NoCopy<1>>;
};

template <typename>
struct TemplatedDirections : db::DataBoxTag {
  static constexpr db::DataBoxString label = "TemplatedDirections";
  using type = std::unordered_set<Direction<3>>;
};
}  // namespace TestTags
}  // namespace

namespace Tags {
template <typename DirectionsTag>
struct Interface<DirectionsTag, TestTags::Int>
    : InterfaceBase<DirectionsTag, TestTags::Int, TestTags::AddThree> {};
}  // namespace Tags

SPECTRE_TEST_CASE("Unit.Domain.InterfaceItems", "[Unit][Domain]") {
  constexpr size_t dim = 3;
  using internal_directions = Tags::InternalDirections<dim>;
  using templated_directions = TestTags::TemplatedDirections<int>;

  Element<dim> element{ElementId<3>(0),
                       {{Direction<dim>::lower_xi(), {}},
                        {Direction<dim>::upper_xi(), {}},
                        {Direction<dim>::upper_zeta(), {}}}};

  std::unordered_map<Direction<dim>, NoCopy<2>> nocopy_map_item;
  nocopy_map_item.emplace(Direction<dim>::lower_xi(), NoCopy<2>{});
  nocopy_map_item.emplace(Direction<dim>::upper_xi(), NoCopy<2>{});
  nocopy_map_item.emplace(Direction<dim>::upper_zeta(), NoCopy<2>{});
  const auto box = db::create<
      db::AddSimpleTags<
          Tags::Element<dim>, TestTags::Int,
          Tags::Interface<internal_directions, TestTags::Double>,
          TestTags::NoCopy<1>,
          Tags::Interface<internal_directions, TestTags::NoCopy<2>>,
          templated_directions,
          Tags::Interface<templated_directions, TestTags::Double>>,
      db::AddComputeTags<
          internal_directions,
          Tags::Interface<internal_directions, Tags::Direction<dim>>,
          TestTags::Negate<TestTags::Int>,
          Tags::Interface<internal_directions, TestTags::Int>,
          Tags::Interface<internal_directions,
                          TestTags::Negate<TestTags::Double>>,
          Tags::Interface<internal_directions,
                          TestTags::ComplexComputeItem<dim>>,
          Tags::Interface<templated_directions, Tags::Direction<dim>>,
          Tags::Interface<templated_directions,
                          TestTags::Negate<TestTags::Double>>>>(
      std::move(element), 5,
      std::unordered_map<Direction<dim>, double>{
          {Direction<dim>::lower_xi(), 1.5},
          {Direction<dim>::upper_xi(), 2.5},
          {Direction<dim>::upper_zeta(), 3.5}},
      NoCopy<1>{}, std::move(nocopy_map_item),
      std::unordered_set<Direction<dim>>{Direction<dim>::upper_xi()},
      std::unordered_map<Direction<dim>, double>{
          {Direction<dim>::upper_xi(), 4.5}});

  CHECK(
      (get<Tags::Interface<internal_directions, Tags::Direction<dim>>>(box)) ==
      (std::unordered_map<Direction<dim>, Direction<dim>>{
          {Direction<dim>::lower_xi(), Direction<dim>::lower_xi()},
          {Direction<dim>::upper_xi(), Direction<dim>::upper_xi()},
          {Direction<dim>::upper_zeta(), Direction<dim>::upper_zeta()}}));

  CHECK(get<TestTags::Negate<TestTags::Int>>(box) == -5);
  CHECK((get<Tags::Interface<internal_directions,
                             TestTags::Negate<TestTags::Double>>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::lower_xi(), -1.5},
            {Direction<dim>::upper_xi(), -2.5},
            {Direction<dim>::upper_zeta(), -3.5}}));

  CHECK((get<Tags::Interface<internal_directions, TestTags::Int>>(box)) ==
        (std::unordered_map<Direction<dim>, int>{
            {Direction<dim>::lower_xi(), 8},
            {Direction<dim>::upper_xi(), 8},
            {Direction<dim>::upper_zeta(), 8}}));

  CHECK((get<Tags::Interface<internal_directions,
                             TestTags::ComplexComputeItem<dim>>>(box)) ==
        (std::unordered_map<Direction<dim>, std::pair<int, double>>{
            {Direction<dim>::lower_xi(), {5, 1.5}},
            {Direction<dim>::upper_xi(), {5, 2.5}},
            {Direction<dim>::upper_zeta(), {5, 3.5}}}));

  CHECK((get<Tags::Interface<templated_directions,
                             TestTags::Negate<TestTags::Double>>>(box)) ==
        (std::unordered_map<Direction<dim>, double>{
            {Direction<dim>::upper_xi(), -4.5}}));
}

namespace {
constexpr size_t dim = 2;

struct Dirs : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Dirs";
  static auto function() noexcept {
    return std::unordered_set<Direction<dim>>{Direction<dim>::lower_xi(),
                                              Direction<dim>::upper_eta()};
  }
  using argument_tags = tmpl::list<>;
};

template <size_t N>
struct Var : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Var";
  using type = Scalar<DataVector>;
  static constexpr bool should_be_sliced_to_boundary =
      N == 3 or N == 30 or  // sliced_simple_item_tag below
      N == 2 or N == 20;    // sliced_compute_item_tag below
};

template <size_t VolumeDim>
struct Compute : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "Compute";
  static auto function(const Index<VolumeDim>& extents) {
    auto ret = Variables<tmpl::list<Var<VolumeDim>, Var<10 * VolumeDim>>>(
        extents.product(), VolumeDim);
    get(get<Var<10 * VolumeDim>>(ret)) *= 5;
    return ret;
  }
  using argument_tags = tmpl::list<Tags::Extents<VolumeDim>>;
};

template <size_t N>
auto make_interface_variables(DataVector value_xi,
                              DataVector value_eta) noexcept {
  const auto make = [](DataVector value) noexcept {
    Variables<tmpl::list<Var<N>>> v(value.size());
    get(get<Var<N>>(v)) = std::move(value);
    return v;
  };
  std::unordered_map<Direction<dim>, decltype(make(value_xi))> ret;
  ret.emplace(Direction<dim>::lower_xi(), make(std::move(value_xi)));
  ret.emplace(Direction<dim>::upper_eta(), make(std::move(value_eta)));
  return ret;
}

template <size_t N0, size_t N1>
auto make_interface_variables(
    DataVector value_xi0, DataVector value_xi1,
    DataVector value_eta0, DataVector value_eta1) noexcept {
  const auto make = [](DataVector value0, DataVector value1) noexcept {
    Variables<tmpl::list<Var<N0>, Var<N1>>> v(value0.size());
    get(get<Var<N0>>(v)) = std::move(value0);
    get(get<Var<N1>>(v)) = std::move(value1);
    return v;
  };
  std::unordered_map<Direction<dim>, decltype(make(value_xi0, value_xi1))> ret;
  ret.emplace(Direction<dim>::lower_xi(),
              make(std::move(value_xi0), std::move(value_xi1)));
  ret.emplace(Direction<dim>::upper_eta(),
              make(std::move(value_eta0), std::move(value_eta1)));
  return ret;
}

auto make_interface_tensor(DataVector value_xi, DataVector value_eta) noexcept {
  std::unordered_map<Direction<dim>, Scalar<DataVector>> ret;
  ret.emplace(Direction<dim>::lower_xi(),
              Scalar<DataVector>(std::move(value_xi)));
  ret.emplace(Direction<dim>::upper_eta(),
              Scalar<DataVector>(std::move(value_eta)));
  return ret;
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.InterfaceItems.Subitems", "[Unit][Domain]") {
  const Index<dim> extents{{{4, 3}}};

  const DataVector boundary_vars_xi{10., 11., 12.};
  const DataVector boundary_vars_eta{20., 21., 22., 23.};

  auto box = db::create<
      db::AddSimpleTags<
          Tags::Extents<dim>,
          Tags::Interface<Dirs, Tags::Variables<tmpl::list<Var<0>>>>>,
      db::AddComputeTags<Dirs, Tags::Interface<Dirs, Tags::Direction<dim>>,
                         Tags::Interface<Dirs, Tags::Extents<dim - 1>>,
                         Tags::Interface<Dirs, Compute<1>>>>(
      extents,
      make_interface_variables<0>(boundary_vars_xi, boundary_vars_eta));

  CHECK((db::get<Tags::Interface<Dirs, Var<0>>>(box)) ==
        make_interface_tensor(boundary_vars_xi, boundary_vars_eta));
  CHECK((db::get<Tags::Interface<Dirs, Var<1>>>(box)) ==
        make_interface_tensor({1., 1., 1.}, {1., 1., 1., 1.}));
  CHECK((db::get<Tags::Interface<Dirs, Var<10>>>(box)) ==
        make_interface_tensor({5., 5., 5.}, {5., 5., 5., 5.}));

  db::mutate<Tags::Interface<Dirs, Var<0>>>(
      box, [](auto& boundary_tensor) noexcept {
        get(boundary_tensor.at(Direction<dim>::lower_xi())) *= 3.;
      });
  CHECK((db::get<Tags::Interface<Dirs, Var<0>>>(box)) ==
        make_interface_tensor(3. * boundary_vars_xi, boundary_vars_eta));
}

namespace {
using simple_item_tag = Tags::Variables<tmpl::list<Var<0>>>;
using compute_item_tag = Compute<1>;
using sliced_compute_item_tag = Compute<2>;
using sliced_simple_item_tag = Tags::Variables<tmpl::list<Var<3>>>;
}  // namespace

SPECTRE_TEST_CASE("Unit.Domain.InterfaceItems.Slice", "[Unit][Domain]") {
  const Index<dim> extents{{{4, 3}}};

  db::item_type<sliced_simple_item_tag> volume_vars(extents.product());
  get<Var<3>>(volume_vars).get() =
      DataVector{0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};
  const DataVector boundary_vars_xi{10., 11., 12.};
  const DataVector boundary_vars_eta{20., 21., 22., 23.};

  auto box = db::create<
      db::AddSimpleTags<Tags::Extents<dim>, sliced_simple_item_tag,
                        Tags::Interface<Dirs, simple_item_tag>>,
      db::AddComputeTags<Dirs, sliced_compute_item_tag,
                         Tags::Interface<Dirs, Tags::Direction<dim>>,
                         Tags::Interface<Dirs, Tags::Extents<dim - 1>>,
                         Tags::Interface<Dirs, compute_item_tag>,
                         Tags::Interface<Dirs, sliced_compute_item_tag>,
                         Tags::Interface<Dirs, sliced_simple_item_tag>>>(
      extents, std::move(volume_vars),
      make_interface_variables<0>(boundary_vars_xi, boundary_vars_eta));

  CHECK((db::get<Tags::Interface<Dirs, simple_item_tag>>(box)) ==
        make_interface_variables<0>(boundary_vars_xi, boundary_vars_eta));
  CHECK((db::get<Tags::Interface<Dirs, compute_item_tag>>(box)) ==
        (make_interface_variables<1, 10>({1., 1., 1.}, {5., 5., 5.},
                                         {1., 1., 1., 1.}, {5., 5., 5., 5.})));
  CHECK((db::get<Tags::Interface<Dirs, sliced_compute_item_tag>>(box)) ==
        (make_interface_variables<2, 20>(
             {2., 2., 2.}, {10., 10., 10.},
             {2., 2., 2., 2.}, {10., 10., 10., 10.})));
  CHECK((db::get<Tags::Interface<Dirs, sliced_simple_item_tag>>(box)) ==
        make_interface_variables<3>({0., 4., 8.}, {8., 9., 10., 11.}));
}

namespace partial_slice {
template <size_t N>
struct Scalar : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Scalar";
  using type = ::Scalar<DataVector>;
  static constexpr bool should_be_sliced_to_boundary = N == 0;
};

constexpr size_t vector_dim = 3;
template <size_t N>
struct Vector : db::DataBoxTag {
  static constexpr db::DataBoxString label = "Vector";
  using type = tnsr::i<DataVector, vector_dim, Frame::Logical>;
  static constexpr bool should_be_sliced_to_boundary = N == 1;
};

template <size_t VolumeDim>
struct ComputedVars : db::ComputeItemTag {
  static constexpr db::DataBoxString label = "ComputedVars";
  static auto function(const Index<VolumeDim>& extents) noexcept {
    const DataVector volume_data{
      11., 10., 9., 8., 7., 6., 5., 4., 3., 2., 1., 0.};

    Variables<tmpl::list<Scalar<1>, Vector<1>>> vars(extents.product());
    get<Scalar<1>>(vars).get() = volume_data;
    for (size_t i = 0; i < vector_dim; ++i) {
      get<Vector<1>>(vars).get(i) = volume_data * (i + 2);
    }
    return vars;
  }
  using argument_tags = tmpl::list<Tags::Extents<VolumeDim>>;
};

using volume_vars_tag = Tags::Variables<tmpl::list<Scalar<0>, Vector<0>>>;
using slice_tag = Tags::Variables<tmpl::list<Vector<1>, Scalar<0>>>;
}  // namespace partial_slice

SPECTRE_TEST_CASE("Unit.Domain.InterfaceItems.PartialSlice", "[Unit][Domain]") {
  const Index<dim> extents{{{4, 3}}};

  const DataVector volume_data{
    0., 1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11.};

  db::item_type<partial_slice::volume_vars_tag> volume_vars(extents.product());
  get<partial_slice::Scalar<0>>(volume_vars).get() = volume_data;
  for (size_t i = 0; i < partial_slice::vector_dim; ++i) {
    get<partial_slice::Vector<0>>(volume_vars).get(i) = volume_data * (i + 2);
  }

  const auto box = db::create<
      db::AddSimpleTags<Tags::Extents<dim>, partial_slice::volume_vars_tag>,
      db::AddComputeTags<Dirs, partial_slice::ComputedVars<dim>,
                         Tags::Interface<Dirs, Tags::Direction<dim>>,
                         Tags::Interface<Dirs, Tags::Extents<dim - 1>>,
                         Tags::Interface<Dirs, partial_slice::slice_tag>>>(
      extents, std::move(volume_vars));

  Variables<tmpl::list<partial_slice::Vector<1>, partial_slice::Scalar<0>>>
      expected_xi(3);
  get(get<partial_slice::Scalar<0>>(expected_xi)) = DataVector{0., 4., 8.};
  for (size_t i = 0; i < partial_slice::vector_dim; ++i) {
    get<partial_slice::Vector<1>>(expected_xi).get(i) =
        (i + 2) * DataVector{11., 7., 3.};
  }

  Variables<tmpl::list<partial_slice::Vector<1>, partial_slice::Scalar<0>>>
      expected_eta(4);
  get(get<partial_slice::Scalar<0>>(expected_eta)) =
      DataVector{8., 9., 10., 11.};
  for (size_t i = 0; i < partial_slice::vector_dim; ++i) {
    get<partial_slice::Vector<1>>(expected_eta).get(i) =
        (i + 2) * DataVector{3., 2., 1., 0.};
  }

  CHECK((db::get<Tags::Interface<Dirs, partial_slice::slice_tag>>(box)) ==
        (std::unordered_map<Direction<dim>, decltype(expected_xi)>{
            {Direction<dim>::lower_xi(), expected_xi},
            {Direction<dim>::upper_eta(), expected_eta}}));
}
