// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Tags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Tensors.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Literals.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

namespace {

struct Tag0 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
struct Tag1 : db::SimpleTag {
  using type = tnsr::i<DataVector, Dim>;
};

template <size_t Dim>
struct Tag2 : db::SimpleTag {
  using type = tnsr::iJ<DataVector, Dim>;
};

template <typename T>
T f(const T& x, const std::array<double, 3>& c) {
  return c[0] + c[1] * x + c[2] * square(x);
}

template <size_t Dim>
DataVector make_component(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x,
    const double scale) {
  const auto number_of_points = get<0>(x).size();
  DataVector result{number_of_points, scale};
  const auto x_coeffs = std::array{0.75, -1.75, 2.75};
  result *= f(x[0], x_coeffs);
  if constexpr (Dim > 1) {
    const auto y_coeffs = std::array{-0.25, 1.25, -2.25};
    result *= f(x[1], y_coeffs);
  }
  if constexpr (Dim > 2) {
    const auto z_coeffs = std::array{0.125, -1.625, -2.875};
    result *= f(x[2], z_coeffs);
  }
  return result;
}

template <typename TensorType, size_t Dim>
TensorType make_tensor(const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x,
                       const double tensor_scale) {
  TensorType result = make_with_value<TensorType>(
      x, std::numeric_limits<double>::signaling_NaN());
  for (size_t i = 0; i < result.size(); ++i) {
    result[i] = make_component(x, tensor_scale + i);
  }
  return result;
}

template <size_t Dim>
void test_p_refine() {
  const ElementId<Dim> element_id{0};
  const Element<Dim> element{element_id, DirectionMap<Dim, Neighbors<Dim>>{}};
  const Mesh<Dim> old_mesh{4, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  std::array<size_t, Dim> new_extents{};
  std::iota(new_extents.begin(), new_extents.end(), 3_st);
  const Mesh<Dim> new_mesh{new_extents, Spectral::Basis::Legendre,
                           Spectral::Quadrature::GaussLobatto};
  const auto x_old = logical_coordinates(old_mesh);
  const auto x_new = logical_coordinates(new_mesh);
  auto scalar = make_tensor<typename Tag0::type>(x_old, 1.0);
  auto one_form = make_tensor<typename Tag1<Dim>::type>(x_old, 4.0);
  auto deriv = make_tensor<typename Tag2<Dim>::type>(x_old, 8.0);
  const auto expected_scalar = make_tensor<typename Tag0::type>(x_new, 1.0);
  const auto expected_one_form =
      make_tensor<typename Tag1<Dim>::type>(x_new, 4.0);
  const auto expected_deriv = make_tensor<typename Tag2<Dim>::type>(x_new, 8.0);

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                        Tag0, Tag1<Dim>, Tag2<Dim>>>(
      new_mesh, element, std::move(scalar), std::move(one_form),
      std::move(deriv));

  db::mutate_apply<amr::projectors::ProjectTensors<
      Dim, tmpl::list<Tag0, Tag1<Dim>, Tag2<Dim>>>>(
      make_not_null(&box), std::make_pair(old_mesh, element));
  CHECK_ITERABLE_APPROX(db::get<Tag0>(box), expected_scalar);
  CHECK_ITERABLE_APPROX(db::get<Tag1<Dim>>(box), expected_one_form);
  CHECK_ITERABLE_APPROX(db::get<Tag2<Dim>>(box), expected_deriv);
}

template <size_t Dim>
void test_h_refine() {
  const ElementId<Dim> parent_element_id{0};
  std::array<size_t, Dim> extents{};
  std::iota(extents.begin(), extents.end(), 3_st);
  const Mesh<Dim> mesh{extents, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto parent_x = logical_coordinates(mesh);

  using ElementData =
      tuples::TaggedTuple<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                          Tag0, Tag1<Dim>, Tag2<Dim>>;
  const ElementData parent_data{
      Element<Dim>{parent_element_id, {}}, mesh,
      make_tensor<typename Tag0::type>(parent_x, 1.0),
      make_tensor<typename Tag1<Dim>::type>(parent_x, 4.0),
      make_tensor<typename Tag2<Dim>::type>(parent_x, 8.0)};

  std::unordered_map<ElementId<Dim>, ElementData> children_data{};
  for (size_t child = 0; child < two_to_the(Dim); ++child) {
    auto child_id = parent_element_id;
    auto child_x = parent_x;
    for (size_t d = 0; d < Dim; ++d) {
      const auto side = (child & (1_st << d)) == 0 ? Side::Lower : Side::Upper;
      child_id = child_id.id_of_child(d, side);
      child_x.get(d) *= 0.5;
      child_x.get(d) += side == Side::Upper ? 0.5 : -0.5;
    }
    children_data.emplace(
        child_id,
        ElementData{Element<Dim>{child_id, {}}, mesh,
                    make_tensor<typename Tag0::type>(child_x, 1.0),
                    make_tensor<typename Tag1<Dim>::type>(child_x, 4.0),
                    make_tensor<typename Tag2<Dim>::type>(child_x, 8.0)});
  }

  for (const auto& [child_id, child_data] : children_data) {
    auto box = db::create<
        db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                          Tag0, Tag1<Dim>, Tag2<Dim>>>(
        mesh, get<domain::Tags::Element<Dim>>(child_data),
        typename Tag0::type{}, typename Tag1<Dim>::type{},
        typename Tag2<Dim>::type{});

    db::mutate_apply<amr::projectors::ProjectTensors<
        Dim, tmpl::list<Tag0, Tag1<Dim>, Tag2<Dim>>>>(make_not_null(&box),
                                                      parent_data);

    const auto& expected = children_data.at(child_id);
    CHECK_ITERABLE_APPROX(db::get<Tag0>(box), get<Tag0>(expected));
    CHECK_ITERABLE_APPROX(db::get<Tag1<Dim>>(box), get<Tag1<Dim>>(expected));
    CHECK_ITERABLE_APPROX(db::get<Tag2<Dim>>(box), get<Tag2<Dim>>(expected));
  }

  {
    auto box = db::create<
        db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                          Tag0, Tag1<Dim>, Tag2<Dim>>>(
        mesh, get<domain::Tags::Element<Dim>>(parent_data),
        typename Tag0::type{}, typename Tag1<Dim>::type{},
        typename Tag2<Dim>::type{});

    db::mutate_apply<amr::projectors::ProjectTensors<
        Dim, tmpl::list<Tag0, Tag1<Dim>, Tag2<Dim>>>>(make_not_null(&box),
                                                      children_data);

    CHECK_ITERABLE_APPROX(db::get<Tag0>(box), get<Tag0>(parent_data));
    CHECK_ITERABLE_APPROX(db::get<Tag1<Dim>>(box), get<Tag1<Dim>>(parent_data));
    CHECK_ITERABLE_APPROX(db::get<Tag2<Dim>>(box), get<Tag2<Dim>>(parent_data));
  }
}

template <size_t Dim>
void test_nonuniform_join() {
  const ElementId<Dim> parent_element_id{0};

  using ElementData =
      tuples::TaggedTuple<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                          Tag0, Tag1<Dim>, Tag2<Dim>>;

  std::unordered_map<ElementId<Dim>, ElementData> children_data{};
  std::vector<Mesh<Dim>> children_meshes{};
  for (size_t child = 0; child < two_to_the(Dim); ++child) {
    const Mesh<Dim> child_mesh{child + 3, Spectral::Basis::Legendre,
                               Spectral::Quadrature::GaussLobatto};
    children_meshes.push_back(child_mesh);
    auto child_id = parent_element_id;
    auto child_x = logical_coordinates(child_mesh);
    for (size_t d = 0; d < Dim; ++d) {
      const auto side = (child & (1_st << d)) == 0 ? Side::Lower : Side::Upper;
      child_id = child_id.id_of_child(d, side);
      child_x.get(d) *= 0.5;
      child_x.get(d) += side == Side::Upper ? 0.5 : -0.5;
    }
    children_data.emplace(
        child_id,
        ElementData{Element<Dim>{child_id, {}}, child_mesh,
                    make_tensor<typename Tag0::type>(child_x, 1.0),
                    make_tensor<typename Tag1<Dim>::type>(child_x, 4.0),
                    make_tensor<typename Tag2<Dim>::type>(child_x, 8.0)});
  }

  const auto parent_mesh = amr::projectors::parent_mesh(children_meshes);
  const auto parent_x = logical_coordinates(parent_mesh);

  const ElementData parent_data{
      Element<Dim>{parent_element_id, {}}, parent_mesh,
      make_tensor<typename Tag0::type>(parent_x, 1.0),
      make_tensor<typename Tag1<Dim>::type>(parent_x, 4.0),
      make_tensor<typename Tag2<Dim>::type>(parent_x, 8.0)};

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                        Tag0, Tag1<Dim>, Tag2<Dim>>>(
      parent_mesh, get<domain::Tags::Element<Dim>>(parent_data),
      typename Tag0::type{}, typename Tag1<Dim>::type{},
      typename Tag2<Dim>::type{});

  db::mutate_apply<amr::projectors::ProjectTensors<
      Dim, tmpl::list<Tag0, Tag1<Dim>, Tag2<Dim>>>>(make_not_null(&box),
                                                    children_data);

  auto custom_approx = Approx::custom().scale(1.0).epsilon(1.e-13);
  CHECK_ITERABLE_CUSTOM_APPROX(db::get<Tag0>(box), get<Tag0>(parent_data),
                               custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(db::get<Tag1<Dim>>(box),
                               get<Tag1<Dim>>(parent_data), custom_approx);
  CHECK_ITERABLE_CUSTOM_APPROX(db::get<Tag2<Dim>>(box),
                               get<Tag2<Dim>>(parent_data), custom_approx);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Projectors.Tensors", "[ParallelAlgorithms][Unit]") {
  static_assert(
      tt::assert_conforms_to_v<amr::projectors::ProjectTensors<
                                   1, tmpl::list<Tag0, Tag1<1>, Tag2<1>>>,
                               amr::protocols::Projector>);
  test_p_refine<1>();
  test_p_refine<2>();
  test_p_refine<3>();
  test_h_refine<1>();
  test_h_refine<2>();
  test_h_refine<3>();
  test_nonuniform_join<1>();
  test_nonuniform_join<2>();
  test_nonuniform_join<3>();
}
