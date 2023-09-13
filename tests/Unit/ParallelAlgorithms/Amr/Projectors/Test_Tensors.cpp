// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Tensors.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

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
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, Tag0, Tag1<Dim>, Tag2<Dim>>>(
      new_mesh, std::move(scalar), std::move(one_form), std::move(deriv));

  db::mutate_apply<amr::projectors::ProjectTensors<
      Dim, tmpl::list<Tag0, Tag1<Dim>, Tag2<Dim>>>>(
      make_not_null(&box), std::make_pair(old_mesh, element));
  CHECK_ITERABLE_APPROX(db::get<Tag0>(box), expected_scalar);
  CHECK_ITERABLE_APPROX(db::get<Tag1<Dim>>(box), expected_one_form);
  CHECK_ITERABLE_APPROX(db::get<Tag2<Dim>>(box), expected_deriv);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Projectors.Tensors", "[ParallelAlgorithms][Unit]") {
  test_p_refine<1>();
  test_p_refine<2>();
  test_p_refine<3>();
}
