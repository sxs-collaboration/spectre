// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Variables.hpp"
#include "ParallelAlgorithms/Amr/Protocols/Projector.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"

namespace {

using VariablesType =
    Variables<tmpl::list<TestHelpers::Tags::Scalar<DataVector>>>;

template <size_t Label>
struct VariablesTag : db::SimpleTag {
  using type = VariablesType;
};

template <typename T>
T f(const T& x, const std::array<double, 3>& c) {
  return c[0] + c[1] * x + c[2] * square(x);
}

template <size_t Dim>
VariablesType make_vars(
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& x,
    const double scale) {
  const auto number_of_points = get<0>(x).size();
  VariablesType result{number_of_points, scale};
  const auto x_coeffs = std::array{0.75, -1.75, 2.75};
  DataVector& s = get(get<TestHelpers::Tags::Scalar<DataVector>>(result));
  s *= f(x[0], x_coeffs);
  if constexpr (Dim > 1) {
    const auto y_coeffs = std::array{-0.25, 1.25, -2.25};
    s *= f(x[1], y_coeffs);
  }
  if constexpr (Dim > 2) {
    const auto z_coeffs = std::array{0.125, -1.625, -2.875};
    s *= f(x[2], z_coeffs);
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
  auto var_0 = make_vars(x_old, 1.0);
  auto var_1 = make_vars(x_old, 2.0);
  const auto expected_var_0 = make_vars(x_new, 1.0);
  const auto expected_var_1 = make_vars(x_new, 2.0);

  auto box = db::create<db::AddSimpleTags<domain::Tags::Mesh<Dim>,
                                          VariablesTag<0>, VariablesTag<1>>>(
      new_mesh, std::move(var_0), std::move(var_1));

  db::mutate_apply<amr::projectors::ProjectVariables<
      Dim, tmpl::list<VariablesTag<0>, VariablesTag<1>>>>(
      make_not_null(&box), std::make_pair(old_mesh, element));

  CHECK_VARIABLES_APPROX(db::get<VariablesTag<0>>(box), expected_var_0);
  CHECK_VARIABLES_APPROX(db::get<VariablesTag<1>>(box), expected_var_1);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Amr.Projectors.Variables",
                  "[ParallelAlgorithms][Unit]") {
  static_assert(tt::assert_conforms_to_v<
                amr::projectors::ProjectVariables<
                    1, tmpl::list<VariablesTag<0>, VariablesTag<1>>>,
                amr::protocols::Projector>);
  test_p_refine<1>();
  test_p_refine<2>();
  test_p_refine<3>();
}
