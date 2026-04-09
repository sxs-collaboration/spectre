// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <cstddef>
#include <numeric>
#include <unordered_map>
#include <utility>

#include "DataStructures/DataBox/DataBox.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/TaggedTuple.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Structure/DirectionMap.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Domain/Structure/Side.hpp"
#include "Domain/Tags.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/TestTags.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
#include "ParallelAlgorithms/Amr/Projectors/Mesh.hpp"
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

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                        VariablesTag<0>, VariablesTag<1>>>(
      new_mesh, element, std::move(var_0), std::move(var_1));

  db::mutate_apply<amr::projectors::ProjectVariables<
      Dim, tmpl::list<VariablesTag<0>, VariablesTag<1>>>>(
      make_not_null(&box), std::make_pair(old_mesh, element));

  CHECK_VARIABLES_APPROX(db::get<VariablesTag<0>>(box), expected_var_0);
  CHECK_VARIABLES_APPROX(db::get<VariablesTag<1>>(box), expected_var_1);
}

template <size_t Dim>
void test_h_refine() {
  const ElementId<Dim> parent_element_id{0};
  const Element<Dim> parent_element{parent_element_id,
                                    DirectionMap<Dim, Neighbors<Dim>>{}};
  const std::array children_element_ids{
      parent_element_id.id_of_child(0, Side::Lower),
      parent_element_id.id_of_child(0, Side::Upper)};
  const std::array children_elements{
      Element<Dim>{children_element_ids[0],
                   DirectionMap<Dim, Neighbors<Dim>>{}},
      Element<Dim>{children_element_ids[1],
                   DirectionMap<Dim, Neighbors<Dim>>{}}};
  const Mesh<Dim> mesh{4, Spectral::Basis::Legendre,
                       Spectral::Quadrature::GaussLobatto};
  const auto parent_logical_coords = logical_coordinates(mesh);
  auto parent_var_0 = make_vars(parent_logical_coords, 1.0);
  auto parent_var_1 = make_vars(parent_logical_coords, 2.0);
  std::array children_logical_coords{parent_logical_coords,
                                     parent_logical_coords};
  get<0>(children_logical_coords[0]) =
      0.5 * (get<0>(parent_logical_coords) - 1.0);
  get<0>(children_logical_coords[1]) =
      0.5 * (get<0>(parent_logical_coords) + 1.0);
  const std::array children_var_0{make_vars(children_logical_coords[0], 1.0),
                                  make_vars(children_logical_coords[1], 1.0)};
  const std::array children_var_1{make_vars(children_logical_coords[0], 2.0),
                                  make_vars(children_logical_coords[1], 2.0)};

  for (size_t child = 0; child < 2; ++child) {
    auto box = db::create<
        db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                          VariablesTag<0>, VariablesTag<1>>>(
        mesh, gsl::at(children_elements, child), VariablesType{},
        VariablesType{});

    db::mutate_apply<amr::projectors::ProjectVariables<
        Dim, tmpl::list<VariablesTag<0>, VariablesTag<1>>>>(
        make_not_null(&box),
        tuples::TaggedTuple<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                            VariablesTag<0>, VariablesTag<1>>(
            parent_element, mesh, parent_var_0, parent_var_1));

    CHECK_VARIABLES_APPROX(db::get<VariablesTag<0>>(box),
                           gsl::at(children_var_0, child));
    CHECK_VARIABLES_APPROX(db::get<VariablesTag<1>>(box),
                           gsl::at(children_var_1, child));
  }

  {
    auto box = db::create<
        db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                          VariablesTag<0>, VariablesTag<1>>>(
        mesh, parent_element, VariablesType{}, VariablesType{});

    std::unordered_map<
        ElementId<Dim>,
        tuples::TaggedTuple<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                            VariablesTag<0>, VariablesTag<1>>>
        children_data{};
    children_data.insert(
        {children_element_ids[0],
         {children_elements[0], mesh, children_var_0[0], children_var_1[0]}});
    children_data.insert(
        {children_element_ids[1],
         {children_elements[1], mesh, children_var_0[1], children_var_1[1]}});

    db::mutate_apply<amr::projectors::ProjectVariables<
        Dim, tmpl::list<VariablesTag<0>, VariablesTag<1>>>>(make_not_null(&box),
                                                            children_data);

    CHECK_VARIABLES_APPROX(db::get<VariablesTag<0>>(box), parent_var_0);
    CHECK_VARIABLES_APPROX(db::get<VariablesTag<1>>(box), parent_var_1);
  }
}

template <size_t Dim>
void test_nonuniform_join() {
  const ElementId<Dim> parent_element_id{0};
  const Element<Dim> parent_element{parent_element_id,
                                    DirectionMap<Dim, Neighbors<Dim>>{}};
  const std::array children_element_ids{
      parent_element_id.id_of_child(0, Side::Lower),
      parent_element_id.id_of_child(0, Side::Upper)};
  const std::array children_elements{
      Element<Dim>{children_element_ids[0],
                   DirectionMap<Dim, Neighbors<Dim>>{}},
      Element<Dim>{children_element_ids[1],
                   DirectionMap<Dim, Neighbors<Dim>>{}}};
  std::vector<Mesh<Dim>> children_meshes{};
  {
    std::array<size_t, Dim> extents{};
    std::generate(extents.begin(), extents.end(),
                  // NOLINTNEXTLINE(spectre-mutable) - false positive
                  [value = 3]() mutable { return value++; });
    children_meshes.emplace_back(extents, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto);
    std::generate(extents.begin(), extents.end(),
                  // NOLINTNEXTLINE(spectre-mutable) - false positive
                  [value = 5]() mutable { return value--; });
    children_meshes.emplace_back(extents, Spectral::Basis::Legendre,
                                 Spectral::Quadrature::GaussLobatto);
  }
  const Mesh<Dim> parent_mesh = amr::projectors::parent_mesh(children_meshes);
  const auto parent_logical_coords = logical_coordinates(parent_mesh);
  auto parent_var_0 = make_vars(parent_logical_coords, 1.0);
  auto parent_var_1 = make_vars(parent_logical_coords, 2.0);
  std::array children_logical_coords{logical_coordinates(children_meshes[0]),
                                     logical_coordinates(children_meshes[1])};
  get<0>(children_logical_coords[0]) =
      0.5 * (get<0>(children_logical_coords[0]) - 1.0);
  get<0>(children_logical_coords[1]) =
      0.5 * (get<0>(children_logical_coords[1]) + 1.0);
  const std::array children_var_0{make_vars(children_logical_coords[0], 1.0),
                                  make_vars(children_logical_coords[1], 1.0)};
  const std::array children_var_1{make_vars(children_logical_coords[0], 2.0),
                                  make_vars(children_logical_coords[1], 2.0)};

  auto box = db::create<
      db::AddSimpleTags<domain::Tags::Mesh<Dim>, domain::Tags::Element<Dim>,
                        VariablesTag<0>, VariablesTag<1>>>(
      parent_mesh, parent_element, VariablesType{}, VariablesType{});

  std::unordered_map<
      ElementId<Dim>,
      tuples::TaggedTuple<domain::Tags::Element<Dim>, domain::Tags::Mesh<Dim>,
                          VariablesTag<0>, VariablesTag<1>>>
      children_data{};
  children_data.insert({children_element_ids[0],
                        {children_elements[0], children_meshes[0],
                         children_var_0[0], children_var_1[0]}});
  children_data.insert({children_element_ids[1],
                        {children_elements[1], children_meshes[1],
                         children_var_0[1], children_var_1[1]}});

  db::mutate_apply<amr::projectors::ProjectVariables<
      Dim, tmpl::list<VariablesTag<0>, VariablesTag<1>>>>(make_not_null(&box),
                                                          children_data);

  CHECK_VARIABLES_APPROX(db::get<VariablesTag<0>>(box), parent_var_0);
  CHECK_VARIABLES_APPROX(db::get<VariablesTag<1>>(box), parent_var_1);
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
  test_h_refine<1>();
  test_h_refine<2>();
  test_h_refine<3>();
  test_nonuniform_join<1>();
  test_nonuniform_join<2>();
  test_nonuniform_join<3>();
}
