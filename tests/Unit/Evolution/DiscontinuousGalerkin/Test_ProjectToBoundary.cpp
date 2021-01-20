// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>

#include "DataStructures/ApplyMatrices.hpp"
#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Index.hpp"
#include "DataStructures/SliceVariables.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Evolution/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace {
struct Var1 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

struct Var2 : db::SimpleTag {
  using type = tnsr::I<DataVector, 2, Frame::Inertial>;
};

struct Var3 : db::SimpleTag {
  using type = Scalar<DataVector>;
};

template <size_t Dim>
Variables<tmpl::list<Var2, Var3>> polynomial_volume_data(
    const tnsr::I<DataVector, Dim, Frame::Logical>& coords,
    const Index<Dim>& powers) noexcept {
  Variables<tmpl::list<Var2, Var3>> result(get<0>(coords).size(), 1.0);
  for (size_t i = 0; i < Dim; ++i) {
    get(get<Var3>(result)) *= pow(coords.get(i), powers[i]);
    get<0>(get<Var2>(result)) *= 2.0 * pow(coords.get(i), powers[i]);
    get<1>(get<Var2>(result)) *= 3.0 * pow(coords.get(i), powers[i]);
  }
  return result;
}

template <size_t Dim>
void test(const Spectral::Quadrature quadrature) {
  CAPTURE(Dim);
  CAPTURE(quadrature);

  MAKE_GENERATOR(gen);
  UniformCustomDistribution<size_t> sdist{5, 10};

  Mesh<Dim> volume_mesh{sdist(gen), Spectral::Basis::Legendre, quadrature};
  Index<Dim> powers{};
  for (size_t i = 0; i < Dim; ++i) {
    powers[i] = volume_mesh.extents(i) - 2 - i;
  }

  const auto volume_data =
      polynomial_volume_data(logical_coordinates(volume_mesh), powers);

  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const Scalar<DataVector> var3_volume = get<Var3>(volume_data);
  // NOLINTNEXTLINE(performance-unnecessary-copy-initialization)
  const tnsr::I<DataVector, 2, Frame::Inertial> var2_volume =
      get<Var2>(volume_data);

  for (const auto& direction : Direction<Dim>::all_directions()) {
    const size_t sliced_dim = direction.dimension();
    const size_t fixed_index = direction.side() == Side::Upper
                                   ? volume_mesh.extents(sliced_dim) - 1
                                   : 0;
    const auto face_mesh = volume_mesh.slice_away(sliced_dim);
    Variables<tmpl::list<Var2, Var3>> expected_face_values{};
    if (quadrature == Spectral::Quadrature::GaussLobatto) {
      expected_face_values = data_on_slice(volume_data, volume_mesh.extents(),
                                           sliced_dim, fixed_index);
    } else {
      expected_face_values = polynomial_volume_data(
          interface_logical_coordinates(face_mesh, direction), powers);
    }
    const Scalar<DataVector> expected_var1{face_mesh.number_of_grid_points(),
                                           0.0};
    const Scalar<DataVector> expected_var3{face_mesh.number_of_grid_points(),
                                           0.0};

    Variables<tmpl::list<Var1, Var2, Var3>> face_values{
        face_mesh.number_of_grid_points(), 0.0};
    evolution::dg::project_tensors_to_boundary<tmpl::list<Var2>>(
        make_not_null(&face_values), volume_data, volume_mesh, direction);
    CHECK(get<Var1>(face_values) == expected_var1);
    CHECK(get<Var2>(face_values) == get<Var2>(expected_face_values));
    CHECK(get<Var3>(face_values) == expected_var3);

    evolution::dg::project_tensors_to_boundary<tmpl::list<Var3>>(
        make_not_null(&face_values), volume_data, volume_mesh, direction);
    CHECK(get<Var1>(face_values) == expected_var1);
    CHECK(get<Var2>(face_values) == get<Var2>(expected_face_values));
    CHECK(get<Var3>(face_values) == get<Var3>(expected_face_values));

    face_values.initialize(face_mesh.number_of_grid_points(), 0.0);
    evolution::dg::project_tensors_to_boundary<tmpl::list<Var2, Var3>>(
        make_not_null(&face_values), volume_data, volume_mesh, direction);
    CHECK(get<Var1>(face_values) == expected_var1);
    CHECK(get<Var2>(face_values) == get<Var2>(expected_face_values));
    CHECK(get<Var3>(face_values) == get<Var3>(expected_face_values));

    Variables<tmpl::list<Var1, Var2, Var3>> face_values_contiguous_project{
        face_mesh.number_of_grid_points(), 0.0};
    evolution::dg::project_contiguous_data_to_boundary(
        make_not_null(&face_values_contiguous_project), volume_data,
        volume_mesh, direction);
    CHECK(get<Var1>(face_values_contiguous_project) == expected_var1);
    CHECK(get<Var2>(face_values_contiguous_project) ==
          get<Var2>(expected_face_values));
    CHECK(get<Var3>(face_values_contiguous_project) ==
          get<Var3>(expected_face_values));

    Scalar<DataVector> var3_face{face_mesh.number_of_grid_points()};
    evolution::dg::project_tensor_to_boundary(
        make_not_null(&var3_face), var3_volume, volume_mesh, direction);
    CHECK(var3_face == get<Var3>(expected_face_values));

    tnsr::I<DataVector, 2, Frame::Inertial> var2_face{
        face_mesh.number_of_grid_points()};
    evolution::dg::project_tensor_to_boundary(
        make_not_null(&var2_face), var2_volume, volume_mesh, direction);
    CHECK(var2_face == get<Var2>(expected_face_values));
  }
}
}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.ProjectToBoundary", "[Unit][Evolution]") {
  for (const auto quadrature :
       {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
    test<1>(quadrature);
    test<2>(quadrature);
    test<3>(quadrature);
  }
}
