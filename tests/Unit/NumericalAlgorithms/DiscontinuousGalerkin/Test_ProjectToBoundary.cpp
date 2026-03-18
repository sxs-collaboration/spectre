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
#include "Domain/InterfaceLogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "NumericalAlgorithms/DiscontinuousGalerkin/ProjectToBoundary.hpp"
#include "NumericalAlgorithms/Spectral/Basis.hpp"
#include "NumericalAlgorithms/Spectral/LogicalCoordinates.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Quadrature.hpp"
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
    const tnsr::I<DataVector, Dim, Frame::ElementLogical>& coords,
    const Index<Dim>& powers) {
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

  const bool using_zernike =
      quadrature == Spectral::Quadrature::GaussRadauUpper;

  const Mesh<Dim> volume_mesh{
      sdist(gen),
      using_zernike ? Spectral::Basis::ZernikeB2 : Spectral::Basis::Legendre,
      quadrature};
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
    if (using_zernike and direction.side() != Side::Upper) {
      continue;
    }
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
    ::dg::project_tensors_to_boundary<tmpl::list<Var2>>(
        make_not_null(&face_values), volume_data, volume_mesh, direction);
    CHECK_ITERABLE_APPROX(get<Var1>(face_values), expected_var1);
    CHECK_ITERABLE_APPROX(get<Var2>(face_values),
                          get<Var2>(expected_face_values));
    CHECK_ITERABLE_APPROX(get<Var3>(face_values), expected_var3);

    ::dg::project_tensors_to_boundary<tmpl::list<Var3>>(
        make_not_null(&face_values), volume_data, volume_mesh, direction);
    CHECK_ITERABLE_APPROX(get<Var1>(face_values), expected_var1);
    CHECK_ITERABLE_APPROX(get<Var2>(face_values),
                          get<Var2>(expected_face_values));
    CHECK_ITERABLE_APPROX(get<Var3>(face_values),
                          get<Var3>(expected_face_values));

    face_values.initialize(face_mesh.number_of_grid_points(), 0.0);
    ::dg::project_tensors_to_boundary<tmpl::list<Var2, Var3>>(
        make_not_null(&face_values), volume_data, volume_mesh, direction);
    CHECK_ITERABLE_APPROX(get<Var1>(face_values), expected_var1);
    CHECK_ITERABLE_APPROX(get<Var2>(face_values),
                          get<Var2>(expected_face_values));
    CHECK_ITERABLE_APPROX(get<Var3>(face_values),
                          get<Var3>(expected_face_values));

    Variables<tmpl::list<Var1, Var2, Var3>> face_values_contiguous_project{
        face_mesh.number_of_grid_points(), 0.0};
    ::dg::project_contiguous_data_to_boundary(
        make_not_null(&face_values_contiguous_project), volume_data,
        volume_mesh, direction);
    CHECK_ITERABLE_APPROX(get<Var1>(face_values_contiguous_project),
                          expected_var1);
    CHECK_ITERABLE_APPROX(get<Var2>(face_values_contiguous_project),
                          get<Var2>(expected_face_values));
    CHECK_ITERABLE_APPROX(get<Var3>(face_values_contiguous_project),
                          get<Var3>(expected_face_values));

    Scalar<DataVector> var3_face{face_mesh.number_of_grid_points()};
    ::dg::project_tensor_to_boundary(make_not_null(&var3_face), var3_volume,
                                     volume_mesh, direction);
    CHECK_ITERABLE_APPROX(var3_face, get<Var3>(expected_face_values));

    tnsr::I<DataVector, 2, Frame::Inertial> var2_face{
        face_mesh.number_of_grid_points()};
    ::dg::project_tensor_to_boundary(make_not_null(&var2_face), var2_volume,
                                     volume_mesh, direction);
    CHECK_ITERABLE_APPROX(var2_face, get<Var2>(expected_face_values));
  }
}

void test_asserts() {
#ifdef SPECTRE_DEBUG
  const Mesh<1> mesh(5, Spectral::Basis::ZernikeB2,
                     Spectral::Quadrature::GaussRadauUpper);
  Variables<tmpl::list<Var1>> face{mesh.number_of_grid_points(), 0.0};
  const Variables<tmpl::list<Var1>> volume{mesh.number_of_grid_points(), 0.0};

  CHECK_THROWS_WITH(
      ::dg::project_tensors_to_boundary<tmpl::list<Var1>>(
          make_not_null(&face), volume, mesh, Direction<1>::lower_xi()),
      Catch::Matchers::ContainsSubstring(
          "Got quadrature without boundary collocation point at"));
#endif  // SPECTRE_DEBUG
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DiscontinuousGalerkin.ProjectToBoundary",
                  "[Unit][NumericalAlgorithms]") {
  for (const auto quadrature :
       {Spectral::Quadrature::GaussLobatto, Spectral::Quadrature::Gauss}) {
    test<1>(quadrature);
    test<2>(quadrature);
    test<3>(quadrature);
  }
  test<1>(Spectral::Quadrature::GaussRadauUpper);
  test_asserts();
}
