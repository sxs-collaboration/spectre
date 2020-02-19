// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "tests/Unit/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/Direction.hpp"
#include "Domain/Element.hpp"
#include "Domain/ElementId.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Mesh.hpp"
#include "Domain/Neighbors.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoImpl.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"
#include "tests/Unit/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"

// IWYU pragma: no_forward_declare Variables

namespace {

struct ScalarTag : db::SimpleTag {
  using type = Scalar<DataVector>;
  static std::string name() noexcept { return "Scalar"; }
};

template <size_t VolumeDim>
struct VectorTag : db::SimpleTag {
  using type = tnsr::I<DataVector, VolumeDim>;
  static std::string name() noexcept { return "Vector"; }
};

void test_secondary_neighbors_to_exclude_from_fit() {
  INFO("Testing Hweno_detail::secondary_neighbors_to_exclude_from_fit");
  struct DummyPackage {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>> means;
  };
  std::unordered_map<std::pair<Direction<2>, ElementId<2>>, DummyPackage,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      dummy_neighbor_data;

  const auto lower_xi =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>{1});
  const auto upper_xi =
      std::make_pair(Direction<2>::upper_xi(), ElementId<2>{2});
  const auto lower_eta =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>{3});
  const auto upper_eta =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>{4});

  get(get<::Tags::Mean<ScalarTag>>(dummy_neighbor_data[lower_xi].means)) = 1.;
  get(get<::Tags::Mean<ScalarTag>>(dummy_neighbor_data[upper_xi].means)) = 2.;
  get(get<::Tags::Mean<ScalarTag>>(dummy_neighbor_data[lower_eta].means)) = 3.;
  get(get<::Tags::Mean<ScalarTag>>(dummy_neighbor_data[upper_eta].means)) = 3.;

  const auto check_excluded_neighbors = [&dummy_neighbor_data](
      const double mean,
      const std::pair<Direction<2>, ElementId<2>>& primary_neighbor,
      const std::unordered_set<
          std::pair<Direction<2>, ElementId<2>>,
          boost::hash<std::pair<Direction<2>, ElementId<2>>>>&
          expected_excluded_neighbors) noexcept {
    const size_t tensor_index = 0;
    const auto excluded_neighbors_vector =
        Limiters::Hweno_detail::secondary_neighbors_to_exclude_from_fit<
            ScalarTag>(mean, tensor_index, dummy_neighbor_data,
                       primary_neighbor);
    // The elements of `excluded_neighbors_vector` are ordered in an undefined
    // way, because they are filled by looping over the unordered_map of
    // neighbor data. To provide meaningful test comparisons, we move the data
    // into an unordered_set. (A sort would also work here, if the Direction and
    // ElementId classes were sortable, which they aren't.)
    const std::unordered_set<std::pair<Direction<2>, ElementId<2>>,
                             boost::hash<std::pair<Direction<2>, ElementId<2>>>>
        excluded_neighbors(excluded_neighbors_vector.begin(),
                           excluded_neighbors_vector.end());
    CHECK(excluded_neighbors == expected_excluded_neighbors);
  };

  check_excluded_neighbors(0., lower_xi, {{lower_eta, upper_eta}});
  check_excluded_neighbors(0., upper_xi, {{lower_eta, upper_eta}});
  check_excluded_neighbors(0., lower_eta, {{upper_eta}});
  check_excluded_neighbors(0., upper_eta, {{lower_eta}});
  check_excluded_neighbors(3., lower_xi, {{upper_xi}});
  check_excluded_neighbors(3., upper_xi, {{lower_xi}});
  check_excluded_neighbors(3., lower_eta, {{lower_xi}});
  check_excluded_neighbors(3., upper_eta, {{lower_xi}});
  check_excluded_neighbors(4., lower_xi, {{upper_xi}});
  check_excluded_neighbors(4., upper_xi, {{lower_xi}});
  check_excluded_neighbors(4., lower_eta, {{lower_xi}});
  check_excluded_neighbors(4., upper_eta, {{lower_xi}});
}

void test_hweno_modified_solution_1d() {
  INFO("Testing hweno_modified_neighbor_solution in 1D");
  using TagsList = tmpl::list<ScalarTag>;
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto mesh = Mesh<1>{
      {{3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  const auto primary_neighbor =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>{2});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    return Scalar<DataVector>{{{1. - 0.2 * x + 0.4 * square(x)}}};
  }
  ();

  const auto primary_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) - 2.;
    get(get<ScalarTag>(result)) = 4. - 0.5 * x - 0.1 * square(x);
    return result;
  }
  ();

  const auto upper_xi_vars = [&mesh]() noexcept {
    // Large values to make sure this neighbor is excluded
    Variables<TagsList> result(mesh.number_of_grid_points(), 1.e3);
    return result;
  }
  ();

  const auto make_tuple_of_means = [&mesh](
      const Variables<TagsList>& vars) noexcept {
    return tuples::TaggedTuple<::Tags::Mean<ScalarTag>>(
        mean_value(get(get<ScalarTag>(vars)), mesh));
  };

  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>> means;
    Variables<TagsList> volume_data;
    Mesh<1> mesh;
  };
  std::unordered_map<std::pair<Direction<1>, ElementId<1>>, PackagedData,
                     boost::hash<std::pair<Direction<1>, ElementId<1>>>>
      neighbor_data{};
  neighbor_data[primary_neighbor].means = make_tuple_of_means(primary_vars);
  neighbor_data[primary_neighbor].volume_data = primary_vars;
  neighbor_data[primary_neighbor].mesh = mesh;
  neighbor_data[upper_xi_neighbor].means = make_tuple_of_means(upper_xi_vars);
  neighbor_data[upper_xi_neighbor].volume_data = upper_xi_vars;
  neighbor_data[upper_xi_neighbor].mesh = mesh;

  Scalar<DataVector> modified_tensor;
  Limiters::hweno_modified_neighbor_solution<ScalarTag>(
      make_not_null(&modified_tensor), local_tensor, element, mesh,
      neighbor_data, primary_neighbor);

  // The expected coefficient values for the result of the constrained fit are
  // found using Mathematica, using the following code (for Mathematica v10):
  // qw3 = {1/3, 4/3, 1/3};
  // qx3 = {-1, 0, 1};
  // quad3[f_, dx_] := Sum[qw3[[qi]] f[qx3[[qi]] + dx], {qi, 1, 3}];
  // trial[c0_, c1_, c2_][x_] := c0 + c1 x + c2 x^2;
  // uLocal[x_] := 1 - 1/5 x + 2/5 x^2;
  // uPrimary[x_] := 4 - 1/2 x - 1/10 x^2;
  // primaryOptimizationTerm[c0_, c1_, c2_][x_] :=
  //     (trial[c0, c1, c2][x] - uPrimary[x])^2;
  // Minimize[
  //     quad3[primaryOptimizationTerm[c0, c1, c2], -2],
  //     quad3[trial[c0, c1, c2], 0] == quad3[uLocal, 0],
  // {c0, c1, c2}
  // ]
  const auto expected = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    constexpr std::array<double, 3> c{{41. / 30., -31. / 10., -7. / 10.}};
    return Scalar<DataVector>{{{c[0] + c[1] * x + c[2] * square(x)}}};
  }
  ();

  // Fit procedure has somewhat larger error scale than default
  Approx local_approx = Approx::custom().epsilon(1e-11).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(modified_tensor, expected, local_approx);
  // Verify that the constraint is in fact satisfied
  CHECK(mean_value(get(modified_tensor), mesh) ==
        local_approx(mean_value(get(local_tensor), mesh)));
}

// Test in 2D using a vector tensor, to test multiple components. In particular,
// check that different components will correctly exclude different neighbors if
// needed.
// Multiple components becomes very tedious in 3D, so 3D will test a scalar.
void test_hweno_modified_solution_2d_vector() {
  INFO("Testing hweno_modified_neighbor_solution in 2D");
  using TagsList = tmpl::list<VectorTag<2>>;
  const auto element = TestHelpers::Limiters::make_element<2>();
  const auto mesh = Mesh<2>{
      {{4, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  const auto lower_xi_neighbor =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<2>::upper_xi(), ElementId<2>{2});
  const auto primary_neighbor =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>{3});
  const auto upper_eta_neighbor =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>{4});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    return VectorTag<2>::type{{{DataVector{1. + 0.1 * x + 0.2 * y +
                                           0.1 * x * y + 0.1 * x * square(y)},
                                DataVector(x.size(), 2.)}}};
  }
  ();

  const auto lower_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    // The x component should exclude this neighbor
    get<0>(get<VectorTag<2>>(result)) = 1.e3;
    get<1>(get<VectorTag<2>>(result)) = 0.;
    return result;
  }
  ();

  const auto upper_xi_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) + 2.;
    const auto& y = get<1>(logical_coords);
    get<0>(get<VectorTag<2>>(result)) =
        1. + 1. / 3. * x + 0.25 * y - 0.05 * square(x);
    get<1>(get<VectorTag<2>>(result)) = -0.5;
    return result;
  }
  ();

  const auto primary_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) - 2.;
    get<0>(get<VectorTag<2>>(result)) =
        1. + 0.25 * x - 0.2 * square(y) + 0.1 * x * square(y);
    get<1>(get<VectorTag<2>>(result)) =
        1.2 + 0.5 * x - 0.1 * square(x) - 0.2 * y + 0.1 * square(x) * square(y);
    return result;
  }
  ();

  const auto upper_eta_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) + 2.;
    get<0>(get<VectorTag<2>>(result)) = 1. + 1. / 3. * x + 0.2 * y;
    // The y component should exclude this neighbor
    get<1>(get<VectorTag<2>>(result)) = 1.e3;
    return result;
  }
  ();

  const auto make_tuple_of_means = [&mesh](
      const Variables<TagsList>& vars) noexcept {
    return tuples::TaggedTuple<::Tags::Mean<VectorTag<2>>>(tnsr::I<double, 2>{
        {{mean_value(get<0>(get<VectorTag<2>>(vars)), mesh),
          mean_value(get<1>(get<VectorTag<2>>(vars)), mesh)}}});
  };

  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<VectorTag<2>>> means;
    Variables<TagsList> volume_data;
    Mesh<2> mesh;
  };
  std::unordered_map<std::pair<Direction<2>, ElementId<2>>, PackagedData,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  neighbor_data[lower_xi_neighbor].means = make_tuple_of_means(lower_xi_vars);
  neighbor_data[lower_xi_neighbor].volume_data = lower_xi_vars;
  neighbor_data[lower_xi_neighbor].mesh = mesh;
  neighbor_data[upper_xi_neighbor].means = make_tuple_of_means(upper_xi_vars);
  neighbor_data[upper_xi_neighbor].volume_data = upper_xi_vars;
  neighbor_data[upper_xi_neighbor].mesh = mesh;
  neighbor_data[primary_neighbor].means = make_tuple_of_means(primary_vars);
  neighbor_data[primary_neighbor].volume_data = primary_vars;
  neighbor_data[primary_neighbor].mesh = mesh;
  neighbor_data[upper_eta_neighbor].means = make_tuple_of_means(upper_eta_vars);
  neighbor_data[upper_eta_neighbor].volume_data = upper_eta_vars;
  neighbor_data[upper_eta_neighbor].mesh = mesh;

  VectorTag<2>::type modified_tensor;
  Limiters::hweno_modified_neighbor_solution<VectorTag<2>>(
      make_not_null(&modified_tensor), local_tensor, element, mesh,
      neighbor_data, primary_neighbor);

  // The expected coefficient values for the result of the constrained fit are
  // found using Mathematica, using the following code (for Mathematica v10).
  // This example computes the expected result for the vector x component; an
  // analogous piece of code gives the expected result for the y component, but
  // note that the neighbor to exclude changes in this case.
  // qw3 = {1/3, 4/3, 1/3};
  // qx3 = {-1, 0, 1};
  // qw4 = {1/6, 5/6, 5/6, 1/6};
  // qx4 = {-1, -1/Sqrt[5], 1/Sqrt[5], 1};
  // quad43[f_, dx_, dy_] := Sum[
  //     qw4[[qi]] qw3[[qj]] f[qx4[[qi]] + dx, qx3[[qj]] + dy],
  //     {qi, 1, 4}, {qj, 1, 3}];
  // trial[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_, c9_, c10_, c11_][
  //       x_, y_] :=
  //     c0 + c1 x + c2 x^2 + c3 x^3 +
  //     y (c4 + c5 x + c6 x^2 + c7 x^3) +
  //     y^2 (c8 + c9 x + c10 x^2 + c11 x^3);
  // uLocal[x_, y_] := 1 + 1/10 x + 1/5 y + 1/10 x y + 1/10 x y^2;
  // uPrimary[x_, y_] := 1 + 1/4 x - 1/5 y^2 + 1/10 x y^2;
  // uUpperXi[x_, y_] := 1 + 1/3 x + 1/4 y - 1/20 x^2;
  // uUpperEta[x_, y_] := 1 + 1/3 x + 1/5 y;
  // primaryOptimizationTerm[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_,
  //                         c9_, c10_, c11_][x_, y_] :=
  //     (trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11][x, y]
  //      - uPrimary[x, y])^2;
  // Minimize[
  //     quad43[primaryOptimizationTerm[c0, c1, c2, c3, c4, c5, c6, c7, c8,
  //                                    c9, c10, c11],
  //            0, -2]
  //     + (quad43[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
  //               2, 0] - quad43[uUpperXi, 2, 0])^2
  //     + (quad43[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
  //               0, 2] - quad43[uUpperEta, 0, 2])^2,
  //     quad43[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
  //            0, 0] == quad43[uLocal, 0, 0],
  // {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11}
  // ]
  const auto expected = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    // x-component coefficients
    constexpr std::array<double, 12> c{
        {6049013. / 5913219., 92367. / 360620., -1659. / 558961.,
         -6083. / 558961., 61831556. / 187251935., 42. / 6935., -126. / 42997.,
         -462. / 42997., -2460491. / 37450387., 18283. / 180310.,
         -378. / 558961., -1386. / 558961.}};
    // y-component coefficients
    constexpr std::array<double, 12> d{
        {991516363. / 468183825., 1142461. / 1991042., -686461. / 1224010.,
         -130350. / 995521., 110206607. / 156061275., 72540. / 995521.,
         -55692. / 122401., -128700. / 995521., 10878374. / 52020425.,
         16740. / 995521., -6119. / 1224010., -29700. / 995521.}};
    return tnsr::I<DataVector, 2>{
        {{DataVector{c[0] + c[1] * x + c[2] * square(x) + c[3] * cube(x) +
                     y * (c[4] + c[5] * x + c[6] * square(x) + c[7] * cube(x)) +
                     square(y) * (c[8] + c[9] * x + c[10] * square(x) +
                                  c[11] * cube(x))},
          DataVector{d[0] + d[1] * x + d[2] * square(x) + d[3] * cube(x) +
                     y * (d[4] + d[5] * x + d[6] * square(x) + d[7] * cube(x)) +
                     square(y) * (d[8] + d[9] * x + d[10] * square(x) +
                                  d[11] * cube(x))}}}};
  }
  ();

  // Fit procedure has somewhat larger error scale than default
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(modified_tensor, expected, local_approx);
  // Verify that the constraint is in fact satisfied
  CHECK(mean_value(get<0>(modified_tensor), mesh) ==
        local_approx(mean_value(get<0>(local_tensor), mesh)));
  CHECK(mean_value(get<1>(modified_tensor), mesh) ==
        local_approx(mean_value(get<1>(local_tensor), mesh)));
}

// Test in 2D (now using a scalar tensor) but with two of the neighbors being
// excluded due to large means.
void test_hweno_modified_solution_2d_exclude_two_neighbors() {
  INFO("Testing hweno_modified_neighbor_solution in 2D");
  using TagsList = tmpl::list<ScalarTag>;
  const auto element = TestHelpers::Limiters::make_element<2>();
  const auto mesh = Mesh<2>{
      {{4, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  const auto lower_xi_neighbor =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<2>::upper_xi(), ElementId<2>{2});
  const auto primary_neighbor =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>{3});
  const auto upper_eta_neighbor =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>{4});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    return ScalarTag::type{{{DataVector{1. + 0.1 * x + 0.2 * y + 0.1 * x * y +
                                        0.1 * x * square(y)}}}};
  }
  ();

  const auto lower_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    // Large values to make sure this neighbor is excluded
    get(get<ScalarTag>(result)) = 1.e3;
    return result;
  }
  ();

  const auto upper_xi_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) + 2.;
    const auto& y = get<1>(logical_coords);
    get(get<ScalarTag>(result)) =
        1. + 1. / 3. * x + 0.25 * y - 0.05 * square(x);
    return result;
  }
  ();

  const auto primary_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) - 2.;
    get(get<ScalarTag>(result)) =
        1. + 0.25 * x - 0.2 * square(y) + 0.1 * x * square(y);
    return result;
  }
  ();

  const auto upper_eta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    // Large values to make sure this neighbor is excluded
    get(get<ScalarTag>(result)) = 1.e3;
    return result;
  }
  ();

  const auto make_tuple_of_means = [&mesh](
      const Variables<TagsList>& vars) noexcept {
    return tuples::TaggedTuple<::Tags::Mean<ScalarTag>>(
        mean_value(get(get<ScalarTag>(vars)), mesh));
  };

  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>> means;
    Variables<TagsList> volume_data;
    Mesh<2> mesh;
  };
  std::unordered_map<std::pair<Direction<2>, ElementId<2>>, PackagedData,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  neighbor_data[lower_xi_neighbor].means = make_tuple_of_means(lower_xi_vars);
  neighbor_data[lower_xi_neighbor].volume_data = lower_xi_vars;
  neighbor_data[lower_xi_neighbor].mesh = mesh;
  neighbor_data[upper_xi_neighbor].means = make_tuple_of_means(upper_xi_vars);
  neighbor_data[upper_xi_neighbor].volume_data = upper_xi_vars;
  neighbor_data[upper_xi_neighbor].mesh = mesh;
  neighbor_data[primary_neighbor].means = make_tuple_of_means(primary_vars);
  neighbor_data[primary_neighbor].volume_data = primary_vars;
  neighbor_data[primary_neighbor].mesh = mesh;
  neighbor_data[upper_eta_neighbor].means = make_tuple_of_means(upper_eta_vars);
  neighbor_data[upper_eta_neighbor].volume_data = upper_eta_vars;
  neighbor_data[upper_eta_neighbor].mesh = mesh;

  ScalarTag::type modified_tensor;
  Limiters::hweno_modified_neighbor_solution<ScalarTag>(
      make_not_null(&modified_tensor), local_tensor, element, mesh,
      neighbor_data, primary_neighbor);

  // The expected coefficient values for the result of the constrained fit are
  // found using Mathematica, using the following code (for Mathematica v10).
  // qw3 = {1/3, 4/3, 1/3};
  // qx3 = {-1, 0, 1};
  // qw4 = {1/6, 5/6, 5/6, 1/6};
  // qx4 = {-1, -1/Sqrt[5], 1/Sqrt[5], 1};
  // quad43[f_, dx_, dy_] := Sum[
  //     qw4[[qi]] qw3[[qj]] f[qx4[[qi]] + dx, qx3[[qj]] + dy],
  //     {qi, 1, 4}, {qj, 1, 3}];
  // trial[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_, c9_, c10_, c11_][
  //       x_, y_] :=
  //     c0 + c1 x + c2 x^2 + c3 x^3 +
  //     y (c4 + c5 x + c6 x^2 + c7 x^3) +
  //     y^2 (c8 + c9 x + c10 x^2 + c11 x^3);
  // uLocal[x_, y_] := 1 + 1/10 x + 1/5 y + 1/10 x y + 1/10 x y^2;
  // uPrimary[x_, y_] := 1 + 1/4 x - 1/5 y^2 + 1/10 x y^2;
  // uUpperXi[x_, y_] := 1 + 1/3 x + 1/4 y - 1/20 x^2;
  // primaryOptimizationTerm[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_,
  //                         c9_, c10_, c11_][x_, y_] :=
  //     (trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11][x, y]
  //      - uPrimary[x, y])^2;
  // Minimize[
  //     quad43[primaryOptimizationTerm[c0, c1, c2, c3, c4, c5, c6, c7, c8,
  //                                    c9, c10, c11],
  //            0, -2]
  //     + (quad43[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
  //               2, 0] - quad43[uUpperXi, 2, 0])^2,
  //     quad43[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
  //            0, 0] == quad43[uLocal, 0, 0],
  // {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11}
  // ]
  const auto expected = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    constexpr std::array<double, 12> c{
        {757538269. / 712675275., 92367. / 360620., -1659. / 558961.,
         -6083. / 558961., 1135772. / 18273725., 42. / 6935., -126. / 42997.,
         -462. / 42997., -44104369. / 237558425., 18283. / 180310.,
         -378. / 558961., -1386. / 558961.}};
    return Scalar<DataVector>{{{DataVector{
        c[0] + c[1] * x + c[2] * square(x) + c[3] * cube(x) +
        y * (c[4] + c[5] * x + c[6] * square(x) + c[7] * cube(x)) +
        square(y) * (c[8] + c[9] * x + c[10] * square(x) + c[11] * cube(x))}}}};
  }
  ();

  // Fit procedure has somewhat larger error scale than default
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(modified_tensor, expected, local_approx);
  // Verify that the constraint is in fact satisfied
  CHECK(mean_value(get(modified_tensor), mesh) ==
        local_approx(mean_value(get(local_tensor), mesh)));
}

void test_hweno_modified_solution_3d() {
  INFO("Testing hweno_modified_neighbor_solution in 3D");
  using TagsList = tmpl::list<ScalarTag>;
  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>> means;
    Variables<TagsList> volume_data;
    Mesh<3> mesh;
  };

  const auto lower_xi_neighbor =
      std::make_pair(Direction<3>::lower_xi(), ElementId<3>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<3>::upper_xi(), ElementId<3>{2});
  const auto lower_eta_neighbor =
      std::make_pair(Direction<3>::lower_eta(), ElementId<3>{3});
  const auto upper_eta_neighbor =
      std::make_pair(Direction<3>::upper_eta(), ElementId<3>{4});
  const auto lower_zeta_neighbor =
      std::make_pair(Direction<3>::lower_zeta(), ElementId<3>{5});
  const auto primary_neighbor =
      std::make_pair(Direction<3>::upper_zeta(), ElementId<3>{6});

  const auto element = TestHelpers::Limiters::make_element<3>();
  const auto mesh = Mesh<3>{{{3, 3, 4}},
                            Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto& z = get<2>(logical_coords);
    return Scalar<DataVector>{{{0.5 + 0.2 * x + 0.1 * square(y) * z}}};
  }
  ();

  const auto lower_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 1.2;
    return result;
  }
  ();

  const auto upper_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 4.;
    return result;
  }
  ();

  const auto lower_eta_vars = [&mesh]() noexcept {
    // Large values to make sure this neighbor is excluded
    Variables<TagsList> result(mesh.number_of_grid_points(), 1.e3);
    return result;
  }
  ();

  const auto upper_eta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 2.5;
    return result;
  }
  ();

  const auto lower_zeta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 2.;
    return result;
  }
  ();

  const auto primary_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto z = get<2>(logical_coords) + 2.;
    get(get<ScalarTag>(result)) =
        1. + 0.25 * x + 0.1 * y * z + 0.5 * x * square(y) * z + 0.1 * cube(z);
    return result;
  }
  ();

  const auto make_tuple_of_means = [&mesh](
      const Variables<TagsList>& vars) noexcept {
    return tuples::TaggedTuple<::Tags::Mean<ScalarTag>>(
        mean_value(get(get<ScalarTag>(vars)), mesh));
  };

  std::unordered_map<std::pair<Direction<3>, ElementId<3>>, PackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data{};
  neighbor_data[lower_xi_neighbor].means = make_tuple_of_means(lower_xi_vars);
  neighbor_data[lower_xi_neighbor].volume_data = lower_xi_vars;
  neighbor_data[lower_xi_neighbor].mesh = mesh;
  neighbor_data[upper_xi_neighbor].means = make_tuple_of_means(upper_xi_vars);
  neighbor_data[upper_xi_neighbor].volume_data = upper_xi_vars;
  neighbor_data[upper_xi_neighbor].mesh = mesh;
  neighbor_data[lower_eta_neighbor].means = make_tuple_of_means(lower_eta_vars);
  neighbor_data[lower_eta_neighbor].volume_data = lower_eta_vars;
  neighbor_data[lower_eta_neighbor].mesh = mesh;
  neighbor_data[upper_eta_neighbor].means = make_tuple_of_means(upper_eta_vars);
  neighbor_data[upper_eta_neighbor].volume_data = upper_eta_vars;
  neighbor_data[upper_eta_neighbor].mesh = mesh;
  neighbor_data[lower_zeta_neighbor].means =
      make_tuple_of_means(lower_zeta_vars);
  neighbor_data[lower_zeta_neighbor].volume_data = lower_zeta_vars;
  neighbor_data[lower_zeta_neighbor].mesh = mesh;
  neighbor_data[primary_neighbor].means = make_tuple_of_means(primary_vars);
  neighbor_data[primary_neighbor].volume_data = primary_vars;
  neighbor_data[primary_neighbor].mesh = mesh;

  Scalar<DataVector> modified_tensor;
  Limiters::hweno_modified_neighbor_solution<ScalarTag>(
      make_not_null(&modified_tensor), local_tensor, element, mesh,
      neighbor_data, primary_neighbor);

  // The expected coefficient values for the result of the constrained fit are
  // found using Mathematica, using the following code (for Mathematica v10).
  // qw3 = {1/3, 4/3, 1/3};
  // qx3 = {-1, 0, 1};
  // qw4 = {1/6, 5/6, 5/6, 1/6};
  // qx4 = {-1, -1/Sqrt[5], 1/Sqrt[5], 1};
  // quad334[f_, dx_, dy_, dz_] := Sum[
  //     qw3[[qi]] qw3[[qj]] qw4[[qk]]
  //     f[qx3[[qi]] + dx, qx3[[qj]] + dy, qx4[[qk]] + dz],
  //     {qi, 1, 3}, {qj, 1, 3}, {qk, 1, 4}];
  // trial[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_, c9_, c10_, c11_,
  //       c12_, c13_, c14_, c15_, c16_, c17_, c18_, c19_, c20_, c21_, c22_,
  //       c23_, c24_, c25_, c26_, c27_, c28_, c29_, c30_, c31_, c32_, c33_,
  //       c34_, c35_][x_, y_, z_] :=
  //     c0 + c1 x + c2 x^2 + y (c3 + c4 x + c5 x^2) +
  //                          y^2 (c6 + c7 x + c8 x^2) +
  //     z (c9 + c10 x + c11 x^2 + y (c12 + c13 x + c14 x^2) +
  //                               y^2 (c15 + c16 x + c17 x^2)) +
  //     z^2 (c18 + c19 x + c20 x^2 + y (c21 + c22 x + c23 x^2) +
  //                                  y^2 (c24 + c25 x + c26 x^2)) +
  //     z^3 (c27 + c28 x + c29 x^2 + y (c30 + c31 x + c32 x^2) +
  //                                  y^2 (c33 + c34 x + c35 x^2));
  // uLocal[x_, y_, z_] := 1/2 + 1/5 x + 1/10 y^2 z;
  // uPrimary[x_, y_, z_] := 1 + 1/4 x + 1/10 y z + 1/2 x y^2 z + 1/10 z^3;
  // uLowerXi[x_, y_, z_] := 6/5;
  // uUpperXi[x_, y_, z_] := 4;
  // uUpperEta[x_, y_, z_] := 5/2;
  // uLowerZeta[x_, y_, z_] := 2;
  // primaryOptimizationTerm[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_,
  //                         c9_, c10_, c11_, c12_, c13_, c14_, c15_, c16_,
  //                         c17_, c18_, c19_, c20_, c21_, c22_, c23_, c24_,
  //                         c25_, c26_, c27_, c28_, c29_, c30_, c31_, c32_,
  //                         c33_, c34_, c35_][x_, y_, z_] :=
  //     (trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
  //            c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //            c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35][x,
  //            y, z] - uPrimary[x, y, z])^2;
  // Minimize[
  //     quad334[primaryOptimizationTerm[c0, c1, c2, c3, c4, c5, c6, c7, c8,
  //                                     c9, c10, c11, c12, c13, c14, c15, c16,
  //                                     c17, c18, c19, c20, c21, c22, c23, c24,
  //                                     c25, c26, c27, c28, c29, c30, c31, c32,
  //                                     c33, c34, c35], 0, 0, 2]
  //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                      c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                      c35], -2, 0, 0] - quad334[uLowerXi, -2, 0, 0])^2
  //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                      c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                      c35], 2, 0, 0] - quad334[uUpperXi, 2, 0, 0])^2
  //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                      c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                      c35], 0, 2, 0] - quad334[uUpperEta, 0, 2, 0])^2
  //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                      c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                      c35], 0, 0, -2] - quad334[uLowerZeta, 0, 0, -2])^2,
  //     quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                   c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                   c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                   c35], 0, 0, 0] == quad334[uLocal, 0, 0, 0],
  // {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14,
  //  c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27,
  //  c28, c29, c30, c31, c32, c33, c34, c35}
  // ]
  const auto expected = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto& z = get<2>(logical_coords);
    constexpr std::array<double, 36> c{
        {7734190419582420551. / 62148673763526035437.,
         765121. / 1263364.,
         786240. / 1895041.,
         124800. / 1105441.,
         0.,
         0.,
         374400. / 1105441.,
         0.,
         0.,
         308447321809657439079. / 621486737635260354370.,
         -892944. / 1579205.,
         -6250608. / 9475205.,
         -878879. / 11054410.,
         0.,
         0.,
         -595296. / 1105441.,
         1. / 2.,
         0.,
         21514562148445593021. / 124297347527052070874.,
         89424. / 315841.,
         625968. / 1895041.,
         99360. / 1105441.,
         0.,
         0.,
         298080. / 1105441.,
         0.,
         0.,
         3655223967127444271. / 310743368817630177185.,
         -14256. / 315841.,
         -99792. / 1895041.,
         -15840. / 1105441.,
         0.,
         0.,
         -47520. / 1105441.,
         0.,
         0.}};
    const DataVector term_z0 = c[0] + c[1] * x + c[2] * square(x) +
                               y * (c[3] + c[4] * x + c[5] * square(x)) +
                               square(y) * (c[6] + c[7] * x + c[8] * square(x));
    const DataVector term_z1 =
        c[9] + c[10] * x + c[11] * square(x) +
        y * (c[12] + c[13] * x + c[14] * square(x)) +
        square(y) * (c[15] + c[16] * x + c[17] * square(x));
    const DataVector term_z2 =
        c[18] + c[19] * x + c[20] * square(x) +
        y * (c[21] + c[22] * x + c[23] * square(x)) +
        square(y) * (c[24] + c[25] * x + c[26] * square(x));
    const DataVector term_z3 =
        c[27] + c[28] * x + c[29] * square(x) +
        y * (c[30] + c[31] * x + c[32] * square(x)) +
        square(y) * (c[33] + c[34] * x + c[35] * square(x));
    return Scalar<DataVector>{
        {{term_z0 + term_z1 * z + term_z2 * square(z) + term_z3 * cube(z)}}};
  }
  ();

  // Fit procedure has somewhat larger error scale than default
  Approx local_approx = Approx::custom().epsilon(1e-8).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(modified_tensor, expected, local_approx);
  // Verify that the constraint is in fact satisfied
  CHECK(mean_value(get(modified_tensor), mesh) ==
        local_approx(mean_value(get(local_tensor), mesh)));
}

void test_hweno_modified_solution_2d_boundary() {
  INFO("Testing hweno_modified_neighbor_solution in 2D at boundary");
  using TagsList = tmpl::list<ScalarTag>;
  const Element<2> element{
      ElementId<2>{0},
      Element<2>::Neighbors_t{
          {Direction<2>::lower_xi(),
           TestHelpers::Limiters::make_neighbor_with_id<2>(1)},
          // upper_xi is external boundary
          {Direction<2>::lower_eta(),
           TestHelpers::Limiters::make_neighbor_with_id<2>(3)},
          {Direction<2>::upper_eta(),
           TestHelpers::Limiters::make_neighbor_with_id<2>(4)}}};
  const auto mesh = Mesh<2>{
      {{4, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  const auto lower_xi_neighbor =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>{1});
  const auto primary_neighbor =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>{3});
  const auto upper_eta_neighbor =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>{4});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    return Scalar<DataVector>{
        {{1. + 0.1 * x + 0.2 * y + 0.1 * x * y + 0.1 * x * square(y)}}};
  }
  ();

  const auto lower_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    // Large values to make sure this neighbor is excluded
    get(get<ScalarTag>(result)) = 1.e3;
    return result;
  }
  ();

  const auto primary_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) - 2.;
    get(get<ScalarTag>(result)) =
        1. + 0.25 * x - 0.2 * square(y) + 0.1 * x * square(y);
    return result;
  }
  ();

  const auto upper_eta_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) + 2.;
    get(get<ScalarTag>(result)) = 1. + 1. / 3. * x + 0.2 * y;
    return result;
  }
  ();

  const auto make_tuple_of_means = [&mesh](
      const Variables<TagsList>& vars) noexcept {
    return tuples::TaggedTuple<::Tags::Mean<ScalarTag>>(
        mean_value(get(get<ScalarTag>(vars)), mesh));
  };

  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>> means;
    Variables<TagsList> volume_data;
    Mesh<2> mesh;
  };
  std::unordered_map<std::pair<Direction<2>, ElementId<2>>, PackagedData,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  neighbor_data[lower_xi_neighbor].means = make_tuple_of_means(lower_xi_vars);
  neighbor_data[lower_xi_neighbor].volume_data = lower_xi_vars;
  neighbor_data[lower_xi_neighbor].mesh = mesh;
  neighbor_data[primary_neighbor].means = make_tuple_of_means(primary_vars);
  neighbor_data[primary_neighbor].volume_data = primary_vars;
  neighbor_data[primary_neighbor].mesh = mesh;
  neighbor_data[upper_eta_neighbor].means = make_tuple_of_means(upper_eta_vars);
  neighbor_data[upper_eta_neighbor].volume_data = upper_eta_vars;
  neighbor_data[upper_eta_neighbor].mesh = mesh;

  Scalar<DataVector> modified_tensor;
  Limiters::hweno_modified_neighbor_solution<ScalarTag>(
      make_not_null(&modified_tensor), local_tensor, element, mesh,
      neighbor_data, primary_neighbor);

  // The expected coefficient values for the result of the constrained fit are
  // found using Mathematica, using the following code (for Mathematica v10).
  // qw3 = {1/3, 4/3, 1/3};
  // qx3 = {-1, 0, 1};
  // qw4 = {1/6, 5/6, 5/6, 1/6};
  // qx4 = {-1, -1/Sqrt[5], 1/Sqrt[5], 1};
  // quad43[f_, dx_, dy_] := Sum[
  //     qw4[[qi]] qw3[[qj]] f[qx4[[qi]] + dx, qx3[[qj]] + dy],
  //     {qi, 1, 4}, {qj, 1, 3}];
  // trial[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_, c9_, c10_, c11_][
  //       x_, y_] :=
  //     c0 + c1 x + c2 x^2 + c3 x^3 +
  //     y (c4 + c5 x + c6 x^2 + c7 x^3) +
  //     y^2 (c8 + c9 x + c10 x^2 + c11 x^3);
  // uLocal[x_, y_] := 1 + 1/10 x + 1/5 y + 1/10 x y + 1/10 x y^2;
  // uPrimary[x_, y_] := 1 + 1/4 x - 1/5 y^2 + 1/10 x y^2;
  // uUpperEta[x_, y_] := 1 + 1/3 x + 1/5 y;
  // primaryOptimizationTerm[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_,
  //                         c9_, c10_, c11_][x_, y_] :=
  //     (trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11][x, y]
  //      - uPrimary[x, y])^2;
  // Minimize[
  //     quad43[primaryOptimizationTerm[c0, c1, c2, c3, c4, c5, c6, c7, c8,
  //                                    c9, c10, c11],
  //            0, -2]
  //     + (quad43[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
  //               0, 2] - quad43[uUpperEta, 0, 2])^2,
  //     quad43[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
  //            0, 0] == quad43[uLocal, 0, 0],
  // {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11}
  // ]
  const auto expected = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    constexpr std::array<double, 12> c{{50738. / 49647., 1. / 4., 0., 0.,
                                        27242. / 82745., 0., 0., 0.,
                                        -1091. / 16549., 1. / 10., 0., 0.}};
    return Scalar<DataVector>{
        {{c[0] + c[1] * x + c[2] * square(x) + c[3] * cube(x) +
          y * (c[4] + c[5] * x + c[6] * square(x) + c[7] * cube(x)) +
          square(y) *
              (c[8] + c[9] * x + c[10] * square(x) + c[11] * cube(x))}}};
  }
  ();

  // Fit procedure has somewhat larger error scale than default
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(modified_tensor, expected, local_approx);
  // Verify that the constraint is in fact satisfied
  CHECK(mean_value(get(modified_tensor), mesh) ==
        local_approx(mean_value(get(local_tensor), mesh)));
}

void test_hweno_modified_solution_2d_boundary_single_neighbor() {
  INFO("Testing hweno_modified_neighbor_solution in 2D at boundary");
  using TagsList = tmpl::list<ScalarTag>;
  const Element<2> element{
      ElementId<2>{0},
      Element<2>::Neighbors_t{
          {Direction<2>::lower_eta(),
           TestHelpers::Limiters::make_neighbor_with_id<2>(3)}}};
  const auto mesh = Mesh<2>{
      {{4, 3}}, Spectral::Basis::Legendre, Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  const auto primary_neighbor =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>{3});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    return Scalar<DataVector>{
        {{1. + 0.1 * x + 0.2 * y + 0.1 * x * y + 0.1 * x * square(y)}}};
  }
  ();

  const auto primary_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) - 2.;
    get(get<ScalarTag>(result)) =
        1. + 0.25 * x - 0.2 * square(y) + 0.1 * x * square(y);
    return result;
  }
  ();

  const auto make_tuple_of_means = [&mesh](
      const Variables<TagsList>& vars) noexcept {
    return tuples::TaggedTuple<::Tags::Mean<ScalarTag>>(
        mean_value(get(get<ScalarTag>(vars)), mesh));
  };

  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>> means;
    Variables<TagsList> volume_data;
    Mesh<2> mesh;
  };
  std::unordered_map<std::pair<Direction<2>, ElementId<2>>, PackagedData,
                     boost::hash<std::pair<Direction<2>, ElementId<2>>>>
      neighbor_data{};
  neighbor_data[primary_neighbor].means = make_tuple_of_means(primary_vars);
  neighbor_data[primary_neighbor].volume_data = primary_vars;
  neighbor_data[primary_neighbor].mesh = mesh;

  Scalar<DataVector> modified_tensor;
  Limiters::hweno_modified_neighbor_solution<ScalarTag>(
      make_not_null(&modified_tensor), local_tensor, element, mesh,
      neighbor_data, primary_neighbor);

  // The expected coefficient values for the result of the constrained fit are
  // found using Mathematica, using the following code (for Mathematica v10).
  // qw3 = {1/3, 4/3, 1/3};
  // qx3 = {-1, 0, 1};
  // qw4 = {1/6, 5/6, 5/6, 1/6};
  // qx4 = {-1, -1/Sqrt[5], 1/Sqrt[5], 1};
  // quad43[f_, dx_, dy_] := Sum[
  //     qw4[[qi]] qw3[[qj]] f[qx4[[qi]] + dx, qx3[[qj]] + dy],
  //     {qi, 1, 4}, {qj, 1, 3}];
  // trial[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_, c9_, c10_, c11_][
  //       x_, y_] :=
  //     c0 + c1 x + c2 x^2 + c3 x^3 +
  //     y (c4 + c5 x + c6 x^2 + c7 x^3) +
  //     y^2 (c8 + c9 x + c10 x^2 + c11 x^3);
  // uLocal[x_, y_] := 1 + 1/10 x + 1/5 y + 1/10 x y + 1/10 x y^2;
  // uPrimary[x_, y_] := 1 + 1/4 x - 1/5 y^2 + 1/10 x y^2;
  // primaryOptimizationTerm[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_,
  //                         c9_, c10_, c11_][x_, y_] :=
  //     (trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11][x, y]
  //      - uPrimary[x, y])^2;
  // Minimize[
  //     quad43[primaryOptimizationTerm[c0, c1, c2, c3, c4, c5, c6, c7, c8,
  //                                    c9, c10, c11],
  //            0, -2],
  //     quad43[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11],
  //            0, 0] == quad43[uLocal, 0, 0],
  // {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11}
  // ]
  const auto expected = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    constexpr std::array<double, 12> c{{1354. / 1275., 1. / 4., 0., 0.,
                                        26. / 425., 0., 0., 0., -79. / 425.,
                                        1. / 10., 0., 0.}};
    return Scalar<DataVector>{
        {{c[0] + c[1] * x + c[2] * square(x) + c[3] * cube(x) +
          y * (c[4] + c[5] * x + c[6] * square(x) + c[7] * cube(x)) +
          square(y) *
              (c[8] + c[9] * x + c[10] * square(x) + c[11] * cube(x))}}};
  }
  ();

  // Fit procedure has somewhat larger error scale than default
  Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(modified_tensor, expected, local_approx);
  // Verify that the constraint is in fact satisfied
  CHECK(mean_value(get(modified_tensor), mesh) ==
        local_approx(mean_value(get(local_tensor), mesh)));
}

void test_hweno_modified_solution_3d_boundary() {
  INFO("Testing hweno_modified_neighbor_solution in 3D at boundary");
  using TagsList = tmpl::list<ScalarTag>;
  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<ScalarTag>> means;
    Variables<TagsList> volume_data;
    Mesh<3> mesh;
  };

  const auto lower_xi_neighbor =
      std::make_pair(Direction<3>::lower_xi(), ElementId<3>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<3>::upper_xi(), ElementId<3>{2});
  const auto lower_eta_neighbor =
      std::make_pair(Direction<3>::lower_eta(), ElementId<3>{3});
  const auto upper_eta_neighbor =
      std::make_pair(Direction<3>::upper_eta(), ElementId<3>{4});
  const auto primary_neighbor =
      std::make_pair(Direction<3>::upper_zeta(), ElementId<3>{6});

  const Element<3> element{
      ElementId<3>{0},
      Element<3>::Neighbors_t{
          {Direction<3>::lower_xi(),
           TestHelpers::Limiters::make_neighbor_with_id<3>(1)},
          {Direction<3>::upper_xi(),
           TestHelpers::Limiters::make_neighbor_with_id<3>(2)},
          {Direction<3>::lower_eta(),
           TestHelpers::Limiters::make_neighbor_with_id<3>(3)},
          {Direction<3>::upper_eta(),
           TestHelpers::Limiters::make_neighbor_with_id<3>(4)},
          // lower_zeta is external boundary
          {Direction<3>::upper_zeta(),
           TestHelpers::Limiters::make_neighbor_with_id<3>(6)}}};
  const auto mesh = Mesh<3>{{{3, 3, 4}},
                            Spectral::Basis::Legendre,
                            Spectral::Quadrature::GaussLobatto};
  const auto logical_coords = logical_coordinates(mesh);

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto& z = get<2>(logical_coords);
    return Scalar<DataVector>{{{0.5 + 0.2 * x + 0.1 * square(y) * z}}};
  }
  ();

  const auto lower_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 1.2;
    return result;
  }
  ();

  const auto upper_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 4.;
    return result;
  }
  ();

  const auto lower_eta_vars = [&mesh]() noexcept {
    // Large values to make sure this neighbor is excluded
    Variables<TagsList> result(mesh.number_of_grid_points(), 1.e3);
    return result;
  }
  ();

  const auto upper_eta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 2.5;
    return result;
  }
  ();

  const auto primary_vars = [&mesh, &logical_coords ]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto z = get<2>(logical_coords) + 2.;
    get(get<ScalarTag>(result)) =
        1. + 0.25 * x + 0.1 * y * z + 0.5 * x * square(y) * z + 0.1 * cube(z);
    return result;
  }
  ();

  const auto make_tuple_of_means = [&mesh](
      const Variables<TagsList>& vars) noexcept {
    return tuples::TaggedTuple<::Tags::Mean<ScalarTag>>(
        mean_value(get(get<ScalarTag>(vars)), mesh));
  };

  std::unordered_map<std::pair<Direction<3>, ElementId<3>>, PackagedData,
                     boost::hash<std::pair<Direction<3>, ElementId<3>>>>
      neighbor_data{};
  neighbor_data[lower_xi_neighbor].means = make_tuple_of_means(lower_xi_vars);
  neighbor_data[lower_xi_neighbor].volume_data = lower_xi_vars;
  neighbor_data[lower_xi_neighbor].mesh = mesh;
  neighbor_data[upper_xi_neighbor].means = make_tuple_of_means(upper_xi_vars);
  neighbor_data[upper_xi_neighbor].volume_data = upper_xi_vars;
  neighbor_data[upper_xi_neighbor].mesh = mesh;
  neighbor_data[lower_eta_neighbor].means = make_tuple_of_means(lower_eta_vars);
  neighbor_data[lower_eta_neighbor].volume_data = lower_eta_vars;
  neighbor_data[lower_eta_neighbor].mesh = mesh;
  neighbor_data[upper_eta_neighbor].means = make_tuple_of_means(upper_eta_vars);
  neighbor_data[upper_eta_neighbor].volume_data = upper_eta_vars;
  neighbor_data[upper_eta_neighbor].mesh = mesh;
  neighbor_data[primary_neighbor].means = make_tuple_of_means(primary_vars);
  neighbor_data[primary_neighbor].volume_data = primary_vars;
  neighbor_data[primary_neighbor].mesh = mesh;

  Scalar<DataVector> modified_tensor;
  Limiters::hweno_modified_neighbor_solution<ScalarTag>(
      make_not_null(&modified_tensor), local_tensor, element, mesh,
      neighbor_data, primary_neighbor);

  // The expected coefficient values for the result of the constrained fit are
  // found using Mathematica, using the following code (for Mathematica v10).
  // qw3 = {1/3, 4/3, 1/3};
  // qx3 = {-1, 0, 1};
  // qw4 = {1/6, 5/6, 5/6, 1/6};
  // qx4 = {-1, -1/Sqrt[5], 1/Sqrt[5], 1};
  // quad334[f_, dx_, dy_, dz_] := Sum[
  //     qw3[[qi]] qw3[[qj]] qw4[[qk]]
  //     f[qx3[[qi]] + dx, qx3[[qj]] + dy, qx4[[qk]] + dz],
  //     {qi, 1, 3}, {qj, 1, 3}, {qk, 1, 4}];
  // trial[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_, c9_, c10_, c11_,
  //       c12_, c13_, c14_, c15_, c16_, c17_, c18_, c19_, c20_, c21_, c22_,
  //       c23_, c24_, c25_, c26_, c27_, c28_, c29_, c30_, c31_, c32_, c33_,
  //       c34_, c35_][x_, y_, z_] :=
  //     c0 + c1 x + c2 x^2 + y (c3 + c4 x + c5 x^2) +
  //                          y^2 (c6 + c7 x + c8 x^2) +
  //     z (c9 + c10 x + c11 x^2 + y (c12 + c13 x + c14 x^2) +
  //                               y^2 (c15 + c16 x + c17 x^2)) +
  //     z^2 (c18 + c19 x + c20 x^2 + y (c21 + c22 x + c23 x^2) +
  //                                  y^2 (c24 + c25 x + c26 x^2)) +
  //     z^3 (c27 + c28 x + c29 x^2 + y (c30 + c31 x + c32 x^2) +
  //                                  y^2 (c33 + c34 x + c35 x^2));
  // uLocal[x_, y_, z_] := 1/2 + 1/5 x + 1/10 y^2 z;
  // uPrimary[x_, y_, z_] := 1 + 1/4 x + 1/10 y z + 1/2 x y^2 z + 1/10 z^3;
  // uLowerXi[x_, y_, z_] := 6/5;
  // uUpperXi[x_, y_, z_] := 4;
  // uUpperEta[x_, y_, z_] := 5/2;
  // primaryOptimizationTerm[c0_, c1_, c2_, c3_, c4_, c5_, c6_, c7_, c8_,
  //                         c9_, c10_, c11_, c12_, c13_, c14_, c15_, c16_,
  //                         c17_, c18_, c19_, c20_, c21_, c22_, c23_, c24_,
  //                         c25_, c26_, c27_, c28_, c29_, c30_, c31_, c32_,
  //                         c33_, c34_, c35_][x_, y_, z_] :=
  //     (trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
  //            c12, c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //            c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34, c35][x,
  //            y, z] - uPrimary[x, y, z])^2;
  // Minimize[
  //     quad334[primaryOptimizationTerm[c0, c1, c2, c3, c4, c5, c6, c7, c8,
  //                                     c9, c10, c11, c12, c13, c14, c15, c16,
  //                                     c17, c18, c19, c20, c21, c22, c23, c24,
  //                                     c25, c26, c27, c28, c29, c30, c31, c32,
  //                                     c33, c34, c35], 0, 0, 2]
  //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                      c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                      c35], -2, 0, 0] - quad334[uLowerXi, -2, 0, 0])^2
  //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                      c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                      c35], 2, 0, 0] - quad334[uUpperXi, 2, 0, 0])^2
  //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                      c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                      c35], 0, 2, 0] - quad334[uUpperEta, 0, 2, 0])^2
  //     quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
  //                   c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
  //                   c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
  //                   c35], 0, 0, 0] == quad334[uLocal, 0, 0, 0],
  // {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14,
  //  c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27,
  //  c28, c29, c30, c31, c32, c33, c34, c35}
  // ]
  const auto expected = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto& z = get<2>(logical_coords);
    constexpr std::array<double, 36> c{{243751581645799. / 689207629948649.,
                                        765121. / 1263364.,
                                        786240. / 1895041.,
                                        124800. / 1105441.,
                                        0.,
                                        0.,
                                        374400. / 1105441.,
                                        0.,
                                        0.,
                                        1416550233603063. / 1378415259897298.,
                                        -892944. / 1579205.,
                                        -6250608. / 9475205.,
                                        -878879. / 11054410.,
                                        0.,
                                        0.,
                                        -595296. / 1105441.,
                                        1. / 2.,
                                        0.,
                                        -709303092297615. / 1378415259897298.,
                                        89424. / 315841.,
                                        625968. / 1895041.,
                                        99360. / 1105441.,
                                        0.,
                                        0.,
                                        298080. / 1105441.,
                                        0.,
                                        0.,
                                        627297076397287. / 3446038149743245.,
                                        -14256. / 315841.,
                                        -99792. / 1895041.,
                                        -15840. / 1105441.,
                                        0.,
                                        0.,
                                        -47520. / 1105441.,
                                        0.,
                                        0.}};
    const DataVector term_z0 = c[0] + c[1] * x + c[2] * square(x) +
                               y * (c[3] + c[4] * x + c[5] * square(x)) +
                               square(y) * (c[6] + c[7] * x + c[8] * square(x));
    const DataVector term_z1 =
        c[9] + c[10] * x + c[11] * square(x) +
        y * (c[12] + c[13] * x + c[14] * square(x)) +
        square(y) * (c[15] + c[16] * x + c[17] * square(x));
    const DataVector term_z2 =
        c[18] + c[19] * x + c[20] * square(x) +
        y * (c[21] + c[22] * x + c[23] * square(x)) +
        square(y) * (c[24] + c[25] * x + c[26] * square(x));
    const DataVector term_z3 =
        c[27] + c[28] * x + c[29] * square(x) +
        y * (c[30] + c[31] * x + c[32] * square(x)) +
        square(y) * (c[33] + c[34] * x + c[35] * square(x));
    return Scalar<DataVector>{
        {{term_z0 + term_z1 * z + term_z2 * square(z) + term_z3 * cube(z)}}};
  }
  ();

  // Fit procedure has somewhat larger error scale than default
  Approx local_approx = Approx::custom().epsilon(1e-8).scale(1.);
  CHECK_ITERABLE_CUSTOM_APPROX(modified_tensor, expected, local_approx);
  // Verify that the constraint is in fact satisfied
  CHECK(mean_value(get(modified_tensor), mesh) ==
        local_approx(mean_value(get(local_tensor), mesh)));
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.HwenoModifiedSolution",
                  "[Limiters][Unit]") {
  test_secondary_neighbors_to_exclude_from_fit();

  test_hweno_modified_solution_1d();
  test_hweno_modified_solution_2d_vector();
  test_hweno_modified_solution_2d_exclude_two_neighbors();
  test_hweno_modified_solution_3d();

  test_hweno_modified_solution_2d_boundary();
  test_hweno_modified_solution_2d_boundary_single_neighbor();
  test_hweno_modified_solution_3d_boundary();
}
