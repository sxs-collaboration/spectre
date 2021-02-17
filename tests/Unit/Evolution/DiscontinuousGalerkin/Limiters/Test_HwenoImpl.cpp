// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <array>
#include <boost/functional/hash.hpp>
#include <cstddef>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tags.hpp"  // IWYU pragma: keep
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "Domain/LogicalCoordinates.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/Element.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Neighbors.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/HwenoImpl.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoHelpers.hpp"
#include "Evolution/DiscontinuousGalerkin/Limiters/WenoOscillationIndicator.hpp"
#include "Helpers/Evolution/DiscontinuousGalerkin/Limiters/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearOperators/MeanValue.hpp"
#include "NumericalAlgorithms/Spectral/Mesh.hpp"
#include "NumericalAlgorithms/Spectral/Spectral.hpp"
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeArray.hpp"
#include "Utilities/StdHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

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

void test_secondary_neighbors_to_exclude_from_fit() noexcept {
  INFO("Testing Weno_detail::secondary_neighbors_to_exclude_from_fit");
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

  const auto check_excluded_neighbors =
      [&dummy_neighbor_data](
          const double mean,
          const std::pair<Direction<2>, ElementId<2>>& primary_neighbor,
          const std::unordered_set<
              std::pair<Direction<2>, ElementId<2>>,
              boost::hash<std::pair<Direction<2>, ElementId<2>>>>&
              expected_excluded_neighbors) noexcept {
        const size_t tensor_index = 0;
        const auto excluded_neighbors_vector =
            Limiters::Weno_detail::secondary_neighbors_to_exclude_from_fit<
                ScalarTag>(mean, tensor_index, dummy_neighbor_data,
                           primary_neighbor);
        // The elements of `excluded_neighbors_vector` are ordered in an
        // undefined way, because they are filled by looping over the
        // unordered_map of neighbor data. To provide meaningful test
        // comparisons, we move the data into an unordered_set. (A sort would
        // also work here, if the Direction and ElementId classes were sortable,
        // which they aren't.)
        const std::unordered_set<
            std::pair<Direction<2>, ElementId<2>>,
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

void test_constrained_fit_1d(const Spectral::Quadrature quadrature =
                                 Spectral::Quadrature::GaussLobatto) noexcept {
  INFO("Testing Weno_detail::solve_constrained_fit in 1D");
  CAPTURE(quadrature);
  using TagsList = tmpl::list<ScalarTag>;
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto mesh = Mesh<1>{{{3}}, Spectral::Basis::Legendre, quadrature};
  const auto logical_coords = logical_coordinates(mesh);

  const auto lower_xi_neighbor =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>{2});

  const auto local_data = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    return DataVector{1. - 0.2 * x + 0.4 * square(x)};
  }();

  const auto lower_xi_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) - 2.;
    get(get<ScalarTag>(result)) = 4. - 0.5 * x - 0.1 * square(x);
    return result;
  }();

  const auto upper_xi_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) + 2.;
    get(get<ScalarTag>(result)) = 1. - 0.2 * x + 0.1 * square(x);
    return result;
  }();

  const auto make_tuple_of_means =
      [&mesh](const Variables<TagsList>& vars) noexcept {
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
  neighbor_data[lower_xi_neighbor].means = make_tuple_of_means(lower_xi_vars);
  neighbor_data[lower_xi_neighbor].volume_data = lower_xi_vars;
  neighbor_data[lower_xi_neighbor].mesh = mesh;
  neighbor_data[upper_xi_neighbor].means = make_tuple_of_means(upper_xi_vars);
  neighbor_data[upper_xi_neighbor].volume_data = upper_xi_vars;
  neighbor_data[upper_xi_neighbor].mesh = mesh;

  // primary = lower_xi
  // excluded = upper_xi
  {
    INFO("One excluded neighbor");
    const auto primary_neighbor = lower_xi_neighbor;
    const std::vector<std::pair<Direction<1>, ElementId<1>>>
        neighbors_to_exclude = {upper_xi_neighbor};

    DataVector constrained_fit;
    Limiters::Weno_detail::solve_constrained_fit<ScalarTag>(
        make_not_null(&constrained_fit), local_data, 0, mesh, element,
        neighbor_data, primary_neighbor, neighbors_to_exclude);

    // The expected coefficient values for the result of the constrained fit are
    // found using Mathematica, using the following code (for Mathematica v10):
    // qw3 = {1/3, 4/3, 1/3};  (* or Gauss point equivalent *)
    // qx3 = {-1, 0, 1};       (* or Gauss point equivalent *)
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
    const auto expected = [&quadrature, &logical_coords]() noexcept {
      const auto& x = get<0>(logical_coords);
      const auto c =
          (quadrature == Spectral::Quadrature::GaussLobatto)
              ? std::array<double, 3>{{41. / 30., -31. / 10., -7. / 10.}}
              : std::array<double, 3>{
                    {803. / 579., -1247. / 386., -734. / 965.}};
      return DataVector{c[0] + c[1] * x + c[2] * square(x)};
    }();

    // Fit procedure has somewhat larger error scale than default
    Approx local_approx = Approx::custom().epsilon(1e-11).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(constrained_fit, expected, local_approx);
    // Verify that the constraint is in fact satisfied
    CHECK(mean_value(constrained_fit, mesh) ==
          local_approx(mean_value(local_data, mesh)));
  }

  // external = lower_xi
  // primary = upper_xi
  // excluded = {}
  {
    INFO("One external neighbor on lower_xi side");
    const Element<1> element_at_lower_xi_bdry{
        ElementId<1>{0},
        Element<1>::Neighbors_t{
            // lower_xi is external boundary
            {Direction<1>::upper_xi(),
             TestHelpers::Limiters::make_neighbor_with_id<1>(1)}}};
    auto neighbor_data_at_lower_xi_bdry = neighbor_data;
    neighbor_data_at_lower_xi_bdry.erase(lower_xi_neighbor);

    const auto primary_neighbor = upper_xi_neighbor;
    const std::vector<std::pair<Direction<1>, ElementId<1>>>
        neighbors_to_exclude = {};

    DataVector constrained_fit;
    Limiters::Weno_detail::solve_constrained_fit<ScalarTag>(
        make_not_null(&constrained_fit), local_data, 0, mesh,
        element_at_lower_xi_bdry, neighbor_data_at_lower_xi_bdry,
        primary_neighbor, neighbors_to_exclude);

    // Coefficients from Mathematica, using code similar to the one above.
    const auto expected = [&quadrature, &logical_coords]() noexcept {
      const auto& x = get<0>(logical_coords);
      const auto c =
          (quadrature == Spectral::Quadrature::GaussLobatto)
              ? std::array<double, 3>{{929. / 850., -124. / 425., 103. / 850.}}
              : std::array<double, 3>{
                    {1054. / 965., -286. / 965., 119. / 965.}};
      return DataVector{c[0] + c[1] * x + c[2] * square(x)};
    }();

    Approx local_approx = Approx::custom().epsilon(1e-11).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(constrained_fit, expected, local_approx);
    CHECK(mean_value(constrained_fit, mesh) ==
          local_approx(mean_value(local_data, mesh)));
  }

  // external = upper_xi
  // primary = lower_xi
  // excluded = {}
  {
    INFO("One external neighbor on upper_xi side");
    const Element<1> element_at_upper_xi_bdry{
        ElementId<1>{0},
        Element<1>::Neighbors_t{
            {Direction<1>::lower_xi(),
             TestHelpers::Limiters::make_neighbor_with_id<1>(1)}
            // upper_xi is external boundary
        }};
    auto neighbor_data_at_upper_xi_bdry = neighbor_data;
    neighbor_data_at_upper_xi_bdry.erase(upper_xi_neighbor);

    const auto primary_neighbor = lower_xi_neighbor;
    const std::vector<std::pair<Direction<1>, ElementId<1>>>
        neighbors_to_exclude = {};

    DataVector constrained_fit;
    Limiters::Weno_detail::solve_constrained_fit<ScalarTag>(
        make_not_null(&constrained_fit), local_data, 0, mesh,
        element_at_upper_xi_bdry, neighbor_data_at_upper_xi_bdry,
        primary_neighbor, neighbors_to_exclude);

    // This test case should produce the same fit as the first test case above,
    // because in both cases the lower_xi neighbor is NOT part of the fit.
    // In the earlier test case, the lower_xi neighbor was in the domain but was
    // excluded from the fit; here, it is not even in the domain. So while this
    // test is not an orthogonal test of the fitting itself, it is a useful test
    // of the caching mechanism in the corner case of having a single
    // neighboring element.
    const auto expected = [&quadrature, &logical_coords]() noexcept {
      const auto& x = get<0>(logical_coords);
      const auto c =
          (quadrature == Spectral::Quadrature::GaussLobatto)
              ? std::array<double, 3>{{41. / 30., -31. / 10., -7. / 10.}}
              : std::array<double, 3>{
                    {803. / 579., -1247. / 386., -734. / 965.}};
      return DataVector{c[0] + c[1] * x + c[2] * square(x)};
    }();

    Approx local_approx = Approx::custom().epsilon(1e-11).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(constrained_fit, expected, local_approx);
    CHECK(mean_value(constrained_fit, mesh) ==
          local_approx(mean_value(local_data, mesh)));
  }
}

// Test in 2D using a vector tensor, to test multiple components.
// Multiple components becomes very tedious in 3D, so 3D will test a scalar.
void test_constrained_fit_2d_vector(
    const Spectral::Quadrature quadrature =
        Spectral::Quadrature::GaussLobatto) noexcept {
  INFO("Testing Weno_detail::solve_constrained_fit in 2D");
  CAPTURE(quadrature);
  using TagsList = tmpl::list<VectorTag<2>>;
  const auto element = TestHelpers::Limiters::make_element<2>();
  const auto mesh = Mesh<2>{{{4, 3}}, Spectral::Basis::Legendre, quadrature};
  const auto logical_coords = logical_coordinates(mesh);

  const auto lower_xi_neighbor =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<2>::upper_xi(), ElementId<2>{2});
  const auto lower_eta_neighbor =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>{3});
  const auto upper_eta_neighbor =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>{4});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    return VectorTag<2>::type{{{DataVector{1. + 0.1 * x + 0.2 * y +
                                           0.1 * x * y + 0.1 * x * square(y)},
                                DataVector(x.size(), 2.)}}};
  }();

  const auto lower_xi_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) - 2.;
    const auto& y = get<1>(logical_coords);
    get<0>(get<VectorTag<2>>(result)) = 2. + 0.2 * x - 0.1 * y;
    get<1>(get<VectorTag<2>>(result)) = 1.;
    return result;
  }();

  const auto upper_xi_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) + 2.;
    const auto& y = get<1>(logical_coords);
    get<0>(get<VectorTag<2>>(result)) =
        1. + 1. / 3. * x + 0.25 * y - 0.05 * square(x);
    get<1>(get<VectorTag<2>>(result)) = -0.5;
    return result;
  }();

  const auto lower_eta_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) - 2.;
    get<0>(get<VectorTag<2>>(result)) =
        1. + 0.25 * x - 0.2 * square(y) + 0.1 * x * square(y);
    get<1>(get<VectorTag<2>>(result)) =
        1.2 + 0.5 * x - 0.1 * square(x) - 0.2 * y + 0.1 * square(x) * square(y);
    return result;
  }();

  const auto upper_eta_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) + 2.;
    get<0>(get<VectorTag<2>>(result)) = 1. + 1. / 3. * x + 0.2 * y;
    get<1>(get<VectorTag<2>>(result)) = 0.1;
    return result;
  }();

  const auto make_tuple_of_means =
      [&mesh](const Variables<TagsList>& vars) noexcept {
        return tuples::TaggedTuple<::Tags::Mean<VectorTag<2>>>(
            tnsr::I<double, 2>{
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
  neighbor_data[lower_eta_neighbor].means = make_tuple_of_means(lower_eta_vars);
  neighbor_data[lower_eta_neighbor].volume_data = lower_eta_vars;
  neighbor_data[lower_eta_neighbor].mesh = mesh;
  neighbor_data[upper_eta_neighbor].means = make_tuple_of_means(upper_eta_vars);
  neighbor_data[upper_eta_neighbor].volume_data = upper_eta_vars;
  neighbor_data[upper_eta_neighbor].mesh = mesh;

  // primary = lower_eta
  // excluded = lower_xi
  {
    INFO("One excluded neighbor");
    const auto primary_neighbor = lower_eta_neighbor;
    // In realistic uses, the calls to solve_constrained_fit for different
    // tensor component would have different excluded neighbors. But for the
    // test of the constrained fit itself, this is not important, and so we
    // simplify by using the same excluded neighbors for each component.
    const std::vector<std::pair<Direction<2>, ElementId<2>>>
        neighbors_to_exclude = {lower_xi_neighbor};

    tnsr::I<DataVector, 2> constrained_fit{};
    for (size_t tensor_index = 0; tensor_index < 2; ++tensor_index) {
      Limiters::Weno_detail::solve_constrained_fit<VectorTag<2>>(
          make_not_null(&(constrained_fit.get(tensor_index))),
          local_tensor.get(tensor_index), tensor_index, mesh, element,
          neighbor_data, primary_neighbor, neighbors_to_exclude);
    }

    // The expected coefficient values for the result of the constrained fit are
    // found using Mathematica, using the following code (for Mathematica v10).
    // This example computes the expected result for the vector x component; an
    // analogous piece of code gives the expected result for the y component,
    // but note that the neighbor to exclude changes in this case.
    // qw3 = {1/3, 4/3, 1/3};                 (* or Gauss point equivalent *)
    // qx3 = {-1, 0, 1};                      (* or Gauss point equivalent *)
    // qw4 = {1/6, 5/6, 5/6, 1/6};            (* or Gauss point equivalent *)
    // qx4 = {-1, -1/Sqrt[5], 1/Sqrt[5], 1};  (* or Gauss point equivalent *)
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
    const auto expected = [&quadrature, &logical_coords]() noexcept {
      const auto& x = get<0>(logical_coords);
      const auto& y = get<1>(logical_coords);
      // x-component coefficients
      const auto c =
          (quadrature == Spectral::Quadrature::GaussLobatto)
              ? std::array<double, 12>{{6049013. / 5913219., 92367. / 360620.,
                                        -1659. / 558961., -6083. / 558961.,
                                        61831556. / 187251935., 42. / 6935.,
                                        -126. / 42997., -462. / 42997.,
                                        -2460491. / 37450387., 18283. / 180310.,
                                        -378. / 558961., -1386. / 558961.}}
              : std::array<double, 12>{
                    {249938816573. / 244480323945., 405503. / 1579292.,
                     -534. / 394823., -1246. / 107679.,
                     26841345508. / 81493441315., 2790. / 394823.,
                     -558. / 394823., -434. / 35893.,
                     -5338984133. / 81493441315., 401573. / 3948230.,
                     -135. / 394823., -105. / 35893.}};
      // y-component coefficients
      const auto d =
          (quadrature == Spectral::Quadrature::GaussLobatto)
              ? std::array<double, 12>{{3609701941. / 1685267415.,
                                        120807. / 180310., -1018741. / 5589610.,
                                        -168586. / 558961.,
                                        -18411919. / 43211985., 1164. / 6935.,
                                        -3492. / 42997., -12804. / 42997.,
                                        -50666458. / 187251935., 3492. / 90155.,
                                        454201. / 5589610., -38412. / 558961.}}
              : std::array<double, 12>{
                    {10921212941666. / 5134086802845., 3799721. / 5527522.,
                     -3799721. / 27637610., -34532. / 107679.,
                     -751998511637. / 1711362267615., 541260. / 2763761.,
                     -108252. / 2763761., -12028. / 35893.,
                     -31292971603. / 114090817841., 130950. / 2763761.,
                     2501861. / 27637610., -2910. / 35893.}};
      return tnsr::I<DataVector, 2>{
          {{DataVector{
                c[0] + c[1] * x + c[2] * square(x) + c[3] * cube(x) +
                y * (c[4] + c[5] * x + c[6] * square(x) + c[7] * cube(x)) +
                square(y) *
                    (c[8] + c[9] * x + c[10] * square(x) + c[11] * cube(x))},
            DataVector{
                d[0] + d[1] * x + d[2] * square(x) + d[3] * cube(x) +
                y * (d[4] + d[5] * x + d[6] * square(x) + d[7] * cube(x)) +
                square(y) *
                    (d[8] + d[9] * x + d[10] * square(x) + d[11] * cube(x))}}}};
    }();

    // Fit procedure has somewhat larger error scale than default
    Approx local_approx = Approx::custom().epsilon(1e-10).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(constrained_fit, expected, local_approx);
    // Verify that the constraint is in fact satisfied
    CHECK(mean_value(get<0>(constrained_fit), mesh) ==
          local_approx(mean_value(get<0>(local_tensor), mesh)));
    CHECK(mean_value(get<1>(constrained_fit), mesh) ==
          local_approx(mean_value(get<1>(local_tensor), mesh)));
  }

  // external = lower_eta
  // primary = lower_xi
  // excluded = {upper_xi, upper_eta}
  {
    INFO("One external, two excluded neighbors");
    const Element<2> element_at_lower_eta_bdry{
        ElementId<2>{0},
        Element<2>::Neighbors_t{
            {Direction<2>::lower_xi(),
             TestHelpers::Limiters::make_neighbor_with_id<2>(0)},
            {Direction<2>::upper_xi(),
             TestHelpers::Limiters::make_neighbor_with_id<2>(1)},
            // lower_eta is external boundary
            {Direction<2>::upper_eta(),
             TestHelpers::Limiters::make_neighbor_with_id<2>(4)}}};
    auto neighbor_data_at_lower_eta_bdry = neighbor_data;
    neighbor_data_at_lower_eta_bdry.erase(lower_eta_neighbor);

    const auto primary_neighbor = lower_xi_neighbor;
    const std::vector<std::pair<Direction<2>, ElementId<2>>>
        neighbors_to_exclude = {upper_xi_neighbor, upper_eta_neighbor};

    tnsr::I<DataVector, 2> constrained_fit{};
    for (size_t tensor_index = 0; tensor_index < 2; ++tensor_index) {
      Limiters::Weno_detail::solve_constrained_fit<VectorTag<2>>(
          make_not_null(&(constrained_fit.get(tensor_index))),
          local_tensor.get(tensor_index), tensor_index, mesh,
          element_at_lower_eta_bdry, neighbor_data_at_lower_eta_bdry,
          primary_neighbor, neighbors_to_exclude);
    }

    // Coefficients from Mathematica, using code similar to the one above.
    const auto expected = [&quadrature, &logical_coords]() noexcept {
      const auto& x = get<0>(logical_coords);
      const auto& y = get<1>(logical_coords);
      const auto c =
          (quadrature == Spectral::Quadrature::GaussLobatto)
              ? std::array<double, 12>{{398. / 329., -1738. / 1645.,
                                        -207. / 329., -33. / 329., -1. / 10.,
                                        0., 0., 0., 0., 0., 0., 0.}}
              : std::array<double, 12>{{4366. / 3581., -19294. / 17905.,
                                        -2355. / 3581., -385. / 3581.,
                                        -1. / 10., 0., 0., 0., 0., 0., 0., 0.}};
      const auto d = (quadrature == Spectral::Quadrature::GaussLobatto)
                         ? std::array<double, 12>{{589. / 329., 2067. / 1645.,
                                                   207. / 329., 33. / 329., 0.,
                                                   0., 0., 0., 0., 0., 0., 0.}}
                         : std::array<double, 12>{
                               {6377. / 3581., 4575. / 3581., 2355. / 3581.,
                                385. / 3581., 0., 0., 0., 0., 0., 0., 0., 0.}};
      return tnsr::I<DataVector, 2>{
          {{DataVector{
                c[0] + c[1] * x + c[2] * square(x) + c[3] * cube(x) +
                y * (c[4] + c[5] * x + c[6] * square(x) + c[7] * cube(x)) +
                square(y) *
                    (c[8] + c[9] * x + c[10] * square(x) + c[11] * cube(x))},
            DataVector{
                d[0] + d[1] * x + d[2] * square(x) + d[3] * cube(x) +
                y * (d[4] + d[5] * x + d[6] * square(x) + d[7] * cube(x)) +
                square(y) *
                    (d[8] + d[9] * x + d[10] * square(x) + d[11] * cube(x))}}}};
    }();

    // Fit procedure has somewhat larger error scale than default
    Approx local_approx = Approx::custom().epsilon(1e-9).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(constrained_fit, expected, local_approx);
    // Verify that the constraint is in fact satisfied
    CHECK(mean_value(get<0>(constrained_fit), mesh) ==
          local_approx(mean_value(get<0>(local_tensor), mesh)));
    CHECK(mean_value(get<1>(constrained_fit), mesh) ==
          local_approx(mean_value(get<1>(local_tensor), mesh)));
  }
}

void test_constrained_fit_3d(const Spectral::Quadrature quadrature =
                                 Spectral::Quadrature::GaussLobatto) noexcept {
  INFO("Testing Weno_detail::solve_constrained_fit in 3D");
  CAPTURE(quadrature);
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
  const auto upper_zeta_neighbor =
      std::make_pair(Direction<3>::upper_zeta(), ElementId<3>{6});

  const auto element = TestHelpers::Limiters::make_element<3>();
  const auto mesh = Mesh<3>{{{3, 3, 4}}, Spectral::Basis::Legendre, quadrature};
  const auto logical_coords = logical_coordinates(mesh);

  const auto local_data = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto& z = get<2>(logical_coords);
    return DataVector{0.5 + 0.2 * x + 0.1 * square(y) * z};
  }();

  const auto lower_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 1.2;
    return result;
  }();

  const auto upper_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 4.;
    return result;
  }();

  const auto lower_eta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 3.;
    return result;
  }();

  const auto upper_eta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 2.5;
    return result;
  }();

  const auto lower_zeta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get(get<ScalarTag>(result)) = 2.;
    return result;
  }();

  const auto upper_zeta_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto z = get<2>(logical_coords) + 2.;
    get(get<ScalarTag>(result)) =
        1. + 0.25 * x + 0.1 * y * z + 0.5 * x * square(y) * z + 0.1 * cube(z);
    return result;
  }();

  const auto make_tuple_of_means =
      [&mesh](const Variables<TagsList>& vars) noexcept {
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
  neighbor_data[upper_zeta_neighbor].means =
      make_tuple_of_means(upper_zeta_vars);
  neighbor_data[upper_zeta_neighbor].volume_data = upper_zeta_vars;
  neighbor_data[upper_zeta_neighbor].mesh = mesh;

  // primary = upper_zeta
  // excluded = lower_eta
  {
    INFO("One excluded neighbor");
    const auto primary_neighbor = upper_zeta_neighbor;
    const std::vector<std::pair<Direction<3>, ElementId<3>>>
        neighbors_to_exclude = {lower_eta_neighbor};

    DataVector constrained_fit;
    Limiters::Weno_detail::solve_constrained_fit<ScalarTag>(
        make_not_null(&constrained_fit), local_data, 0, mesh, element,
        neighbor_data, primary_neighbor, neighbors_to_exclude);

    // The expected coefficient values for the result of the constrained fit are
    // found using Mathematica, using the following code (for Mathematica v10).
    // qw3 = {1/3, 4/3, 1/3};                 (* or Gauss point equivalent *)
    // qx3 = {-1, 0, 1};                      (* or Gauss point equivalent *)
    // qw4 = {1/6, 5/6, 5/6, 1/6};            (* or Gauss point equivalent *)
    // qx4 = {-1, -1/Sqrt[5], 1/Sqrt[5], 1};  (* or Gauss point equivalent *)
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
    //                                     c9, c10, c11, c12, c13, c14, c15,
    //                                     c16, c17, c18, c19, c20, c21, c22,
    //                                     c23, c24, c25, c26, c27, c28, c29,
    //                                     c30, c31, c32, c33, c34, c35], 0, 0,
    //                                     2]
    //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
    //     c12,
    //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22,
    //                      c23, c24, c25, c26, c27, c28, c29, c30, c31, c32,
    //                      c33, c34, c35], -2, 0, 0] - quad334[uLowerXi, -2, 0,
    //                      0])^2
    //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
    //     c12,
    //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22,
    //                      c23, c24, c25, c26, c27, c28, c29, c30, c31, c32,
    //                      c33, c34, c35], 2, 0, 0] - quad334[uUpperXi, 2, 0,
    //                      0])^2
    //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
    //     c12,
    //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22,
    //                      c23, c24, c25, c26, c27, c28, c29, c30, c31, c32,
    //                      c33, c34, c35], 0, 2, 0] - quad334[uUpperEta, 0, 2,
    //                      0])^2
    //     + (quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11,
    //     c12,
    //                      c13, c14, c15, c16, c17, c18, c19, c20, c21, c22,
    //                      c23, c24, c25, c26, c27, c28, c29, c30, c31, c32,
    //                      c33, c34, c35], 0, 0, -2] - quad334[uLowerZeta, 0,
    //                      0, -2])^2,
    //     quad334[trial[c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12,
    //                   c13, c14, c15, c16, c17, c18, c19, c20, c21, c22, c23,
    //                   c24, c25, c26, c27, c28, c29, c30, c31, c32, c33, c34,
    //                   c35], 0, 0, 0] == quad334[uLocal, 0, 0, 0],
    // {c0, c1, c2, c3, c4, c5, c6, c7, c8, c9, c10, c11, c12, c13, c14,
    //  c15, c16, c17, c18, c19, c20, c21, c22, c23, c24, c25, c26, c27,
    //  c28, c29, c30, c31, c32, c33, c34, c35}
    // ]
    const auto expected = [&quadrature, &logical_coords]() noexcept {
      const auto& x = get<0>(logical_coords);
      const auto& y = get<1>(logical_coords);
      const auto& z = get<2>(logical_coords);
      const auto c =
          (quadrature == Spectral::Quadrature::GaussLobatto)
              ? std::array<double,
                           36>{{7734190419582420551. / 62148673763526035437.,
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
                                0.}}
              : std::array<double, 36>{
                    {458679916884839601803. / 3920643646799006444317.,
                     8269253. / 13751060.,
                     4227552. / 10313281.,
                     268416. / 5500417.,
                     0.,
                     0.,
                     2013120. / 5500417.,
                     0.,
                     0.,
                     4086360148702064037975. / 7841287293598012888634.,
                     -395280. / 687553.,
                     -6917400. / 10313281.,
                     1108417. / 55004170.,
                     0.,
                     0.,
                     -3294000. / 5500417.,
                     1. / 2.,
                     0.,
                     1217573258478400072485. / 7841287293598012888634.,
                     203472. / 687553.,
                     3560760. / 10313281.,
                     226080. / 5500417.,
                     0.,
                     0.,
                     1695600. / 5500417.,
                     0.,
                     0.,
                     3895938147037532473. / 254587249792143275605.,
                     -33264. / 687553.,
                     -52920. / 937571.,
                     -36960. / 5500417.,
                     0.,
                     0.,
                     -277200. / 5500417.,
                     0.,
                     0.}};
      const DataVector term_z0 =
          c[0] + c[1] * x + c[2] * square(x) +
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
      return DataVector{term_z0 + term_z1 * z + term_z2 * square(z) +
                        term_z3 * cube(z)};
    }();

    // Fit procedure has somewhat larger error scale than default
    Approx local_approx = Approx::custom().epsilon(1e-8).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(constrained_fit, expected, local_approx);
    // Verify that the constraint is in fact satisfied
    CHECK(mean_value(constrained_fit, mesh) ==
          local_approx(mean_value(local_data, mesh)));
  }

  // external = {lower_xi, lower_eta}
  // primary = upper_xi
  // excluded = {upper_eta, upper_zeta}
  {
    INFO("Two external, two excluded neighbors");
    const Element<3> element_two_bdries{
        ElementId<3>{0},
        Element<3>::Neighbors_t{
            // lower_xi is external boundary
            {Direction<3>::upper_xi(),
             TestHelpers::Limiters::make_neighbor_with_id<3>(1)},
            // lower_eta is external boundary
            {Direction<3>::upper_eta(),
             TestHelpers::Limiters::make_neighbor_with_id<3>(4)},
            {Direction<3>::lower_zeta(),
             TestHelpers::Limiters::make_neighbor_with_id<3>(5)},
            {Direction<3>::upper_zeta(),
             TestHelpers::Limiters::make_neighbor_with_id<3>(6)}}};
    auto neighbor_data_two_bdries = neighbor_data;
    neighbor_data_two_bdries.erase(lower_xi_neighbor);
    neighbor_data_two_bdries.erase(lower_eta_neighbor);

    const auto primary_neighbor = upper_xi_neighbor;
    const std::vector<std::pair<Direction<3>, ElementId<3>>>
        neighbors_to_exclude = {upper_eta_neighbor, upper_zeta_neighbor};

    DataVector constrained_fit;
    Limiters::Weno_detail::solve_constrained_fit<ScalarTag>(
        make_not_null(&constrained_fit), local_data, 0, mesh,
        element_two_bdries, neighbor_data_two_bdries, primary_neighbor,
        neighbors_to_exclude);

    // Coefficients from Mathematica, using code similar to the one above.
    const auto expected = [&quadrature, &logical_coords]() noexcept {
      const auto& x = get<0>(logical_coords);
      const auto& y = get<1>(logical_coords);
      const auto& z = get<2>(logical_coords);
      auto c = make_array<36>(0.);
      if (quadrature == Spectral::Quadrature::GaussLobatto) {
        c[0] = 139558567. / 190046570.;
        c[1] = 306385833. / 95023285.;
        c[2] = -70704423. / 95023285.;
        c[9] = 88164. / 1117921.;
        c[10] = -87048. / 1117921.;
        c[11] = 20088. / 1117921.;
        c[18] = 42660. / 1117921.;
        c[19] = -42120. / 1117921.;
        c[20] = 9720. / 1117921.;
        c[27] = -156420. / 1117921.;
        c[28] = 154440. / 1117921.;
        c[29] = -35640. / 1117921.;
      } else {
        c[0] = 90824101. / 118534617.;
        c[1] = 133513993. / 39511539.;
        c[2] = -21534515. / 26341026.;
        c[9] = 17800. / 204723.;
        c[10] = -6200. / 68241.;
        c[11] = 500. / 22747.;
        c[18] = 3560. / 204723.;
        c[19] = -1240. / 68241.;
        c[20] = 100. / 22747.;
        c[27] = -274120. / 1842507.;
        c[28] = 95480. / 614169.;
        c[29] = -7700. / 204723.;
      }

      const DataVector term_z0 =
          c[0] + c[1] * x + c[2] * square(x) +
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
      return DataVector{term_z0 + term_z1 * z + term_z2 * square(z) +
                        term_z3 * cube(z)};
    }();

    // Fit procedure has somewhat larger error scale than default
    Approx local_approx = Approx::custom().epsilon(1e-8).scale(1.);
    CHECK_ITERABLE_CUSTOM_APPROX(constrained_fit, expected, local_approx);
    // Verify that the constraint is in fact satisfied
    CHECK(mean_value(constrained_fit, mesh) ==
          local_approx(mean_value(local_data, mesh)));
  }
}

template <size_t VolumeDim>
void test_hweno_work(
    const tnsr::I<DataVector, VolumeDim>& local_vector,
    const Mesh<VolumeDim>& mesh, const Element<VolumeDim>& element,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        Variables<tmpl::list<VectorTag<VolumeDim>>>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        neighbor_vars,
    const std::unordered_map<
        std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>,
        std::array<
            std::vector<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>,
            VolumeDim>,
        boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>&
        expected_excluded_neighbors,
    Approx local_approx = approx) noexcept {
  struct PackagedData {
    tuples::TaggedTuple<::Tags::Mean<VectorTag<VolumeDim>>> means;
    Variables<tmpl::list<VectorTag<VolumeDim>>> volume_data;
    Mesh<VolumeDim> mesh;
  };

  const auto make_tuple_of_means =
      [&mesh](
          const Variables<tmpl::list<VectorTag<VolumeDim>>>& vars) noexcept {
        tuples::TaggedTuple<::Tags::Mean<VectorTag<VolumeDim>>> result(
            tnsr::I<double, VolumeDim>{});
        for (size_t i = 0; i < VolumeDim; ++i) {
          get<::Tags::Mean<VectorTag<VolumeDim>>>(result).get(i) =
              mean_value(get<VectorTag<VolumeDim>>(vars).get(i), mesh);
        }
        return result;
      };

  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, PackagedData,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      neighbor_data{};
  for (auto& neighbor_and_vars : neighbor_vars) {
    const auto& neighbor = neighbor_and_vars.first;
    const auto& vars = neighbor_and_vars.second;
    neighbor_data[neighbor].means = make_tuple_of_means(vars);
    neighbor_data[neighbor].volume_data = vars;
    neighbor_data[neighbor].mesh = mesh;
  }

  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      modified_neighbor_solution_buffer{};
  for (const auto& neighbor_and_data : neighbor_data) {
    const auto& neighbor = neighbor_and_data.first;
    modified_neighbor_solution_buffer.insert(
        make_pair(neighbor, DataVector(mesh.number_of_grid_points())));
  }

  auto vector_to_limit = local_vector;
  const double neighbor_linear_weight = 0.001;
  Limiters::Weno_detail::hweno_impl<VectorTag<VolumeDim>>(
      make_not_null(&modified_neighbor_solution_buffer),
      make_not_null(&vector_to_limit), neighbor_linear_weight, mesh, element,
      neighbor_data);

  // Check data mean was preserved
  for (size_t i = 0; i < VolumeDim; ++i) {
    CHECK(mean_value(vector_to_limit.get(i), mesh) ==
          local_approx(mean_value(local_vector.get(i), mesh)));
  }

  auto expected_hweno = local_vector;
  std::unordered_map<
      std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>, DataVector,
      boost::hash<std::pair<Direction<VolumeDim>, ElementId<VolumeDim>>>>
      expected_neighbor_polynomials;
  for (size_t i = 0; i < VolumeDim; ++i) {
    // Call solve_constrained_fit to make the expected neighbors for each
    // component of the vector
    for (auto& neighbor_and_excluded : expected_excluded_neighbors) {
      const auto& primary_neighbor = neighbor_and_excluded.first;
      const auto& neighbors_to_exclude =
          gsl::at(neighbor_and_excluded.second, i);

      DataVector& constrained_fit =
          expected_neighbor_polynomials[primary_neighbor];
      Limiters::Weno_detail::solve_constrained_fit<VectorTag<VolumeDim>>(
          make_not_null(&constrained_fit), local_vector.get(i), i, mesh,
          element, neighbor_data, primary_neighbor, neighbors_to_exclude);
    }
    Limiters::Weno_detail::reconstruct_from_weighted_sum(
        make_not_null(&expected_hweno.get(i)), neighbor_linear_weight,
        Limiters::Weno_detail::DerivativeWeight::PowTwoEllOverEllFactorial,
        mesh, expected_neighbor_polynomials);
  }

  // Check limited fields as expected
  CHECK_ITERABLE_CUSTOM_APPROX(vector_to_limit, expected_hweno, local_approx);
}

void test_hweno_impl_1d(const Spectral::Quadrature quadrature =
                            Spectral::Quadrature::GaussLobatto) noexcept {
  INFO("Testing hweno_impl in 1D");
  CAPTURE(quadrature);
  using TagsList = tmpl::list<VectorTag<1>>;
  const auto mesh = Mesh<1>{{{3}}, Spectral::Basis::Legendre, quadrature};
  const auto element = TestHelpers::Limiters::make_element<1>();
  const auto logical_coords = logical_coordinates(mesh);

  const auto lower_xi_neighbor =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>{2});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    return VectorTag<1>::type{{{DataVector{1. + 2.1 * x + 0.3 * square(x)}}}};
  }();

  const auto lower_xi_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) - 2.;
    get<0>(get<VectorTag<1>>(result)) = 4. - 0.5 * x - 0.1 * square(x);
    return result;
  }();

  const auto upper_xi_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) + 2.;
    get<0>(get<VectorTag<1>>(result)) = 1. - 0.2 * x + 0.1 * square(x);
    return result;
  }();

  Approx local_approx = Approx::custom().epsilon(1e-11).scale(1.);
  test_hweno_work<1>(
      local_tensor, mesh, element,
      {std::make_pair(lower_xi_neighbor, lower_xi_vars),
       std::make_pair(upper_xi_neighbor, upper_xi_vars)},
      {std::make_pair(
           lower_xi_neighbor,
           make_array<1>(std::vector<std::pair<Direction<1>, ElementId<1>>>{
               upper_xi_neighbor})),
       std::make_pair(
           upper_xi_neighbor,
           make_array<1>(std::vector<std::pair<Direction<1>, ElementId<1>>>{
               lower_xi_neighbor}))},
      local_approx);
}

void test_hweno_impl_2d(const Spectral::Quadrature quadrature =
                            Spectral::Quadrature::GaussLobatto) noexcept {
  INFO("Testing hweno_impl in 2D");
  CAPTURE(quadrature);
  using TagsList = tmpl::list<VectorTag<2>>;
  const auto mesh = Mesh<2>{{{4, 3}}, Spectral::Basis::Legendre, quadrature};
  const auto element = TestHelpers::Limiters::make_element<2>();
  const auto logical_coords = logical_coordinates(mesh);

  const auto lower_xi_neighbor =
      std::make_pair(Direction<2>::lower_xi(), ElementId<2>{1});
  const auto upper_xi_neighbor =
      std::make_pair(Direction<2>::upper_xi(), ElementId<2>{2});
  const auto lower_eta_neighbor =
      std::make_pair(Direction<2>::lower_eta(), ElementId<2>{3});
  const auto upper_eta_neighbor =
      std::make_pair(Direction<2>::upper_eta(), ElementId<2>{4});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    return VectorTag<2>::type{{{DataVector{1. + 0.1 * x + 0.2 * y +
                                           0.1 * x * y + 0.1 * x * square(y)},
                                DataVector(x.size(), 2.)}}};
  }();

  const auto lower_xi_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) - 2.;
    const auto& y = get<1>(logical_coords);
    get<0>(get<VectorTag<2>>(result)) = 2. + 0.2 * x - 0.1 * y;
    get<1>(get<VectorTag<2>>(result)) = 1.;
    return result;
  }();

  const auto upper_xi_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto x = get<0>(logical_coords) + 2.;
    const auto& y = get<1>(logical_coords);
    get<0>(get<VectorTag<2>>(result)) =
        1. + 1. / 3. * x + 0.25 * y - 0.05 * square(x);
    get<1>(get<VectorTag<2>>(result)) = -0.5;
    return result;
  }();

  const auto lower_eta_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) - 2.;
    get<0>(get<VectorTag<2>>(result)) =
        1. + 0.25 * x - 0.2 * square(y) + 0.1 * x * square(y);
    get<1>(get<VectorTag<2>>(result)) =
        1.2 + 0.5 * x - 0.1 * square(x) - 0.2 * y + 0.1 * square(x) * square(y);
    return result;
  }();

  const auto upper_eta_vars = [&mesh, &logical_coords]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    const auto& x = get<0>(logical_coords);
    const auto y = get<1>(logical_coords) + 2.;
    get<0>(get<VectorTag<2>>(result)) = 1. + 1. / 3. * x + 0.2 * y;
    get<1>(get<VectorTag<2>>(result)) = 0.1;
    return result;
  }();

  using DirKey = std::pair<Direction<2>, ElementId<2>>;
  Approx local_approx = Approx::custom().epsilon(1e-11).scale(1.);
  test_hweno_work<2>(
      local_tensor, mesh, element,
      {std::make_pair(lower_xi_neighbor, lower_xi_vars),
       std::make_pair(upper_xi_neighbor, upper_xi_vars),
       std::make_pair(lower_eta_neighbor, lower_eta_vars),
       std::make_pair(upper_eta_neighbor, upper_eta_vars)},
      {std::make_pair(lower_xi_neighbor,
                      std::array<std::vector<DirKey>, 2>{
                          {std::vector<DirKey>{lower_eta_neighbor},
                           std::vector<DirKey>{upper_xi_neighbor}}}),
       std::make_pair(upper_xi_neighbor,
                      std::array<std::vector<DirKey>, 2>{
                          {std::vector<DirKey>{lower_eta_neighbor},
                           std::vector<DirKey>{upper_eta_neighbor}}}),
       std::make_pair(lower_eta_neighbor,
                      std::array<std::vector<DirKey>, 2>{
                          {std::vector<DirKey>{lower_xi_neighbor},
                           std::vector<DirKey>{upper_xi_neighbor}}}),
       std::make_pair(upper_eta_neighbor,
                      std::array<std::vector<DirKey>, 2>{
                          {std::vector<DirKey>{lower_eta_neighbor},
                           std::vector<DirKey>{upper_xi_neighbor}}})},
      local_approx);
}

void test_hweno_impl_3d(const Spectral::Quadrature quadrature =
                            Spectral::Quadrature::GaussLobatto) noexcept {
  INFO("Testing hweno_impl in 3D");
  CAPTURE(quadrature);
  using TagsList = tmpl::list<VectorTag<3>>;
  const auto mesh = Mesh<3>{{{3, 3, 4}}, Spectral::Basis::Legendre, quadrature};
  const auto element = TestHelpers::Limiters::make_element<3>();
  const auto logical_coords = logical_coordinates(mesh);

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
  const auto upper_zeta_neighbor =
      std::make_pair(Direction<3>::upper_zeta(), ElementId<3>{6});

  const auto local_tensor = [&logical_coords]() noexcept {
    const auto& x = get<0>(logical_coords);
    const auto& y = get<1>(logical_coords);
    const auto& z = get<2>(logical_coords);
    return VectorTag<3>::type{{{DataVector{-2. + 0.2 * y * square(z)},
                                DataVector{0.8 - 0.1 * square(x) * z},
                                DataVector{5. + 0.5 * x * y * z}}}};
  }();

  const auto lower_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get<0>(get<VectorTag<3>>(result)) = 1.;
    get<1>(get<VectorTag<3>>(result)) = 1.;
    get<2>(get<VectorTag<3>>(result)) = 1.;
    return result;
  }();

  const auto upper_xi_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get<0>(get<VectorTag<3>>(result)) = -8.1;
    get<1>(get<VectorTag<3>>(result)) = -1.5;
    get<2>(get<VectorTag<3>>(result)) = 2.5;
    return result;
  }();

  const auto lower_eta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get<0>(get<VectorTag<3>>(result)) = 0.7;
    get<1>(get<VectorTag<3>>(result)) = 0.1;
    get<2>(get<VectorTag<3>>(result)) = 10.3;
    return result;
  }();

  const auto upper_eta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get<0>(get<VectorTag<3>>(result)) = -3.9;
    get<1>(get<VectorTag<3>>(result)) = 1.2;
    get<2>(get<VectorTag<3>>(result)) = -0.3;
    return result;
  }();

  const auto lower_zeta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get<0>(get<VectorTag<3>>(result)) = -5.4;
    get<1>(get<VectorTag<3>>(result)) = 0.1;
    get<2>(get<VectorTag<3>>(result)) = 4.2;
    return result;
  }();

  const auto upper_zeta_vars = [&mesh]() noexcept {
    Variables<TagsList> result(mesh.number_of_grid_points());
    get<0>(get<VectorTag<3>>(result)) = -2.3;
    get<1>(get<VectorTag<3>>(result)) = -0.9;
    get<2>(get<VectorTag<3>>(result)) = 1.1;
    return result;
  }();

  using DirKey = std::pair<Direction<3>, ElementId<3>>;
  Approx local_approx = Approx::custom().epsilon(1e-11).scale(1.);
  test_hweno_work<3>(
      local_tensor, mesh, element,
      {std::make_pair(lower_xi_neighbor, lower_xi_vars),
       std::make_pair(upper_xi_neighbor, upper_xi_vars),
       std::make_pair(lower_eta_neighbor, lower_eta_vars),
       std::make_pair(upper_eta_neighbor, upper_eta_vars),
       std::make_pair(lower_zeta_neighbor, lower_zeta_vars),
       std::make_pair(upper_zeta_neighbor, upper_zeta_vars)},
      {std::make_pair(
           lower_xi_neighbor,
           std::array<std::vector<DirKey>, 3>{
               {std::vector<DirKey>{upper_xi_neighbor},
                std::vector<DirKey>{upper_xi_neighbor},
                std::vector<DirKey>{lower_eta_neighbor, upper_eta_neighbor}}}),
       std::make_pair(
           upper_xi_neighbor,
           std::array<std::vector<DirKey>, 3>{
               {std::vector<DirKey>{lower_zeta_neighbor},
                std::vector<DirKey>{upper_zeta_neighbor},
                std::vector<DirKey>{lower_eta_neighbor, upper_eta_neighbor}}}),
       std::make_pair(lower_eta_neighbor,
                      std::array<std::vector<DirKey>, 3>{
                          {std::vector<DirKey>{upper_xi_neighbor},
                           std::vector<DirKey>{upper_xi_neighbor},
                           std::vector<DirKey>{upper_eta_neighbor}}}),
       std::make_pair(upper_eta_neighbor,
                      std::array<std::vector<DirKey>, 3>{
                          {std::vector<DirKey>{upper_xi_neighbor},
                           std::vector<DirKey>{upper_xi_neighbor},
                           std::vector<DirKey>{lower_eta_neighbor}}}),
       std::make_pair(
           lower_zeta_neighbor,
           std::array<std::vector<DirKey>, 3>{
               {std::vector<DirKey>{upper_xi_neighbor},
                std::vector<DirKey>{upper_xi_neighbor},
                std::vector<DirKey>{lower_eta_neighbor, upper_eta_neighbor}}}),
       std::make_pair(
           upper_zeta_neighbor,
           std::array<std::vector<DirKey>, 3>{
               {std::vector<DirKey>{upper_xi_neighbor},
                std::vector<DirKey>{upper_xi_neighbor},
                std::vector<DirKey>{lower_eta_neighbor, upper_eta_neighbor}}})},
      local_approx);
}

}  // namespace

SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.HwenoImpl", "[Limiters][Unit]") {
  test_secondary_neighbors_to_exclude_from_fit();

  // These functions test the constrained fit algorithm.
  // In particular, each function tests:
  // - the typical case with one excluded neighbor
  // - an atypical case with 1+ external boundaries and 1+ excluded neighbors
  test_constrained_fit_1d();
  test_constrained_fit_2d_vector();
  test_constrained_fit_3d();

  // It is difficult to test the HWENO algorithm without entirely reimplementing
  // it. However, each of the main pieces ...
  //  - Weno_detail::secondary_neighbors_to_exclude_from_fit
  //  - Weno_detail::solve_constrained_fit
  //  - Weno_detail::reconstruct_from_weighted_sum
  // ... has a detailed independent test (in this file or in Test_WenoHelpers).
  // So here we provide a simple test in which we manually list the neighbors to
  // exclude, then call the constrained fit and reconstruction functions. We
  // check that the hweno_impl results match this simple reconstruction. This
  // tests that the different pieces are plugged together in the right way.
  test_hweno_impl_1d();
  test_hweno_impl_2d();
  test_hweno_impl_3d();
}

// Separate the Gauss quadrature tests for two reasons:
// 1. Generalizing the Hweno matrix static caches to handle both LGL and LG
//    points in the same run would be somewhat tedious. By testing each basis
//    in a separate SPECTRE_TEST_CASE, run in separate calls to the test
//    executable, the static cache clashes are avoided.
// 2. To keep the test case duration comfortably under the 2s time limit
SPECTRE_TEST_CASE("Unit.Evolution.DG.Limiters.HwenoImpl.GaussQuadrature",
                  "[Limiters][Unit]") {
  const auto gauss = Spectral::Quadrature::Gauss;
  test_constrained_fit_1d(gauss);
  test_constrained_fit_2d_vector(gauss);
  test_constrained_fit_3d(gauss);

  test_hweno_impl_1d(gauss);
  test_hweno_impl_2d(gauss);
  test_hweno_impl_3d(gauss);
}
