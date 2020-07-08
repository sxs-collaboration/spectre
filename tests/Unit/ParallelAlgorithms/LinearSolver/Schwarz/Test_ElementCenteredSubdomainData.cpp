// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/Direction.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Framework/TestHelpers.hpp"
#include "NumericalAlgorithms/LinearSolver/InnerProduct.hpp"
#include "ParallelAlgorithms/LinearSolver/Schwarz/ElementCenteredSubdomainData.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace LinearSolver::Schwarz {

namespace {
struct ScalarField : db::SimpleTag {
  using type = Scalar<DataVector>;
};
}  // namespace

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.ElementCenteredSubdomainData",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  const auto west_id =
      std::make_pair(Direction<1>::lower_xi(), ElementId<1>{0, {{{2, 0}}}});
  const auto east_id =
      std::make_pair(Direction<1>::upper_xi(), ElementId<1>{0, {{{2, 2}}}});
  const auto make_subdomain_data = [&west_id, &east_id](
                                       DataVector element_data,
                                       DataVector east_overlap_data,
                                       DataVector west_overlap_data) noexcept {
    ElementCenteredSubdomainData<1, tmpl::list<ScalarField>> subdomain_data{
        element_data.size()};
    get(get<ScalarField>(subdomain_data.element_data)) =
        std::move(element_data);
    subdomain_data.overlap_data.emplace(west_id, west_overlap_data.size());
    get(get<ScalarField>(subdomain_data.overlap_data.at(west_id))) =
        std::move(west_overlap_data);
    subdomain_data.overlap_data.emplace(east_id, east_overlap_data.size());
    get(get<ScalarField>(subdomain_data.overlap_data.at(east_id))) =
        std::move(east_overlap_data);
    return subdomain_data;
  };
  auto subdomain_data1 = make_subdomain_data({1., 2., 3.}, {4., 5.}, {6.});
  const auto subdomain_data2 =
      make_subdomain_data({2., 1., 0.}, {1., 2.}, {3.});
  SECTION("Addition") {
    const auto subdomain_data_sum =
        make_subdomain_data({3., 3., 3.}, {5., 7.}, {9.});
    CHECK(subdomain_data1 + subdomain_data2 == subdomain_data_sum);
    subdomain_data1 += subdomain_data2;
    CHECK(subdomain_data1 == subdomain_data_sum);
  }
  SECTION("Subtraction") {
    const auto subdomain_data_diff =
        make_subdomain_data({-1., 1., 3.}, {3., 3.}, {3.});
    CHECK(subdomain_data1 - subdomain_data2 == subdomain_data_diff);
    subdomain_data1 -= subdomain_data2;
    CHECK(subdomain_data1 == subdomain_data_diff);
  }
  SECTION("Scalar multiplication") {
    const auto subdomain_data_double =
        make_subdomain_data({2., 4., 6.}, {8., 10.}, {12.});
    CHECK(2. * subdomain_data1 == subdomain_data_double);
    CHECK(subdomain_data1 * 2. == subdomain_data_double);
    subdomain_data1 *= 2.;
    CHECK(subdomain_data1 == subdomain_data_double);
  }
  SECTION("Scalar division") {
    const auto subdomain_data_half =
        make_subdomain_data({0.5, 1., 1.5}, {2., 2.5}, {3.});
    CHECK(subdomain_data1 / 2. == subdomain_data_half);
    subdomain_data1 /= 2.;
    CHECK(subdomain_data1 == subdomain_data_half);
  }
  SECTION("Remaining tests") {
    test_serialization(subdomain_data1);
    test_copy_semantics(subdomain_data1);
    auto copied_subdomain_data = subdomain_data1;
    test_move_semantics(std::move(copied_subdomain_data), subdomain_data1);
    CHECK(inner_product(subdomain_data1, subdomain_data2) == 36.);
    CHECK(make_with_value<
              ElementCenteredSubdomainData<1, tmpl::list<ScalarField>>>(
              subdomain_data1, 1.) ==
          make_subdomain_data({1., 1., 1.}, {1., 1.}, {1.}));
  }
}

}  // namespace LinearSolver::Schwarz
