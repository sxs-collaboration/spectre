// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <complex>
#include <cstddef>

#include "DataStructures/ComplexDataVector.hpp"
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

using namespace std::complex_literals;

namespace {
template <typename DataType>
struct ScalarField : db::SimpleTag {
  using type = Scalar<DataType>;
};
}  // namespace

template <typename DataType>
void test_subdomain_data() {
  const DirectionalId<1> west_id{Direction<1>::lower_xi(),
                                 ElementId<1>{0, {{{2, 0}}}}};
  const DirectionalId<1> east_id{Direction<1>::upper_xi(),
                                 ElementId<1>{0, {{{2, 2}}}}};
  const auto make_subdomain_data = [&west_id, &east_id](
                                       DataType element_data,
                                       DataType east_overlap_data,
                                       DataType west_overlap_data) {
    ElementCenteredSubdomainData<1, tmpl::list<ScalarField<DataType>>>
        subdomain_data{element_data.size()};
    get(get<ScalarField<DataType>>(subdomain_data.element_data)) =
        std::move(element_data);
    subdomain_data.overlap_data.emplace(west_id, west_overlap_data.size());
    get(get<ScalarField<DataType>>(subdomain_data.overlap_data.at(west_id))) =
        std::move(west_overlap_data);
    subdomain_data.overlap_data.emplace(east_id, east_overlap_data.size());
    get(get<ScalarField<DataType>>(subdomain_data.overlap_data.at(east_id))) =
        std::move(east_overlap_data);
    return subdomain_data;
  };
  if constexpr (std::is_same_v<DataType, DataVector>) {
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
    SECTION("Iterate raw data") {
      CAPTURE(subdomain_data1);
      CAPTURE(subdomain_data2);
      std::copy(subdomain_data2.begin(), subdomain_data2.end(),
                subdomain_data1.begin());
      CHECK(subdomain_data1 == subdomain_data2);
    }
    SECTION("Remaining tests") {
      test_serialization(subdomain_data1);
      test_copy_semantics(subdomain_data1);
      auto copied_subdomain_data = subdomain_data1;
      test_move_semantics(std::move(copied_subdomain_data), subdomain_data1);
      CHECK(inner_product(subdomain_data1, subdomain_data2) == 36.);
      CHECK(make_with_value<ElementCenteredSubdomainData<
                1, tmpl::list<ScalarField<DataType>>>>(subdomain_data1, 1.) ==
            make_subdomain_data({1., 1., 1.}, {1., 1.}, {1.}));
    }
    SECTION("Resizing") {
      const DirectionalId<1> extra_id{Direction<1>::upper_xi(),
                                      ElementId<1>{1, {{{2, 2}}}}};
      subdomain_data1.overlap_data.erase(west_id);
      subdomain_data1.overlap_data.emplace(extra_id, 3);
      subdomain_data1.destructive_resize(subdomain_data2);
      CHECK(subdomain_data1.overlap_data.size() == 2);
      CHECK(subdomain_data1.overlap_data.count(extra_id) == 0);
      CHECK(subdomain_data1.overlap_data.count(west_id) == 1);
      CHECK(subdomain_data1.overlap_data.count(east_id) == 1);
    }
  } else {  // ComplexDataVector
    auto subdomain_data1 = make_subdomain_data(
        {1. + 2.i, 2. + 3.i, 3. + 4.i}, {4. + 5.i, 5. + 6.i}, {6. + 7.i});
    const auto subdomain_data2 = make_subdomain_data(
        {2. + 3.i, 1. + 2.i, 0. + 1.i}, {1. + 2.i, 2. + 3.i}, {3. + 4.i});
    SECTION("Addition") {
      const auto subdomain_data_sum = make_subdomain_data(
          {3. + 5.i, 3. + 5.i, 3. + 5.i}, {5. + 7.i, 7. + 9.i}, {9. + 11.i});
      CHECK(subdomain_data1 + subdomain_data2 == subdomain_data_sum);
      subdomain_data1 += subdomain_data2;
      CHECK(subdomain_data1 == subdomain_data_sum);
    }
    SECTION("Subtraction") {
      const auto subdomain_data_diff = make_subdomain_data(
          {-1. - 1.i, 1. + 1.i, 3. + 3.i}, {3. + 3.i, 3. + 3.i}, {3. + 3.i});
      CHECK(subdomain_data1 - subdomain_data2 == subdomain_data_diff);
      subdomain_data1 -= subdomain_data2;
      CHECK(subdomain_data1 == subdomain_data_diff);
    }
    SECTION("Scalar multiplication") {
      const auto subdomain_data_double =
          make_subdomain_data({2. + 4.i, 4. + 6.i, 6. + 8.i},
                              {8. + 10.i, 10. + 12.i}, {12. + 14.i});
      CHECK(2. * subdomain_data1 == subdomain_data_double);
      CHECK(subdomain_data1 * 2. == subdomain_data_double);
      subdomain_data1 *= 2.;
      CHECK(subdomain_data1 == subdomain_data_double);
    }
    SECTION("Scalar division") {
      const auto subdomain_data_half =
          make_subdomain_data({0.5 + 1.i, 1. + 1.5i, 1.5 + 2.i},
                              {2. + 2.5i, 2.5 + 3.i}, {3. + 3.5i});
      CHECK(subdomain_data1 / 2. == subdomain_data_half);
      subdomain_data1 /= 2.;
      CHECK(subdomain_data1 == subdomain_data_half);
    }
    SECTION("Iterate raw data") {
      CAPTURE(subdomain_data1);
      CAPTURE(subdomain_data2);
      std::copy(subdomain_data2.begin(), subdomain_data2.end(),
                subdomain_data1.begin());
      CHECK(subdomain_data1 == subdomain_data2);
    }
    SECTION("Remaining tests") {
      test_serialization(subdomain_data1);
      test_copy_semantics(subdomain_data1);
      auto copied_subdomain_data = subdomain_data1;
      test_move_semantics(std::move(copied_subdomain_data), subdomain_data1);
      CHECK(inner_product(subdomain_data1, subdomain_data2) ==
            std::complex<double>(108., 12.));
      CHECK(magnitude_square(subdomain_data1) == 230.);
      CHECK(make_with_value<ElementCenteredSubdomainData<
                1, tmpl::list<ScalarField<DataType>>>>(subdomain_data1, 1.) ==
            make_subdomain_data({1. + 0.i, 1. + 0.i, 1. + 0.i},
                                {1. + 0.i, 1. + 0.i}, {1. + 0.i}));
    }
    SECTION("Resizing") {
      const DirectionalId<1> extra_id{Direction<1>::upper_xi(),
                                      ElementId<1>{1, {{{2, 2}}}}};
      subdomain_data1.overlap_data.erase(west_id);
      subdomain_data1.overlap_data.emplace(extra_id, 3);
      subdomain_data1.destructive_resize(subdomain_data2);
      CHECK(subdomain_data1.overlap_data.size() == 2);
      CHECK(subdomain_data1.overlap_data.count(extra_id) == 0);
      CHECK(subdomain_data1.overlap_data.count(west_id) == 1);
      CHECK(subdomain_data1.overlap_data.count(east_id) == 1);
    }
  }
}

SPECTRE_TEST_CASE("Unit.ParallelSchwarz.ElementCenteredSubdomainData",
                  "[Unit][ParallelAlgorithms][LinearSolver]") {
  test_subdomain_data<DataVector>();
  test_subdomain_data<ComplexDataVector>();
}

}  // namespace LinearSolver::Schwarz
