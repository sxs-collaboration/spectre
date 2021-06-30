// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Framework/TestingFramework.hpp"

#include <cstddef>
#include <string>
#include <type_traits>

#include "DataStructures/Tensor/Symmetry.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/TMPL.hpp"

namespace {
void test_impl_consistency() {
  const std::string error_msg =
      "Symmetry and AddSubSymmetry are no longer using the same canonical "
      "form. You must update the implementation for "
      "TensorExpressions::detail::get_addsub_symm to use the same canonical "
      "form as Symmetry and update the expected result symmetries for the unit "
      "test cases in this file.";

  if (not std::is_same_v<Symmetry<>, tmpl::integral_list<std::int32_t>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<4>, tmpl::integral_list<std::int32_t, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<1, 2>,
                         tmpl::integral_list<std::int32_t, 2, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<3, 5>,
                         tmpl::integral_list<std::int32_t, 2, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<2, 2, 2>,
                         tmpl::integral_list<std::int32_t, 1, 1, 1>>) {
    ERROR(error_msg);
  }
  if (not std::is_same_v<Symmetry<8, 4, 5, 5, 8>,
                         tmpl::integral_list<std::int32_t, 1, 3, 2, 2, 1>>) {
    ERROR(error_msg);
  }
}

void test_rank0() {
  using symm = Symmetry<>;
  using tensorindex_list = make_tensorindex_list<>;

  CHECK(
      std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                         symm, symm, tensorindex_list, tensorindex_list>::type,
                     tmpl::integral_list<std::int32_t>>);
}

void test_rank1() {
  using symm = Symmetry<1>;
  using tensorindex_list = make_tensorindex_list<ti_a>;

  CHECK(
      std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                         symm, symm, tensorindex_list, tensorindex_list>::type,
                     tmpl::integral_list<std::int32_t, 1>>);
}

void test_rank2() {
  using symmetric_symm = Symmetry<1, 1>;
  using asymmetric_symm = Symmetry<2, 1>;
  using tensorindex_list_ij = make_tensorindex_list<ti_i, ti_j>;
  using tensorindex_list_ji = make_tensorindex_list<ti_j, ti_i>;

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symmetric_symm, symmetric_symm, tensorindex_list_ij,
                           tensorindex_list_ij>::type,
                       tmpl::integral_list<std::int32_t, 1, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symmetric_symm, symmetric_symm, tensorindex_list_ij,
                           tensorindex_list_ji>::type,
                       tmpl::integral_list<std::int32_t, 1, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           asymmetric_symm, symmetric_symm, tensorindex_list_ij,
                           tensorindex_list_ij>::type,
                       tmpl::integral_list<std::int32_t, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           asymmetric_symm, symmetric_symm, tensorindex_list_ij,
                           tensorindex_list_ji>::type,
                       tmpl::integral_list<std::int32_t, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symmetric_symm, asymmetric_symm, tensorindex_list_ij,
                           tensorindex_list_ij>::type,
                       tmpl::integral_list<std::int32_t, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symmetric_symm, asymmetric_symm, tensorindex_list_ij,
                           tensorindex_list_ji>::type,
                       tmpl::integral_list<std::int32_t, 2, 1>>);
}

void test_rank3() {
  using symm_111 = Symmetry<1, 1, 1>;
  using symm_121 = Symmetry<1, 2, 1>;
  using symm_211 = Symmetry<2, 1, 1>;
  using symm_221 = Symmetry<2, 2, 1>;
  using symm_321 = Symmetry<3, 2, 1>;

  using tensorindex_list_abc = make_tensorindex_list<ti_a, ti_b, ti_c>;
  using tensorindex_list_acb = make_tensorindex_list<ti_a, ti_c, ti_b>;
  using tensorindex_list_bac = make_tensorindex_list<ti_b, ti_a, ti_c>;
  using tensorindex_list_bca = make_tensorindex_list<ti_b, ti_c, ti_a>;
  using tensorindex_list_cab = make_tensorindex_list<ti_c, ti_a, ti_b>;
  using tensorindex_list_cba = make_tensorindex_list<ti_c, ti_b, ti_a>;

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_111, symm_121, tensorindex_list_abc,
                           tensorindex_list_bca>::type,
                       tmpl::integral_list<std::int32_t, 2, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_121, symm_111, tensorindex_list_abc,
                           tensorindex_list_bca>::type,
                       tmpl::integral_list<std::int32_t, 1, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_111, symm_221, tensorindex_list_abc,
                           tensorindex_list_acb>::type,
                       tmpl::integral_list<std::int32_t, 1, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_221, symm_111, tensorindex_list_abc,
                           tensorindex_list_acb>::type,
                       tmpl::integral_list<std::int32_t, 2, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_121, symm_221, tensorindex_list_abc,
                           tensorindex_list_cab>::type,
                       tmpl::integral_list<std::int32_t, 1, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_221, symm_121, tensorindex_list_abc,
                           tensorindex_list_cab>::type,
                       tmpl::integral_list<std::int32_t, 3, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_221, symm_121, tensorindex_list_cab,
                           tensorindex_list_abc>::type,
                       tmpl::integral_list<std::int32_t, 2, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_121, symm_221, tensorindex_list_cab,
                           tensorindex_list_abc>::type,
                       tmpl::integral_list<std::int32_t, 3, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_121, symm_221, tensorindex_list_abc,
                           tensorindex_list_acb>::type,
                       tmpl::integral_list<std::int32_t, 1, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_221, symm_121, tensorindex_list_abc,
                           tensorindex_list_acb>::type,
                       tmpl::integral_list<std::int32_t, 2, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_111, symm_321, tensorindex_list_abc,
                           tensorindex_list_bac>::type,
                       tmpl::integral_list<std::int32_t, 3, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_321, symm_111, tensorindex_list_abc,
                           tensorindex_list_bac>::type,
                       tmpl::integral_list<std::int32_t, 3, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_211, symm_321, tensorindex_list_abc,
                           tensorindex_list_cba>::type,
                       tmpl::integral_list<std::int32_t, 3, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           symm_321, symm_211, tensorindex_list_abc,
                           tensorindex_list_cba>::type,
                       tmpl::integral_list<std::int32_t, 3, 2, 1>>);
}

void test_high_rank() {
  using tensorindex_list = make_tensorindex_list<ti_a, ti_b, ti_c, ti_d, ti_e>;

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           Symmetry<2, 1, 1, 1, 1>, Symmetry<3, 2, 2, 1, 1>,
                           tensorindex_list, tensorindex_list>::type,
                       tmpl::integral_list<std::int32_t, 3, 2, 2, 1, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           Symmetry<3, 2, 2, 1, 1>, Symmetry<2, 1, 1, 1, 1>,
                           tensorindex_list, tensorindex_list>::type,
                       tmpl::integral_list<std::int32_t, 3, 2, 2, 1, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           Symmetry<1, 1, 2, 1, 1>, Symmetry<4, 3, 1, 2, 1>,
                           tensorindex_list, tensorindex_list>::type,
                       tmpl::integral_list<std::int32_t, 5, 4, 3, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           Symmetry<4, 3, 1, 2, 1>, Symmetry<1, 1, 2, 1, 1>,
                           tensorindex_list, tensorindex_list>::type,
                       tmpl::integral_list<std::int32_t, 5, 4, 3, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           Symmetry<1, 2, 2, 2, 1>, Symmetry<1, 2, 1, 1, 1>,
                           tensorindex_list, tensorindex_list>::type,
                       tmpl::integral_list<std::int32_t, 1, 3, 2, 2, 1>>);

  CHECK(std::is_same_v<typename TensorExpressions::detail::AddSubSymmetry<
                           Symmetry<1, 2, 1, 1, 1>, Symmetry<1, 2, 2, 2, 1>,
                           tensorindex_list, tensorindex_list>::type,
                       tmpl::integral_list<std::int32_t, 1, 3, 2, 2, 1>>);
}
}  // namespace

SPECTRE_TEST_CASE("Unit.DataStructures.Tensor.Expression.AddSubSymmetry",
                  "[DataStructures][Unit]") {
  test_impl_consistency();
  test_rank0();
  test_rank1();
  test_rank2();
  test_rank3();
  test_high_rank();
}
