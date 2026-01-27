// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <limits>
#include <random>
#include <utility>

#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "DataStructures/Variables.hpp"
#include "DataStructures/VectorImpl.hpp"
#include "Framework/TestHelpers.hpp"
#include "Helpers/DataStructures/MakeWithRandomValues.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/ComponentPlaceholder.hpp"
#include "Helpers/DataStructures/Tensor/Expressions/TestHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"
#include "Utilities/Requires.hpp"
#include "Utilities/TMPL.hpp"

namespace TestHelpers::tenex {

/// \ingroup TestingFrameworkGroup
/// \brief Test that evaluating a right hand side tensor expression containing a
/// single rank 4 tensor correctly assigns the data to the evaluated left hand
/// side tensor
///
/// \details See `test_evaluate_rank_3_impl` for general details.
///
/// \tparam ReturnLhsTensor whether to test tensor expression evaluation by
/// returning the result tensor or not, which instead tests evaluation by
/// assigning to the result tensor passed in as an argument
/// \tparam DataType the type of data being stored in the Tensors
/// \tparam RhsSymmetry the ::Symmetry of the RHS Tensor
/// \tparam RhsTensorIndexTypeList the RHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
/// \tparam TensorIndexA the first TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::a`
/// \tparam TensorIndexB the second TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::B`
/// \tparam TensorIndexC the third TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::c`
/// \tparam TensorIndexD the fourth TensorIndex used on the RHS of the
/// TensorExpression, e.g. `ti::D`
/// \tparam LhsSymmetry the ::Symmetry of the LHS Tensor
/// \tparam LhsTensorIndexTypeList the LHS Tensor's typelist of
/// \ref SpacetimeIndex "TensorIndexType"s
template <bool ReturnLhsTensor, typename DataType, typename RhsSymmetry,
          typename RhsTensorIndexTypeList, auto& TensorIndexA,
          auto& TensorIndexB, auto& TensorIndexC, auto& TensorIndexD,
          typename LhsSymmetry = RhsSymmetry,
          typename LhsTensorIndexTypeList = RhsTensorIndexTypeList>
void test_evaluate_rank_4() {
  MAKE_GENERATOR(generator);
  std::uniform_real_distribution<> distribution(-5.0, 5.0);
  const size_t used_for_size = 3;
  const auto R_abcd = make_with_random_values<
      Tensor<DataType, RhsSymmetry, RhsTensorIndexTypeList>>(
      make_not_null(&generator), distribution, used_for_size);
  auto expected_L_abcd =
      ReturnLhsTensor
          ? Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>{}
          : make_with_value<
                Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>>(
                used_for_size, component_placeholder_value<DataType>::value);

  const std::int32_t lhs_symmetry_element_a = tmpl::at_c<LhsSymmetry, 0>::value;
  const std::int32_t lhs_symmetry_element_b = tmpl::at_c<LhsSymmetry, 1>::value;
  const std::int32_t lhs_symmetry_element_c = tmpl::at_c<LhsSymmetry, 2>::value;
  const std::int32_t lhs_symmetry_element_d = tmpl::at_c<LhsSymmetry, 3>::value;
  using lhs_tensorindextype_a = tmpl::at_c<LhsTensorIndexTypeList, 0>;
  using lhs_tensorindextype_b = tmpl::at_c<LhsTensorIndexTypeList, 1>;
  using lhs_tensorindextype_c = tmpl::at_c<LhsTensorIndexTypeList, 2>;
  using lhs_tensorindextype_d = tmpl::at_c<LhsTensorIndexTypeList, 3>;
  using rhs_tensorindextype_a = tmpl::at_c<RhsTensorIndexTypeList, 0>;
  using rhs_tensorindextype_b = tmpl::at_c<RhsTensorIndexTypeList, 1>;
  using rhs_tensorindextype_c = tmpl::at_c<RhsTensorIndexTypeList, 2>;
  using rhs_tensorindextype_d = tmpl::at_c<RhsTensorIndexTypeList, 3>;

  std::array<std::pair<size_t, size_t>, 4> lhs_index_value_ranges{};
  lhs_index_value_ranges[0] =
      get_index_value_range<lhs_tensorindextype_a, TensorIndexA>();
  lhs_index_value_ranges[1] =
      get_index_value_range<lhs_tensorindextype_b, TensorIndexB>();
  lhs_index_value_ranges[2] =
      get_index_value_range<lhs_tensorindextype_c, TensorIndexC>();
  lhs_index_value_ranges[3] =
      get_index_value_range<lhs_tensorindextype_d, TensorIndexD>();
  std::array<std::pair<size_t, size_t>, 4> rhs_index_value_ranges{};
  rhs_index_value_ranges[0] =
      get_index_value_range<rhs_tensorindextype_a, TensorIndexA>();
  rhs_index_value_ranges[1] =
      get_index_value_range<rhs_tensorindextype_b, TensorIndexB>();
  rhs_index_value_ranges[2] =
      get_index_value_range<rhs_tensorindextype_c, TensorIndexC>();
  rhs_index_value_ranges[3] =
      get_index_value_range<rhs_tensorindextype_d, TensorIndexD>();

  for (size_t lhs_a = lhs_index_value_ranges[0].first,
              rhs_a = rhs_index_value_ranges[0].first;
       lhs_a <= lhs_index_value_ranges[0].second; lhs_a++, rhs_a++) {
    for (size_t lhs_b = lhs_index_value_ranges[1].first,
                rhs_b = rhs_index_value_ranges[1].first;
         lhs_b <= lhs_index_value_ranges[1].second; lhs_b++, rhs_b++) {
      for (size_t lhs_c = lhs_index_value_ranges[2].first,
                  rhs_c = rhs_index_value_ranges[2].first;
           lhs_c <= lhs_index_value_ranges[2].second; lhs_c++, rhs_c++) {
        for (size_t lhs_d = lhs_index_value_ranges[3].first,
                    rhs_d = rhs_index_value_ranges[3].first;
             lhs_d <= lhs_index_value_ranges[3].second; lhs_d++, rhs_d++) {
          expected_L_abcd.get(lhs_a, lhs_b, lhs_c, lhs_d) =
              R_abcd.get(rhs_a, rhs_b, rhs_c, rhs_d);
        }
      }
    }
  }

  const auto rhs_expression =
      R_abcd(TensorIndexA, TensorIndexB, TensorIndexC, TensorIndexD);

  // L_{abcd} = R_{abcd}
  using L_abcd_type = Tensor<DataType, LhsSymmetry, LhsTensorIndexTypeList>;
  L_abcd_type L_abcd(used_for_size);
  // component placeholder is used to detect which components have incorrectly
  // or correctly (in the case of using spatial or time indices for spacetime
  // indices) not been modified by evaluation of the RHS expression
  std::fill(L_abcd.begin(), L_abcd.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexC,
                TensorIndexD>(make_not_null(&L_abcd), rhs_expression);

  // L_{abdc} = R_{abcd}
  using L_abdc_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_b,
               lhs_symmetry_element_d, lhs_symmetry_element_c>;
  using L_abdc_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_b,
                 lhs_tensorindextype_d, lhs_tensorindextype_c>;
  using L_abdc_type =
      Tensor<DataType, L_abdc_symmetry, L_abdc_tensorindextype_list>;
  L_abdc_type L_abdc(used_for_size);
  std::fill(L_abdc.begin(), L_abdc.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexB, TensorIndexD,
                TensorIndexC>(make_not_null(&L_abdc), rhs_expression);

  // L_{acbd} = R_{abcd}
  using L_acbd_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_c,
               lhs_symmetry_element_b, lhs_symmetry_element_d>;
  using L_acbd_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_c,
                 lhs_tensorindextype_b, lhs_tensorindextype_d>;
  using L_acbd_type =
      Tensor<DataType, L_acbd_symmetry, L_acbd_tensorindextype_list>;
  L_acbd_type L_acbd(used_for_size);
  std::fill(L_acbd.begin(), L_acbd.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexC, TensorIndexB,
                TensorIndexD>(make_not_null(&L_acbd), rhs_expression);

  // L_{acdb} = R_{abcd}
  using L_acdb_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_c,
               lhs_symmetry_element_d, lhs_symmetry_element_b>;
  using L_acdb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_c,
                 lhs_tensorindextype_d, lhs_tensorindextype_b>;
  using L_acdb_type =
      Tensor<DataType, L_acdb_symmetry, L_acdb_tensorindextype_list>;
  L_acdb_type L_acdb(used_for_size);
  std::fill(L_acdb.begin(), L_acdb.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexC, TensorIndexD,
                TensorIndexB>(make_not_null(&L_acdb), rhs_expression);

  // L_{adbc} = R_{abcd}
  using L_adbc_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_d,
               lhs_symmetry_element_b, lhs_symmetry_element_c>;
  using L_adbc_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_d,
                 lhs_tensorindextype_b, lhs_tensorindextype_c>;
  using L_adbc_type =
      Tensor<DataType, L_adbc_symmetry, L_adbc_tensorindextype_list>;
  L_adbc_type L_adbc(used_for_size);
  std::fill(L_adbc.begin(), L_adbc.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexD, TensorIndexB,
                TensorIndexC>(make_not_null(&L_adbc), rhs_expression);

  // L_{adcb} = R_{abcd}
  using L_adcb_symmetry =
      Symmetry<lhs_symmetry_element_a, lhs_symmetry_element_d,
               lhs_symmetry_element_c, lhs_symmetry_element_b>;
  using L_adcb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_a, lhs_tensorindextype_d,
                 lhs_tensorindextype_c, lhs_tensorindextype_b>;
  using L_adcb_type =
      Tensor<DataType, L_adcb_symmetry, L_adcb_tensorindextype_list>;
  L_adcb_type L_adcb(used_for_size);
  std::fill(L_adcb.begin(), L_adcb.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexA, TensorIndexD, TensorIndexC,
                TensorIndexB>(make_not_null(&L_adcb), rhs_expression);

  // L_{bacd} = R_{abcd}
  using L_bacd_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a,
               lhs_symmetry_element_c, lhs_symmetry_element_d>;
  using L_bacd_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a,
                 lhs_tensorindextype_c, lhs_tensorindextype_d>;
  using L_bacd_type =
      Tensor<DataType, L_bacd_symmetry, L_bacd_tensorindextype_list>;
  L_bacd_type L_bacd(used_for_size);
  std::fill(L_bacd.begin(), L_bacd.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA, TensorIndexC,
                TensorIndexD>(make_not_null(&L_bacd), rhs_expression);

  // L_{badc} = R_{abcd}
  using L_badc_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_a,
               lhs_symmetry_element_d, lhs_symmetry_element_c>;
  using L_badc_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_a,
                 lhs_tensorindextype_d, lhs_tensorindextype_c>;
  using L_badc_type =
      Tensor<DataType, L_badc_symmetry, L_badc_tensorindextype_list>;
  L_badc_type L_badc(used_for_size);
  std::fill(L_badc.begin(), L_badc.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexA, TensorIndexD,
                TensorIndexC>(make_not_null(&L_badc), rhs_expression);

  // L_{bcad} = R_{abcd}
  using L_bcad_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_c,
               lhs_symmetry_element_a, lhs_symmetry_element_d>;
  using L_bcad_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_c,
                 lhs_tensorindextype_a, lhs_tensorindextype_d>;
  using L_bcad_type =
      Tensor<DataType, L_bcad_symmetry, L_bcad_tensorindextype_list>;
  L_bcad_type L_bcad(used_for_size);
  std::fill(L_bcad.begin(), L_bcad.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexC, TensorIndexA,
                TensorIndexD>(make_not_null(&L_bcad), rhs_expression);

  // L_{bcda} = R_{abcd}
  using L_bcda_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_c,
               lhs_symmetry_element_d, lhs_symmetry_element_a>;
  using L_bcda_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_c,
                 lhs_tensorindextype_d, lhs_tensorindextype_a>;
  using L_bcda_type =
      Tensor<DataType, L_bcda_symmetry, L_bcda_tensorindextype_list>;
  L_bcda_type L_bcda(used_for_size);
  std::fill(L_bcda.begin(), L_bcda.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexC, TensorIndexD,
                TensorIndexA>(make_not_null(&L_bcda), rhs_expression);

  // L_{bdac} = R_{abcd}
  using L_bdac_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_d,
               lhs_symmetry_element_a, lhs_symmetry_element_c>;
  using L_bdac_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_d,
                 lhs_tensorindextype_a, lhs_tensorindextype_c>;
  using L_bdac_type =
      Tensor<DataType, L_bdac_symmetry, L_bdac_tensorindextype_list>;
  L_bdac_type L_bdac(used_for_size);
  std::fill(L_bdac.begin(), L_bdac.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexD, TensorIndexA,
                TensorIndexC>(make_not_null(&L_bdac), rhs_expression);

  // L_{bdca} = R_{abcd}
  using L_bdca_symmetry =
      Symmetry<lhs_symmetry_element_b, lhs_symmetry_element_d,
               lhs_symmetry_element_c, lhs_symmetry_element_a>;
  using L_bdca_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_b, lhs_tensorindextype_d,
                 lhs_tensorindextype_c, lhs_tensorindextype_a>;
  using L_bdca_type =
      Tensor<DataType, L_bdca_symmetry, L_bdca_tensorindextype_list>;
  L_bdca_type L_bdca(used_for_size);
  std::fill(L_bdca.begin(), L_bdca.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexB, TensorIndexD, TensorIndexC,
                TensorIndexA>(make_not_null(&L_bdca), rhs_expression);

  // L_{cabd} = R_{abcd}
  using L_cabd_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_a,
               lhs_symmetry_element_b, lhs_symmetry_element_d>;
  using L_cabd_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_a,
                 lhs_tensorindextype_b, lhs_tensorindextype_d>;
  using L_cabd_type =
      Tensor<DataType, L_cabd_symmetry, L_cabd_tensorindextype_list>;
  L_cabd_type L_cabd(used_for_size);
  std::fill(L_cabd.begin(), L_cabd.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexA, TensorIndexB,
                TensorIndexD>(make_not_null(&L_cabd), rhs_expression);

  // L_{cadb} = R_{abcd}
  using L_cadb_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_a,
               lhs_symmetry_element_d, lhs_symmetry_element_b>;
  using L_cadb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_a,
                 lhs_tensorindextype_d, lhs_tensorindextype_b>;
  using L_cadb_type =
      Tensor<DataType, L_cadb_symmetry, L_cadb_tensorindextype_list>;
  L_cadb_type L_cadb(used_for_size);
  std::fill(L_cadb.begin(), L_cadb.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexA, TensorIndexD,
                TensorIndexB>(make_not_null(&L_cadb), rhs_expression);

  // L_{cbad} = R_{abcd}
  using L_cbad_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_b,
               lhs_symmetry_element_a, lhs_symmetry_element_d>;
  using L_cbad_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_b,
                 lhs_tensorindextype_a, lhs_tensorindextype_d>;
  using L_cbad_type =
      Tensor<DataType, L_cbad_symmetry, L_cbad_tensorindextype_list>;
  L_cbad_type L_cbad(used_for_size);
  std::fill(L_cbad.begin(), L_cbad.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexB, TensorIndexA,
                TensorIndexD>(make_not_null(&L_cbad), rhs_expression);

  // L_{cbda} = R_{abcd}
  using L_cbda_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_b,
               lhs_symmetry_element_d, lhs_symmetry_element_a>;
  using L_cbda_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_b,
                 lhs_tensorindextype_d, lhs_tensorindextype_a>;
  using L_cbda_type =
      Tensor<DataType, L_cbda_symmetry, L_cbda_tensorindextype_list>;
  L_cbda_type L_cbda(used_for_size);
  std::fill(L_cbda.begin(), L_cbda.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexB, TensorIndexD,
                TensorIndexA>(make_not_null(&L_cbda), rhs_expression);

  // L_{cdab} = R_{abcd}
  using L_cdab_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_d,
               lhs_symmetry_element_a, lhs_symmetry_element_b>;
  using L_cdab_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_d,
                 lhs_tensorindextype_a, lhs_tensorindextype_b>;
  using L_cdab_type =
      Tensor<DataType, L_cdab_symmetry, L_cdab_tensorindextype_list>;
  L_cdab_type L_cdab(used_for_size);
  std::fill(L_cdab.begin(), L_cdab.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexD, TensorIndexA,
                TensorIndexB>(make_not_null(&L_cdab), rhs_expression);

  // L_{cdba} = R_{abcd}
  using L_cdba_symmetry =
      Symmetry<lhs_symmetry_element_c, lhs_symmetry_element_d,
               lhs_symmetry_element_b, lhs_symmetry_element_a>;
  using L_cdba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_c, lhs_tensorindextype_d,
                 lhs_tensorindextype_b, lhs_tensorindextype_a>;
  using L_cdba_type =
      Tensor<DataType, L_cdba_symmetry, L_cdba_tensorindextype_list>;
  L_cdba_type L_cdba(used_for_size);
  std::fill(L_cdba.begin(), L_cdba.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexC, TensorIndexD, TensorIndexB,
                TensorIndexA>(make_not_null(&L_cdba), rhs_expression);

  // L_{dabc} = R_{abcd}
  using L_dabc_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_a,
               lhs_symmetry_element_b, lhs_symmetry_element_c>;
  using L_dabc_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_a,
                 lhs_tensorindextype_b, lhs_tensorindextype_c>;
  using L_dabc_type =
      Tensor<DataType, L_dabc_symmetry, L_dabc_tensorindextype_list>;
  L_dabc_type L_dabc(used_for_size);
  std::fill(L_dabc.begin(), L_dabc.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexA, TensorIndexB,
                TensorIndexC>(make_not_null(&L_dabc), rhs_expression);

  // L_{dacb} = R_{abcd}
  using L_dacb_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_a,
               lhs_symmetry_element_c, lhs_symmetry_element_b>;
  using L_dacb_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_a,
                 lhs_tensorindextype_c, lhs_tensorindextype_b>;
  using L_dacb_type =
      Tensor<DataType, L_dacb_symmetry, L_dacb_tensorindextype_list>;
  L_dacb_type L_dacb(used_for_size);
  std::fill(L_dacb.begin(), L_dacb.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexA, TensorIndexC,
                TensorIndexB>(make_not_null(&L_dacb), rhs_expression);

  // L_{dbac} = R_{abcd}
  using L_dbac_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_b,
               lhs_symmetry_element_a, lhs_symmetry_element_c>;
  using L_dbac_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_b,
                 lhs_tensorindextype_a, lhs_tensorindextype_c>;
  using L_dbac_type =
      Tensor<DataType, L_dbac_symmetry, L_dbac_tensorindextype_list>;
  L_dbac_type L_dbac(used_for_size);
  std::fill(L_dbac.begin(), L_dbac.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexB, TensorIndexA,
                TensorIndexC>(make_not_null(&L_dbac), rhs_expression);

  // L_{dbca} = R_{abcd}
  using L_dbca_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_b,
               lhs_symmetry_element_c, lhs_symmetry_element_a>;
  using L_dbca_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_b,
                 lhs_tensorindextype_c, lhs_tensorindextype_a>;
  using L_dbca_type =
      Tensor<DataType, L_dbca_symmetry, L_dbca_tensorindextype_list>;
  L_dbca_type L_dbca(used_for_size);
  std::fill(L_dbca.begin(), L_dbca.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexB, TensorIndexC,
                TensorIndexA>(make_not_null(&L_dbca), rhs_expression);

  // L_{dcab} = R_{abcd}
  using L_dcab_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_c,
               lhs_symmetry_element_a, lhs_symmetry_element_b>;
  using L_dcab_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_c,
                 lhs_tensorindextype_a, lhs_tensorindextype_b>;
  using L_dcab_type =
      Tensor<DataType, L_dcab_symmetry, L_dcab_tensorindextype_list>;
  L_dcab_type L_dcab(used_for_size);
  std::fill(L_dcab.begin(), L_dcab.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexC, TensorIndexA,
                TensorIndexB>(make_not_null(&L_dcab), rhs_expression);

  // L_{dcba} = R_{abcd}
  using L_dcba_symmetry =
      Symmetry<lhs_symmetry_element_d, lhs_symmetry_element_c,
               lhs_symmetry_element_b, lhs_symmetry_element_a>;
  using L_dcba_tensorindextype_list =
      tmpl::list<lhs_tensorindextype_d, lhs_tensorindextype_c,
                 lhs_tensorindextype_b, lhs_tensorindextype_a>;
  using L_dcba_type =
      Tensor<DataType, L_dcba_symmetry, L_dcba_tensorindextype_list>;
  L_dcba_type L_dcba(used_for_size);
  std::fill(L_dcba.begin(), L_dcba.end(),
            component_placeholder_value<DataType>::value);
  call_evaluate<ReturnLhsTensor, TensorIndexD, TensorIndexC, TensorIndexB,
                TensorIndexA>(make_not_null(&L_dcba), rhs_expression);

  const size_t dim_a = tmpl::at_c<LhsTensorIndexTypeList, 0>::dim;
  const size_t dim_b = tmpl::at_c<LhsTensorIndexTypeList, 1>::dim;
  const size_t dim_c = tmpl::at_c<LhsTensorIndexTypeList, 2>::dim;
  const size_t dim_d = tmpl::at_c<LhsTensorIndexTypeList, 3>::dim;

  // check LHS evaluated correctly
  for (size_t lhs_a = 0; lhs_a < dim_a; ++lhs_a) {
    for (size_t lhs_b = 0; lhs_b < dim_b; ++lhs_b) {
      for (size_t lhs_c = 0; lhs_c < dim_c; ++lhs_c) {
        for (size_t lhs_d = 0; lhs_d < dim_d; ++lhs_d) {
          const auto& expected_result =
              expected_L_abcd.get(lhs_a, lhs_b, lhs_c, lhs_d);

          CHECK(L_abcd.get(lhs_a, lhs_b, lhs_c, lhs_d) == expected_result);
          CHECK(L_abdc.get(lhs_a, lhs_b, lhs_d, lhs_c) == expected_result);
          CHECK(L_acbd.get(lhs_a, lhs_c, lhs_b, lhs_d) == expected_result);
          CHECK(L_acdb.get(lhs_a, lhs_c, lhs_d, lhs_b) == expected_result);
          CHECK(L_adbc.get(lhs_a, lhs_d, lhs_b, lhs_c) == expected_result);
          CHECK(L_adcb.get(lhs_a, lhs_d, lhs_c, lhs_b) == expected_result);
          CHECK(L_bacd.get(lhs_b, lhs_a, lhs_c, lhs_d) == expected_result);
          CHECK(L_badc.get(lhs_b, lhs_a, lhs_d, lhs_c) == expected_result);
          CHECK(L_bcad.get(lhs_b, lhs_c, lhs_a, lhs_d) == expected_result);
          CHECK(L_bcda.get(lhs_b, lhs_c, lhs_d, lhs_a) == expected_result);
          CHECK(L_bdac.get(lhs_b, lhs_d, lhs_a, lhs_c) == expected_result);
          CHECK(L_bdca.get(lhs_b, lhs_d, lhs_c, lhs_a) == expected_result);
          CHECK(L_cabd.get(lhs_c, lhs_a, lhs_b, lhs_d) == expected_result);
          CHECK(L_cadb.get(lhs_c, lhs_a, lhs_d, lhs_b) == expected_result);
          CHECK(L_cbad.get(lhs_c, lhs_b, lhs_a, lhs_d) == expected_result);
          CHECK(L_cbda.get(lhs_c, lhs_b, lhs_d, lhs_a) == expected_result);
          CHECK(L_cdab.get(lhs_c, lhs_d, lhs_a, lhs_b) == expected_result);
          CHECK(L_cdba.get(lhs_c, lhs_d, lhs_b, lhs_a) == expected_result);
          CHECK(L_dabc.get(lhs_d, lhs_a, lhs_b, lhs_c) == expected_result);
          CHECK(L_dacb.get(lhs_d, lhs_a, lhs_c, lhs_b) == expected_result);
          CHECK(L_dbac.get(lhs_d, lhs_b, lhs_a, lhs_c) == expected_result);
          CHECK(L_dbca.get(lhs_d, lhs_b, lhs_c, lhs_a) == expected_result);
          CHECK(L_dcab.get(lhs_d, lhs_c, lhs_a, lhs_b) == expected_result);
          CHECK(L_dcba.get(lhs_d, lhs_c, lhs_b, lhs_a) == expected_result);
        }
      }
    }
  }
}

}  // namespace TestHelpers::tenex
