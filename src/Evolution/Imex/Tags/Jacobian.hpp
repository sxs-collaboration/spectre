// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/Tensor/EagerMath/OuterProduct.hpp"
#include "DataStructures/Tensor/Metafunctions.hpp"

namespace imex {
namespace Tags {
/*!
 * Tag for the derivative of an implicit source (`Dependent`) with
 * respect to an implicit variable (`Independent`).
 *
 * The independent indices are given first, so
 * \f{equation}
 *   \frac{S(U_{j...})}{U_{i...}} = J_{i\ldots,j\ldots}
 * \f}
 * with appropriate upper and lower indices.
 */
template <typename Independent, typename Dependent>
struct Jacobian : db::SimpleTag {
  static_assert(std::is_same_v<typename Independent::type::value_type,
                               typename Dependent::type::value_type>);
  using independent = Independent;
  using dependent = Dependent;

 private:
  using Denominator =
      TensorMetafunctions::change_all_valences<typename Independent::type>;

 public:
  using type = OuterProductResultTensor<
      typename Dependent::type::value_type, typename Denominator::symmetry,
      typename Denominator::index_list, typename Dependent::type::symmetry,
      typename Dependent::type::index_list>;
};
}  // namespace Tags

/// Create a list of all jacobian tags for the dependence of \p
/// DependentList on \p IndependentList.
///
/// This can be used to construct the argument for a `Variables` from
/// the `tags_list`s of two other `Variables`.
template <typename IndependentList, typename DependentList>
using jacobian_tags = tmpl::join<tmpl::transform<
    IndependentList,
    tmpl::lazy::transform<
        tmpl::pin<DependentList>,
        tmpl::defer<tmpl::bind<::imex::Tags::Jacobian, tmpl::parent<tmpl::_1>,
                               tmpl::_1>>>>>;
}  // namespace imex
