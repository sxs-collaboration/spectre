// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <string>

#include "DataStructures/DataBox/Tag.hpp"
#include "DataStructures/DataBox/TagName.hpp"
#include "DataStructures/DataVector.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/Functional.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/Numeric.hpp"
#include "Utilities/TMPL.hpp"

namespace L2Norm_detail {
template <typename DataType, typename Symm, typename IndexList>
Scalar<DataType> pointwise_l2_norm_square(
    const Tensor<DataType, Symm, IndexList>& tensor) {
  auto pointwise_l2_normsq = make_with_value<Scalar<DataType>>(tensor, 0.);
  for (auto tensor_element = tensor.begin(); tensor_element != tensor.end();
       ++tensor_element) {
    // In order to handle tensory symmetries, we multiply the square of each
    // stored component with its multiplicity
    get(pointwise_l2_normsq) +=
        tensor.multiplicity(tensor_element) * square(*tensor_element);
  }
  return pointwise_l2_normsq;
}
}  // namespace L2Norm_detail

/// @{
/*!
 * \ingroup TensorGroup
 * \brief Compute point-wise Euclidean \f$L^2\f$-norm of arbitrary Tensors.
 *
 * \details
 * At each grid point \f$p\f$ in the element, this function computes the
 * point-wise Frobenius norm of a given Tensor with arbitrary rank. If the
 * Tensor \f$A\f$ has rank \f$n\f$ and dimensionality \f$D\f$, then its
 * Frobenius norm at point \f$p\f$ is computed as:
 *
 * \f{equation}
 * ||A||_2(p) =
 *    \left(\sum^{D-1}_{i_1=0}\sum^{D-1}_{i_2=0}\cdots \sum^{D-1}_{i_n=0}
 *          |A_{i_1 i_2 \cdots i_n}(p)|^2 \right)^{1/2},
 * \f}
 *
 * where both contra-variant and co-variant indices are shown as lower indices.
 */
template <typename DataType, typename Symm, typename IndexList>
Scalar<DataType> pointwise_l2_norm(
    const Tensor<DataType, Symm, IndexList>& tensor) {
  return Scalar<DataType>{
      sqrt(get(L2Norm_detail::pointwise_l2_norm_square(tensor)))};
}
template <typename DataType, typename Symm, typename IndexList>
void pointwise_l2_norm(const gsl::not_null<Scalar<DataType>*> norm,
                       const Tensor<DataType, Symm, IndexList>& tensor) {
  destructive_resize_components(norm, get_size(tensor[0].size()));
  get(*norm) = sqrt(get(L2Norm_detail::pointwise_l2_norm_square(tensor)));
}
/// @}

/*!
 * \ingroup TensorGroup
 * \brief Compute Euclidean \f$L^2\f$-norm of arbitrary Tensors reduced over an
 * element.
 *
 * \details
 * Computes the RMS value of the point-wise Frobenius norm of a given Tensor
 * with arbitrary rank over all grid points in an element. If the Tensor \f$A\f$
 * has rank \f$n\f$ and dimensionality \f$D\f$, and the element (of order
 * \f$N\f$) has \f$N+1\f$ points, then its element-reduced Frobenius norm is
 * computed as:
 *
 * \f{equation}
 *  ||A||_2 =
 *     \left(\frac{1}{N+1}\sum^{N}_{p=0}
 *          \left(\sum^{D-1}_{i_1=0}\sum^{D-1}_{i_2=0}\cdots
 *                \sum^{D-1}_{i_n=0} |A^p_{i_1 i_2 \cdots i_n}|^2 \right)
 * \right)^{1/2},
 * \f}
 *
 * where both contra-variant and co-variant indices are shown as lower indices,
 * and \f$p\f$ indexes grid points in the element.
 *
 * \warning This function reduces the Frobenius norm over the element, not the
 * whole domain.
 */
template <typename DataType, typename Symm, typename IndexList>
double l2_norm(const Tensor<DataType, Symm, IndexList>& tensor) {
  const auto pointwise_l2_normsq =
      L2Norm_detail::pointwise_l2_norm_square(tensor);
  using Plus = funcl::Plus<funcl::Identity>;
  return sqrt(alg::accumulate(get(pointwise_l2_normsq), 0., Plus{}) /
              tensor.begin()->size());
}

namespace Tags {
/*!
 * \ingroup DataBoxTagsGroup
 * \ingroup DataStructuresGroup
 * Point-wise Euclidean \f$L^2\f$-norm of a Tensor.
 * \see `pointwise_l2_norm()` for details.
 */
template <typename Tag>
struct PointwiseL2Norm : db::SimpleTag {
  using type = Scalar<typename Tag::type::type>;
  static std::string name() {
    return "PointwiseL2Norm(" + db::tag_name<Tag>() + ")";
  }
};

/*!
 * \ingroup DataBoxTagsGroup
 * \ingroup DataStructuresGroup
 * Computes the point-wise Euclidean \f$L^2\f$-norm of a Tensor.
 * \see `pointwise_l2_norm()` for details.
 */
template <typename Tag>
struct PointwiseL2NormCompute : PointwiseL2Norm<Tag>, db::ComputeTag {
  using base = PointwiseL2Norm<Tag>;
  using return_type = typename base::type;
  static constexpr auto function =
      static_cast<void (*)(const gsl::not_null<return_type*>,
                           const typename Tag::type&)>(&pointwise_l2_norm);
  using argument_tags = tmpl::list<Tag>;
};

/*!
 * \ingroup DataBoxTagsGroup
 * \ingroup DataStructuresGroup
 * Euclidean \f$L^2\f$-norm of a Tensor, RMS over grid points in element.
 * \see `l2_norm()` for details.
 */
template <typename Tag>
struct L2Norm : db::SimpleTag {
  using type = double;
  static std::string name() { return "L2Norm(" + db::tag_name<Tag>() + ")"; }
};

/*!
 * \ingroup DataBoxTagsGroup
 * \ingroup DataStructuresGroup
 * Computes the Euclidean \f$L^2\f$-norm of a Tensor, RMS over grid points in
 * element. \see `l2_norm()` for details.
 *
 * \warning This compute tag reduces the Frobenius norm over the element, not
 * the whole domain.
 */
template <typename Tag>
struct L2NormCompute : L2Norm<Tag>, db::ComputeTag {
  using base = L2Norm<Tag>;
  using return_type = typename base::type;
  static void function(const gsl::not_null<return_type*> norm,
                       const typename Tag::type& tensor) {
    *norm = l2_norm(tensor);
  }
  using argument_tags = tmpl::list<Tag>;
};
}  // namespace Tags
