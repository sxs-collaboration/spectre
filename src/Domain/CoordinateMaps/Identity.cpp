// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/CoordinateMaps/Identity.hpp"

#include "DataStructures/MakeWithValue.hpp"
#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"
#include "Utilities/DereferenceWrapper.hpp"

namespace CoordinateMaps {

template <size_t Dim>
template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, Dim> Identity<Dim>::
operator()(const std::array<T, Dim>& source_coords) const {
  return make_array<std::decay_t<tt::remove_reference_wrapper_t<T>>, Dim>(
      source_coords);
}

template <size_t Dim>
template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, Dim>
Identity<Dim>::inverse(const std::array<T, Dim>& target_coords) const {
  return make_array<std::decay_t<tt::remove_reference_wrapper_t<T>>, Dim>(
      target_coords);
}

template <size_t Dim>
template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<Dim, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<Dim, UpLo::Lo, Frame::NoFrame>>>
Identity<Dim>::jacobian(const std::array<T, Dim>& source_coords) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<Dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<Dim, UpLo::Lo, Frame::NoFrame>>>
      jac{make_with_value<std::decay_t<tt::remove_reference_wrapper_t<T>>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  for (size_t i = 0; i < Dim; ++i) {
    jac.get(i, i) = 1.0;
  }
  return jac;
}

template <size_t Dim>
template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<Dim, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<Dim, UpLo::Lo, Frame::NoFrame>>>
Identity<Dim>::inv_jacobian(const std::array<T, Dim>& source_coords) const {
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<Dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<Dim, UpLo::Lo, Frame::NoFrame>>>
      inv_jac{make_with_value<std::decay_t<tt::remove_reference_wrapper_t<T>>>(
          dereference_wrapper(source_coords[0]), 0.0)};
  for (size_t i = 0; i < Dim; ++i) {
    inv_jac.get(i, i) = 1.0;
  }
  return inv_jac;
}

template class Identity<1>;
template class Identity<2>;
// Identity should only be used in ProductMaps if a particular dimension is
// unaffected.  So if the largest dim we do is 3, then you should never use
// Identity<3>

/// \cond HIDDEN_SYMBOLS
template std::array<double, 1> Identity<1>::operator()(
    const std::array<std::reference_wrapper<const double>, 1>& source_coords)
    const;
template std::array<double, 1> Identity<1>::operator()(
    const std::array<double, 1>& source_coords) const;
template std::array<DataVector, 1> Identity<1>::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 1>&
        source_coords) const;
template std::array<DataVector, 1> Identity<1>::operator()(
    const std::array<DataVector, 1>& source_coords) const;

template std::array<double, 1> Identity<1>::inverse(
    const std::array<std::reference_wrapper<const double>, 1>& target_coords)
    const;
template std::array<double, 1> Identity<1>::inverse(
    const std::array<double, 1>& target_coords) const;
template std::array<DataVector, 1> Identity<1>::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 1>&
        target_coords) const;
template std::array<DataVector, 1> Identity<1>::inverse(
    const std::array<DataVector, 1>& target_coords) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Identity<1>::jacobian(const std::array<std::reference_wrapper<const double>, 1>&
                          source_coords) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Identity<1>::jacobian(const std::array<double, 1>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Identity<1>::jacobian(const std::array<std::reference_wrapper<const DataVector>,
                                       1>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Identity<1>::jacobian(const std::array<DataVector, 1>& source_coords) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Identity<1>::inv_jacobian(const std::array<std::reference_wrapper<const double>,
                                           1>& source_coords) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Identity<1>::inv_jacobian(const std::array<double, 1>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Identity<1>::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 1>&
        source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<1, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<1, UpLo::Lo, Frame::NoFrame>>>
Identity<1>::inv_jacobian(const std::array<DataVector, 1>& source_coords) const;

template std::array<double, 2> Identity<2>::operator()(
    const std::array<std::reference_wrapper<const double>, 2>& source_coords)
    const;
template std::array<double, 2> Identity<2>::operator()(
    const std::array<double, 2>& source_coords) const;
template std::array<DataVector, 2> Identity<2>::operator()(
    const std::array<std::reference_wrapper<const DataVector>, 2>&
        source_coords) const;
template std::array<DataVector, 2> Identity<2>::operator()(
    const std::array<DataVector, 2>& source_coords) const;

template std::array<double, 2> Identity<2>::inverse(
    const std::array<std::reference_wrapper<const double>, 2>& target_coords)
    const;
template std::array<double, 2> Identity<2>::inverse(
    const std::array<double, 2>& target_coords) const;
template std::array<DataVector, 2> Identity<2>::inverse(
    const std::array<std::reference_wrapper<const DataVector>, 2>&
        target_coords) const;
template std::array<DataVector, 2> Identity<2>::inverse(
    const std::array<DataVector, 2>& target_coords) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Identity<2>::jacobian(const std::array<std::reference_wrapper<const double>, 2>&
                          source_coords) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Identity<2>::jacobian(const std::array<double, 2>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Identity<2>::jacobian(const std::array<std::reference_wrapper<const DataVector>,
                                       2>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Identity<2>::jacobian(const std::array<DataVector, 2>& source_coords) const;

template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Identity<2>::inv_jacobian(const std::array<std::reference_wrapper<const double>,
                                           2>& source_coords) const;
template Tensor<double, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Identity<2>::inv_jacobian(const std::array<double, 2>& source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Identity<2>::inv_jacobian(
    const std::array<std::reference_wrapper<const DataVector>, 2>&
        source_coords) const;
template Tensor<DataVector, tmpl::integral_list<std::int32_t, 2, 1>,
                index_list<SpatialIndex<2, UpLo::Up, Frame::NoFrame>,
                           SpatialIndex<2, UpLo::Lo, Frame::NoFrame>>>
Identity<2>::inv_jacobian(const std::array<DataVector, 2>& source_coords) const;
/// \endcond
}  // namespace CoordinateMaps
