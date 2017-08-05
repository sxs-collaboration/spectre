// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class templates ProductOf2Maps and ProductOf3Maps.

#pragma once

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp11.hpp"

namespace CoordinateMaps {

namespace product_detail {
template <typename T, size_t Size, typename Map1, typename Map2,
          typename Function, size_t... Is, size_t... Js>
std::array<T, Size> apply_call(const std::array<T, Size>& xi, const Map1& map1,
                               const Map2& map2, const Function func,
                               std::integer_sequence<size_t, Is...> /*meta*/,
                               std::integer_sequence<size_t, Js...> /*meta*/) {
  // Note: If the maps become higher than 3D we should check we are not wrapping
  // a reference_wrapper inside a reference_wrapper.
  return {{func(
               std::array<std::reference_wrapper<const T>, sizeof...(Is)>{
                   {xi[Is]...}},
               map1)[Is]...,
           func(
               std::array<std::reference_wrapper<const T>, sizeof...(Js)>{
                   {xi[Map1::dim + Js]...}},
               map2)[Js]...}};
}

template <typename T, size_t Size, typename Map1, typename Map2,
          typename Function, size_t... Is, size_t... Js>
Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<Size, UpLo::Up, Frame::NoFrame>,
                  SpatialIndex<Size, UpLo::Lo, Frame::NoFrame>>>
apply_jac(const std::array<T, Size>& xi, const Map1& map1, const Map2& map2,
          const Function func, std::integer_sequence<size_t, Is...> /*meta*/,
          std::integer_sequence<size_t, Js...> /*meta*/) {
  // Note: If the maps become higher than 3D we should check we are not wrapping
  // a reference_wrapper inside a reference_wrapper.
  auto map1_jac = func(
      std::array<std::reference_wrapper<const T>, sizeof...(Is)>{{xi[Is]...}},
      map1);
  auto map2_jac = func(
      std::array<std::reference_wrapper<const T>, sizeof...(Js)>{
          {xi[Map1::dim + Js]...}},
      map2);
  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<Size, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<Size, UpLo::Lo, Frame::NoFrame>>>
      jac{0.0};
  for (size_t i = 0; i < Map1::dim; ++i) {
    for (size_t j = 0; j < Map1::dim; ++j) {
      jac.get(i, j) = std::move(map1_jac.get(i, j));
    }
  }
  for (size_t i = 0; i < Map2::dim; ++i) {
    for (size_t j = 0; j < Map2::dim; ++j) {
      jac.get(Map1::dim + i, Map1::dim + j) = std::move(map2_jac.get(i, j));
    }
  }
  return jac;
}
}  // namespace product_detail

/// \ingroup CoordinateMaps
/// \brief Product of two codimension=0 CoordinateMaps.
///
/// \tparam Map1 the map for the first coordinate(s)
/// \tparam Map2 the map for the second coordinate(s)
template <typename Map1, typename Map2>
class ProductOf2Maps {
 public:
  static constexpr size_t dim = Map1::dim + Map2::dim;
  using map_list = tmpl::list<Map1, Map2>;
  static_assert(dim == 2 or dim == 3,
                "Only 2D and 3D maps are supported by ProductOf2Maps");

  // Needed for Charm++ serialization
  ProductOf2Maps() = default;

  ProductOf2Maps(Map1 map1, Map2 map2);

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, dim> operator()(
      const std::array<T, dim>& xi) const;

  template <typename T>
  std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>, dim> inverse(
      const std::array<T, dim>& x) const;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<dim, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, dim>& xi) const;

  template <typename T>
  Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
         tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<dim, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, dim>& xi) const;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p);  // NOLINT

 private:
  template <typename Map1Op, typename Map2Op>
  friend bool operator==(const ProductOf2Maps<Map1Op, Map2Op>& lhs,
                         const ProductOf2Maps<Map1Op, Map2Op>& rhs) noexcept {
    return lhs.map1_ == rhs.map1_ and lhs.map2_ == rhs.map2_;
  }

  Map1 map1_;
  Map2 map2_;
};

template <typename Map1, typename Map2>
ProductOf2Maps<Map1, Map2>::ProductOf2Maps(Map1 map1, Map2 map2)
    : map1_(std::move(map1)), map2_(std::move(map2)) {}

template <typename Map1, typename Map2>
template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>,
           ProductOf2Maps<Map1, Map2>::dim>
ProductOf2Maps<Map1, Map2>::operator()(const std::array<T, dim>& xi) const {
  return product_detail::apply_call(
      xi, map1_, map2_,
      [](const auto& point, const auto& map) { return map(point); },
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
std::array<std::decay_t<tt::remove_reference_wrapper_t<T>>,
           ProductOf2Maps<Map1, Map2>::dim>
ProductOf2Maps<Map1, Map2>::inverse(const std::array<T, dim>& x) const {
  return product_detail::apply_call(
      x, map1_, map2_,
      [](const auto& point, const auto& map) { return map.inverse(point); },
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<ProductOf2Maps<Map1, Map2>::dim, UpLo::Up,
                               Frame::NoFrame>,
                  SpatialIndex<ProductOf2Maps<Map1, Map2>::dim, UpLo::Lo,
                               Frame::NoFrame>>>
ProductOf2Maps<Map1, Map2>::inv_jacobian(const std::array<T, dim>& xi) const {
  return product_detail::apply_jac(xi, map1_, map2_,
                                   [](const auto& point, const auto& map) {
                                     return map.inv_jacobian(point);
                                   },
                                   std::make_index_sequence<Map1::dim>{},
                                   std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
template <typename T>
Tensor<std::decay_t<tt::remove_reference_wrapper_t<T>>,
       tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<ProductOf2Maps<Map1, Map2>::dim, UpLo::Up,
                               Frame::NoFrame>,
                  SpatialIndex<ProductOf2Maps<Map1, Map2>::dim, UpLo::Lo,
                               Frame::NoFrame>>>
ProductOf2Maps<Map1, Map2>::jacobian(const std::array<T, dim>& xi) const {
  return product_detail::apply_jac(
      xi, map1_, map2_,
      [](const auto& point, const auto& map) { return map.jacobian(point); },
      std::make_index_sequence<Map1::dim>{},
      std::make_index_sequence<Map2::dim>{});
}

template <typename Map1, typename Map2>
void ProductOf2Maps<Map1, Map2>::pup(PUP::er& p) {
  p | map1_;
  p | map2_;
}

template <typename Map1, typename Map2>
bool operator!=(const ProductOf2Maps<Map1, Map2>& lhs,
                const ProductOf2Maps<Map1, Map2>& rhs) noexcept {
  return not(lhs == rhs);
}

/// \ingroup CoordinateMaps
/// \brief Product of three one-dimensional CoordinateMaps.
template <typename Map1, typename Map2, typename Map3>
class ProductOf3Maps {
 public:
  static constexpr size_t dim = Map1::dim + Map2::dim + Map3::dim;
  using map_list = tmpl::list<Map1, Map2, Map3>;
  static_assert(dim == 3, "Only 3D maps are implemented for ProductOf3Maps");

  // Needed for Charm++ serialization
  ProductOf3Maps() = default;

  ProductOf3Maps(Map1 map1, Map2 map2, Map3 map3);

  template <typename T>
  std::array<T, dim> operator()(const std::array<T, dim>& xi) const;

  template <typename T>
  std::array<T, dim> inverse(const std::array<T, dim>& x) const;

  template <typename T>
  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<dim, UpLo::Lo, Frame::NoFrame>>>
  inv_jacobian(const std::array<T, dim>& xi) const;

  template <typename T>
  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<dim, UpLo::Lo, Frame::NoFrame>>>
  jacobian(const std::array<T, dim>& xi) const;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& p);  // NOLINT

 private:
  template <typename Map1Op, typename Map2Op, typename Map3Op>
  friend bool operator==(
      const ProductOf3Maps<Map1Op, Map2Op, Map3Op>& lhs,
      const ProductOf3Maps<Map1Op, Map2Op, Map3Op>& rhs) noexcept {
    return lhs.map1_ == rhs.map1_ and lhs.map2_ == rhs.map2_ and
           lhs.map3_ == rhs.map3_;
  }

  Map1 map1_;
  Map2 map2_;
  Map3 map3_;
};

template <typename Map1, typename Map2, typename Map3>
ProductOf3Maps<Map1, Map2, Map3>::ProductOf3Maps(Map1 map1, Map2 map2,
                                                 Map3 map3)
    : map1_(std::move(map1)), map2_(std::move(map2)), map3_(std::move(map3)) {}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
std::array<T, ProductOf3Maps<Map1, Map2, Map3>::dim>
ProductOf3Maps<Map1, Map2, Map3>::operator()(
    const std::array<T, dim>& xi) const {
  return {{map1_(std::array<std::reference_wrapper<const T>, 1>{{xi[0]}})[0],
           map2_(std::array<std::reference_wrapper<const T>, 1>{{xi[1]}})[0],
           map3_(std::array<std::reference_wrapper<const T>, 1>{{xi[2]}})[0]}};
}

template <typename Map1, typename Map2, typename Map3>
template <typename T>
std::array<T, ProductOf3Maps<Map1, Map2, Map3>::dim>
ProductOf3Maps<Map1, Map2, Map3>::inverse(const std::array<T, dim>& x) const {
  return {
      {map1_.inverse(std::array<std::reference_wrapper<const T>, 1>{{x[0]}})[0],
       map2_.inverse(std::array<std::reference_wrapper<const T>, 1>{{x[1]}})[0],
       map3_.inverse(
           std::array<std::reference_wrapper<const T>, 1>{{x[2]}})[0]}};
}
template <typename Map1, typename Map2, typename Map3>
template <typename T>
Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<ProductOf3Maps<Map1, Map2, Map3>::dim, UpLo::Up,
                               Frame::NoFrame>,
                  SpatialIndex<ProductOf3Maps<Map1, Map2, Map3>::dim, UpLo::Lo,
                               Frame::NoFrame>>>
ProductOf3Maps<Map1, Map2, Map3>::inv_jacobian(
    const std::array<T, dim>& xi) const {
  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<dim, UpLo::Lo, Frame::NoFrame>>>
      inv_jac{0.0};
  inv_jac.template get<0, 0>() = map1_.inv_jacobian(
      std::array<std::reference_wrapper<const T>, 1>{{xi[0]}})[0];
  inv_jac.template get<1, 1>() = map2_.inv_jacobian(
      std::array<std::reference_wrapper<const T>, 1>{{xi[1]}})[0];
  inv_jac.template get<2, 2>() = map3_.inv_jacobian(
      std::array<std::reference_wrapper<const T>, 1>{{xi[2]}})[0];
  return inv_jac;
}
template <typename Map1, typename Map2, typename Map3>
template <typename T>
Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
       index_list<SpatialIndex<ProductOf3Maps<Map1, Map2, Map3>::dim, UpLo::Up,
                               Frame::NoFrame>,
                  SpatialIndex<ProductOf3Maps<Map1, Map2, Map3>::dim, UpLo::Lo,
                               Frame::NoFrame>>>
ProductOf3Maps<Map1, Map2, Map3>::jacobian(const std::array<T, dim>& xi) const {
  Tensor<T, tmpl::integral_list<std::int32_t, 2, 1>,
         index_list<SpatialIndex<dim, UpLo::Up, Frame::NoFrame>,
                    SpatialIndex<dim, UpLo::Lo, Frame::NoFrame>>>
      jac{0.0};
  jac.template get<0, 0>() = map1_.jacobian(
      std::array<std::reference_wrapper<const T>, 1>{{xi[0]}})[0];
  jac.template get<1, 1>() = map2_.jacobian(
      std::array<std::reference_wrapper<const T>, 1>{{xi[1]}})[0];
  jac.template get<2, 2>() = map3_.jacobian(
      std::array<std::reference_wrapper<const T>, 1>{{xi[2]}})[0];
  return jac;
}
template <typename Map1, typename Map2, typename Map3>
void ProductOf3Maps<Map1, Map2, Map3>::pup(PUP::er& p) {
  p | map1_;
  p | map2_;
  p | map3_;
}

template <typename Map1, typename Map2, typename Map3>
bool operator!=(const ProductOf3Maps<Map1, Map2, Map3>& lhs,
                const ProductOf3Maps<Map1, Map2, Map3>& rhs) noexcept {
  return not(lhs == rhs);
}
}  // namespace CoordinateMaps
