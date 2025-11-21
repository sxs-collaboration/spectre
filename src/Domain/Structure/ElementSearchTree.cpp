// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/Structure/ElementSearchTree.hpp"

#include <array>
#include <boost/geometry.hpp>
#include <boost/geometry/index/rtree.hpp>
#include <cstddef>
#include <map>
#include <memory>
#include <type_traits>
#include <vector>

#include "DataStructures/Tensor/Tensor.hpp"
#include "Domain/Structure/ElementId.hpp"
#include "Domain/Structure/Side.hpp"
#include "Utilities/GenerateInstantiations.hpp"

namespace boost::geometry::traits {
// Make Tensor compatible with Boost.Geometry
// This is needed to search for a block-logical coordinate given as a Tensor
// in the `ElementSearchTree`.
template <typename DataType, size_t Dim, typename Frame>
struct tag<tnsr::I<DataType, Dim, Frame>> {
  using type = boost::geometry::point_tag;
};
template <typename DataType, size_t Dim, typename Frame>
struct coordinate_type<tnsr::I<DataType, Dim, Frame>> {
  using type = DataType;
};
template <typename DataType, size_t Dim, typename Frame>
struct coordinate_system<tnsr::I<DataType, Dim, Frame>> {
  using type = boost::geometry::cs::cartesian;
};
template <typename DataType, size_t Dim, typename Frame>
struct dimension<tnsr::I<DataType, Dim, Frame>>
    : std::integral_constant<size_t, Dim> {};
template <typename DataType, size_t Dim, typename Frame, size_t GetDimension>
struct access<tnsr::I<DataType, Dim, Frame>, GetDimension> {
  static constexpr DataType get(const tnsr::I<DataType, Dim, Frame>& point) {
    return ::get<GetDimension>(point);
  }
  static void set(tnsr::I<DataType, Dim, Frame>& point, const DataType& value) {
    ::get<GetDimension>(point) = value;
  }
};
// Make ElementId compatible with Boost.Geometry
// Each ElementId defines a bounding box in block-logical coordinates, which is
// used to search for elements in the `ElementSearchTree`.
template <size_t Dim>
struct tag<ElementId<Dim>> {
  using type = boost::geometry::box_tag;
};
template <size_t Dim>
struct coordinate_type<ElementId<Dim>> {
  using type = double;
};
template <size_t Dim>
struct coordinate_system<ElementId<Dim>> {
  using type = boost::geometry::cs::cartesian;
};
template <size_t Dim>
struct dimension<ElementId<Dim>> : std::integral_constant<size_t, Dim> {};
template <size_t Dim>
struct point_type<ElementId<Dim>> {
  using type = tnsr::I<double, Dim, ::Frame::BlockLogical>;
};
template <size_t Dim, size_t Index, size_t GetDimension>
struct indexed_access<ElementId<Dim>, Index, GetDimension> {
  static constexpr double get(const ElementId<Dim>& element_id) {
    if constexpr (Index == 0) {
      return element_id.segment_id(GetDimension).endpoint(Side::Lower);
    } else {
      return element_id.segment_id(GetDimension).endpoint(Side::Upper);
    }
  }
};
}  // namespace boost::geometry::traits

namespace domain {
namespace {
template <size_t Dim>
using SearchTreeGeometryImpl =
    boost::geometry::index::rtree<ElementId<Dim>,
                                  boost::geometry::index::quadratic<16>>;

template <size_t Dim>
using GeometryIterator =
    typename SearchTreeGeometryImpl<Dim>::const_query_iterator;

template <size_t Dim, size_t N>
GeometryIterator<Dim>& underlying_iterator(std::array<char, N>& data) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return *reinterpret_cast<GeometryIterator<Dim>*>(data.data());
}

template <size_t Dim, size_t N>
const GeometryIterator<Dim>& underlying_iterator(
    const std::array<char, N>& data) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  return *reinterpret_cast<const GeometryIterator<Dim>*>(data.data());
}
}  // namespace

template <size_t Dim>
ElementSearchTreeIterator<Dim>::ElementSearchTreeIterator() : data_{} {
  static_assert(decltype(data_){}.size() >= sizeof(GeometryIterator<Dim>));
  new (data_.data()) GeometryIterator<Dim>{};
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init) - false positive
template <size_t Dim>
ElementSearchTreeIterator<Dim>::ElementSearchTreeIterator(
    ElementSearchTreeIterator&& other)
    : ElementSearchTreeIterator() {
  underlying_iterator<Dim>(data_) =
      std::move(underlying_iterator<Dim>(other.data_));
}

// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init) - false positive
template <size_t Dim>
ElementSearchTreeIterator<Dim>::ElementSearchTreeIterator(
    const ElementSearchTreeIterator& other)
    : ElementSearchTreeIterator() {
  underlying_iterator<Dim>(data_) = underlying_iterator<Dim>(other.data_);
}

template <size_t Dim>
ElementSearchTreeIterator<Dim>& ElementSearchTreeIterator<Dim>::operator=(
    ElementSearchTreeIterator&& other) {
  underlying_iterator<Dim>(data_) =
      std::move(underlying_iterator<Dim>(other.data_));
  return *this;
}

template <size_t Dim>
ElementSearchTreeIterator<Dim>& ElementSearchTreeIterator<Dim>::operator=(
    const ElementSearchTreeIterator& other) {
  underlying_iterator<Dim>(data_) = underlying_iterator<Dim>(other.data_);
  return *this;
}

template <size_t Dim>
ElementSearchTreeIterator<Dim>::~ElementSearchTreeIterator() {
  // This stupid-looking way of writing the destructor works around an
  // nvcc bug.
  []<typename T>(T& iter) { iter.~T(); }(underlying_iterator<Dim>(data_));
}

template <size_t Dim>
auto ElementSearchTreeIterator<Dim>::operator*() const -> reference {
  return *underlying_iterator<Dim>(data_);
}

template <size_t Dim>
auto ElementSearchTreeIterator<Dim>::operator->() const -> pointer {
  return underlying_iterator<Dim>(data_).operator->();
}

template <size_t Dim>
ElementSearchTreeIterator<Dim>& ElementSearchTreeIterator<Dim>::operator++() {
  ++underlying_iterator<Dim>(data_);
  return *this;
}

template <size_t Dim>
ElementSearchTreeIterator<Dim> ElementSearchTreeIterator<Dim>::operator++(int) {
  auto result = *this;
  ++*this;
  return result;
}

template <size_t Dim>
template <typename T>
ElementSearchTreeIterator<Dim> ElementSearchTreeIterator<Dim>::from_impl(
    T impl) {
  static_assert(std::is_same_v<T, GeometryIterator<Dim>>);
  ElementSearchTreeIterator<Dim> result{};
  underlying_iterator<Dim>(result.data_) = std::move(impl);
  return result;
}

template <size_t Dim2>
bool operator==(const ElementSearchTreeIterator<Dim2>& a,
                const ElementSearchTreeIterator<Dim2>& b) {
  return underlying_iterator<Dim2>(a.data_) ==
         underlying_iterator<Dim2>(b.data_);
}

template <size_t Dim>
ElementSearchTree<Dim>::ElementSearchTree() : impl_(std::make_unique<Impl>()) {}

template <size_t Dim>
ElementSearchTree<Dim>::ElementSearchTree(ElementSearchTree<Dim>&&) = default;

template <size_t Dim>
ElementSearchTree<Dim>& ElementSearchTree<Dim>::operator=(
    ElementSearchTree<Dim>&&) = default;

template <size_t Dim>
ElementSearchTree<Dim>::~ElementSearchTree() = default;

template <size_t Dim>
size_t ElementSearchTree<Dim>::size() const {
  return impl_->size();
}

template <size_t Dim>
bool ElementSearchTree<Dim>::empty() const {
  return impl_->empty();
}

template <size_t Dim>
void ElementSearchTree<Dim>::clear() {
  return impl_->clear();
}

template <size_t Dim>
void ElementSearchTree<Dim>::insert(const ElementId<Dim>& id) {
  impl_->insert(id);
}

template <size_t Dim>
ElementSearchTreeIterator<Dim> ElementSearchTree<Dim>::begin_covers(
    const tnsr::I<double, Dim, Frame::BlockLogical>& coords) const {
  return ElementSearchTreeIterator<Dim>::from_impl(
      impl_->qbegin(boost::geometry::index::covers(coords)));
}

template <size_t Dim>
ElementSearchTreeIterator<Dim> ElementSearchTree<Dim>::end_covers() const {
  return ElementSearchTreeIterator<Dim>::from_impl(impl_->qend());
}

template <size_t Dim>
struct ElementSearchTree<Dim>::Impl : SearchTreeGeometryImpl<Dim> {};

template <size_t Dim>
std::map<size_t, ElementSearchTree<Dim>> index_element_ids(
    const std::vector<ElementId<Dim>>& element_ids) {
  std::map<size_t, ElementSearchTree<Dim>> trees{};
  for (const auto& element_id : element_ids) {
    trees[element_id.block_id()].insert(element_id);
  }
  return trees;
}

#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)

#define INSTANTIATE(_, data)                                                 \
  template class ElementSearchTreeIterator<DIM(data)>;                       \
  template class ElementSearchTree<DIM(data)>;                               \
  template std::map<size_t, ElementSearchTree<DIM(data)>> index_element_ids( \
      const std::vector<ElementId<DIM(data)>>& element_ids);                 \
  template bool operator==(const ElementSearchTreeIterator<DIM(data)>& a,    \
                           const ElementSearchTreeIterator<DIM(data)>& b);

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3))

#undef INSTANTIATE
#undef DIM
}  // namespace domain
