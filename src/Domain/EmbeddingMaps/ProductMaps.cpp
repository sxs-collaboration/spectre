// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/EmbeddingMaps/ProductMaps.hpp"
#include "ErrorHandling/Error.hpp"

namespace EmbeddingMaps {

template <size_t Dim1, size_t Dim2>
ProductOf2Maps<Dim1, Dim2>::ProductOf2Maps(const EmbeddingMap<Dim1, Dim1>& map1,
                                           const EmbeddingMap<Dim2, Dim2>& map2)
    : EmbeddingMap<Dim1 + Dim2, Dim1 + Dim2>(),
      map1_(map1.get_clone()),
      map2_(map2.get_clone()) {}

template <size_t Dim1, size_t Dim2>
ProductOf2Maps<Dim1, Dim2>::ProductOf2Maps(
    const ProductOf2Maps<Dim1, Dim2>& other)
    : EmbeddingMap<Dim1 + Dim2, Dim1 + Dim2>(),
      map1_(other.map1_->get_clone()),
      map2_(other.map2_->get_clone()) {}

template <>
Point<2, Frame::Grid> ProductOf2Maps<1, 1>::operator()(
    const Point<2, Frame::Logical>& xi) const {
  return Point<2, Frame::Grid>{
      {{(*map1_)(Point<1, Frame::Logical>(xi[0]))[0],
        (*map2_)(Point<1, Frame::Logical>(xi[1]))[0]}}};
}

template <>
Point<2, Frame::Logical> ProductOf2Maps<1, 1>::inverse(
    const Point<2, Frame::Grid>& x) const {
  return Point<2, Frame::Logical>{
      {{(map1_->inverse(Point<1, Frame::Grid>(x[0]))).get(0),
        (map2_->inverse(Point<1, Frame::Grid>(x[1]))).get(0)}}};
}

template <>
double ProductOf2Maps<1, 1>::jacobian(const Point<2, Frame::Logical>& xi,
                                      const size_t ud, const size_t ld) const {
  if (ud != ld) {
    return 0.;
  }
  if (0 == ld) {
    return map1_->jacobian(Point<1, Frame::Logical>(xi[0]), 0, 0);
  }
  ASSERT(1 == ld, "ld = " << ld << "in jacobian, should be 0 or 1");
  return map2_->jacobian(Point<1, Frame::Logical>(xi[1]), 0, 0);
}

template <>
double ProductOf2Maps<1, 1>::inv_jacobian(const Point<2, Frame::Logical>& xi,
                                          const size_t ud,
                                          const size_t ld) const {
  if (ud != ld) {
    return 0.;
  }
  if (0 == ld) {
    return map1_->inv_jacobian(Point<1, Frame::Logical>(xi[0]), 0, 0);
  }
  ASSERT(1 == ld, "ld = " << ld << "in jacobian, should be 0 or 1");
  return map2_->inv_jacobian(Point<1, Frame::Logical>(xi[1]), 0, 0);
}

template <>
ProductOf2Maps<1, 1>::ProductOf2Maps(CkMigrateMessage* /* m */)
    : map1_(nullptr), map2_(nullptr) {}

template <>
void ProductOf2Maps<1, 1>::pup(PUP::er& p) {  // NOLINT
  EmbeddingMap<2, 2>::pup(p);
  p | map1_;
  p | map2_;
}

template class ProductOf2Maps<1, 1>;

ProductOf3Maps::ProductOf3Maps(const EmbeddingMap<1, 1>& map1,
                               const EmbeddingMap<1, 1>& map2,
                               const EmbeddingMap<1, 1>& map3)
    : EmbeddingMap<3, 3>(),
      map1_(map1.get_clone()),
      map2_(map2.get_clone()),
      map3_(map3.get_clone()) {}

ProductOf3Maps::ProductOf3Maps(const ProductOf3Maps& other)
    : EmbeddingMap<3, 3>(),
      map1_(other.map1_->get_clone()),
      map2_(other.map2_->get_clone()),
      map3_(other.map3_->get_clone()) {}

Point<3, Frame::Grid> ProductOf3Maps::operator()(
    const Point<3, Frame::Logical>& xi) const {
  return Point<3, Frame::Grid>{
      {{(*map1_)(Point<1, Frame::Logical>(xi[0]))[0],
        (*map2_)(Point<1, Frame::Logical>(xi[1]))[0],
        (*map3_)(Point<1, Frame::Logical>(xi[2]))[0]}}};
}

Point<3, Frame::Logical> ProductOf3Maps::inverse(
    const Point<3, Frame::Grid>& x) const {
  return Point<3, Frame::Logical>{
      {{map1_->inverse(Point<1, Frame::Grid>(x[0]))[0],
        map2_->inverse(Point<1, Frame::Grid>(x[1]))[0],
        map3_->inverse(Point<1, Frame::Grid>(x[2]))[0]}}};
}

double ProductOf3Maps::jacobian(const Point<3, Frame::Logical>& xi,
                                const size_t ud, const size_t ld) const {
  if (ud != ld) {
    return 0.;
  }
  if (0 == ld) {
    return map1_->jacobian(Point<1, Frame::Logical>(xi[0]), 0, 0);
  }
  if (1 == ld) {
    return map2_->jacobian(Point<1, Frame::Logical>(xi[1]), 0, 0);
  }
  ASSERT(2 == ld, "ld = " << ld << "in jacobian, should be 0,1 or 2");
  return map3_->jacobian(Point<1, Frame::Logical>(xi[2]), 0, 0);
}

double ProductOf3Maps::inv_jacobian(const Point<3, Frame::Logical>& xi,
                                    const size_t ud, const size_t ld) const {
  if (ud != ld) {
    return 0.;
  }
  if (0 == ld) {
    return map1_->inv_jacobian(Point<1, Frame::Logical>(xi[0]), 0, 0);
  }
  if (1 == ld) {
    return map2_->inv_jacobian(Point<1, Frame::Logical>(xi[1]), 0, 0);
  }
  ASSERT(2 == ld, "ld = " << ld << "in jacobian, should be 0,1 or 2");
  return map3_->inv_jacobian(Point<1, Frame::Logical>(xi[2]), 0, 0);
}

ProductOf3Maps::ProductOf3Maps(CkMigrateMessage* /* m */)
    : map1_(nullptr), map2_(nullptr), map3_(nullptr) {}

void ProductOf3Maps::pup(PUP::er& p) {
  EmbeddingMap<3, 3>::pup(p);
  p | map1_;
  p | map2_;
  p | map3_;
}

}  // namespace EmbeddingMaps

/// \cond HIDDEN_SYMBOLS
PUP::able::PUP_ID EmbeddingMaps::ProductOf3Maps::my_PUP_ID =  // NOLINT
    0;
/// \endcond
