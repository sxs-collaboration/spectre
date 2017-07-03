
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
  return Point<2, Frame::Grid>{{(*map1_)(Point<1, Frame::Logical>(xi[0]))[0],
                                (*map2_)(Point<1, Frame::Logical>(xi[1]))[0]}};
}

template <>
Point<2, Frame::Logical> ProductOf2Maps<1, 1>::inverse(
    const Point<2, Frame::Grid>& x) const {
  return Point<2, Frame::Logical>{
      {(map1_->inverse(Point<1, Frame::Grid>(x[0]))).get(0),
       (map2_->inverse(Point<1, Frame::Grid>(x[1]))).get(0)}};
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
  return map2_->inv_jacobian(Point<1, Frame::Logical>(xi[1]), 0, 0);
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

std::unique_ptr<EmbeddingMap<3, 3>> ProductOf3Maps::get_clone() const {
  return std::make_unique<ProductOf3Maps>(*this);
}

Point<3, Frame::Grid> ProductOf3Maps::operator()(
    const Point<3, Frame::Logical>& xi) const {
  return Point<3, Frame::Grid>{{(*map1_)(Point<1, Frame::Logical>(xi[0]))[0],
                                (*map2_)(Point<1, Frame::Logical>(xi[1]))[0],
                                (*map3_)(Point<1, Frame::Logical>(xi[2]))[0]}};
}

Point<3, Frame::Logical> ProductOf3Maps::inverse(
    const Point<3, Frame::Grid>& x) const {
  return Point<3, Frame::Logical>{
      {map1_->inverse(Point<1, Frame::Grid>(x[0]))[0],
       map2_->inverse(Point<1, Frame::Grid>(x[1]))[0],
       map3_->inverse(Point<1, Frame::Grid>(x[2]))[0]}};
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
  return map3_->inv_jacobian(Point<1, Frame::Logical>(xi[2]), 0, 0);
}

}  // namespace EmbeddingMaps

/// \cond HIDDEN_SYMBOLS
PUP::able::PUP_ID EmbeddingMaps::ProductOf3Maps::my_PUP_ID =  // NOLINT
    0;
/// \endcond
