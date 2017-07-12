// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Domain/EmbeddingMaps/Identity.hpp"

#include "DataStructures/Tensor/Tensor.hpp"
#include "ErrorHandling/Assert.hpp"

namespace EmbeddingMaps {

template <size_t Dim>
Point<Dim, Frame::Grid> Identity<Dim>::operator()(
    const Point<Dim, Frame::Logical>& xi) const {
  Point<Dim, Frame::Grid> x{};
  std::copy(xi.begin(), xi.end(), x.begin());
  return x;
}

template <size_t Dim>
Point<Dim, Frame::Logical> Identity<Dim>::inverse(
    const Point<Dim, Frame::Grid>& x) const {
  Point<Dim, Frame::Logical> xi{};
  std::copy(x.begin(), x.end(), xi.begin());
  return xi;
}

template <size_t Dim>
double Identity<Dim>::jacobian(const Point<Dim, Frame::Logical>& /* xi */,
                               const size_t ud, const size_t ld) const {
  ASSERT(ld < Dim, "ld = " << ld);
  ASSERT(ud < Dim, "ud = " << ud);
  return (ld == ud ? 1.0 : 0.0);
}

template <size_t Dim>
double Identity<Dim>::inv_jacobian(const Point<Dim, Frame::Logical>& /* xi */,
                                   const size_t ud, const size_t ld) const {
  ASSERT(ld < Dim, "ld = " << ld);
  ASSERT(ud < Dim, "ud = " << ud);
  return (ld == ud ? 1.0 : 0.0);
}

template <size_t Dim>
std::unique_ptr<EmbeddingMap<Dim, Dim>> Identity<Dim>::get_clone() const {
  return std::make_unique<Identity<Dim>>();
}

template <size_t Dim>
void Identity<Dim>::pup(PUP::er& p) {
  EmbeddingMap<Dim, Dim>::pup(p);
}  // NOLINT

template class Identity<1>;
template class Identity<2>;
// Identity should only be used in ProductMaps if a particular dimension is
// unaffected.  So if the largest dim we do is 3, then you should never use
// Identity<3>
}  // namespace EmbeddingMaps
