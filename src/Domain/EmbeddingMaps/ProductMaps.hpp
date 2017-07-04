// Distributed under the MIT License.
// See LICENSE.txt for details.

/// \file
/// Defines the class templates ProductOf2Maps and ProductOf3Maps.

#pragma once

#include <memory>

#include "DataStructures/Tensor/TypeAliases.hpp"
#include "EmbeddingMap.hpp"
#include "Parallel/CharmPupable.hpp"
#include "Parallel/PupStlCpp11.hpp"

namespace EmbeddingMaps {

/// \ingroup EmbeddingMaps
/// \brief Product of two codimension=0 EmbeddingMaps.
///
/// \tparam Dim1 dimension of the domain and range of the first mapping.
/// \tparam Dim2 dimension of the domain and range of the second mapping.
template <size_t Dim1, size_t Dim2>
class ProductOf2Maps final : public EmbeddingMap<Dim1 + Dim2, Dim1 + Dim2> {
 public:
  ProductOf2Maps(const EmbeddingMap<Dim1, Dim1>& map_1,
                 const EmbeddingMap<Dim2, Dim2>& map_2);
  ProductOf2Maps() = default;
  ~ProductOf2Maps() override = default;
  ProductOf2Maps(const ProductOf2Maps<Dim1, Dim2>& other);
  ProductOf2Maps<Dim1, Dim2>& operator=(const ProductOf2Maps<Dim1, Dim2>&) =
      delete;
  ProductOf2Maps(ProductOf2Maps<Dim1, Dim2>&&) noexcept = default;  // NOLINT
  ProductOf2Maps<Dim1, Dim2>& operator=(ProductOf2Maps<Dim1, Dim2>&&) = delete;

  std::unique_ptr<EmbeddingMap<Dim1 + Dim2, Dim1 + Dim2>> get_clone()
      const override {
    return std::make_unique<ProductOf2Maps<Dim1, Dim2>>(*this);
  }

  Point<Dim1 + Dim2, Frame::Grid> operator()(
      const Point<Dim1 + Dim2, Frame::Logical>& xi) const override;

  Point<Dim1 + Dim2, Frame::Logical> inverse(
      const Point<Dim1 + Dim2, Frame::Grid>& x) const override;

  double jacobian(const Point<Dim1 + Dim2, Frame::Logical>& xi, size_t ud,
                  size_t ld) const override;

  double inv_jacobian(const Point<Dim1 + Dim2, Frame::Logical>& xi, size_t ud,
                      size_t ld) const override;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EmbeddingMap<Dim1 + Dim2, Dim1 + Dim2>), ProductOf2Maps);
  explicit ProductOf2Maps(CkMigrateMessage* /* m */);

  void pup(PUP::er& p) override;  // NOLINT

 private:
  std::unique_ptr<EmbeddingMap<Dim1, Dim1>> map1_;
  std::unique_ptr<EmbeddingMap<Dim2, Dim2>> map2_;
};

/// \ingroup EmbeddingMaps
/// \brief Product of three one-dimensional EmbeddingMaps.
class ProductOf3Maps final : public EmbeddingMap<3, 3> {
 public:
  ProductOf3Maps(const EmbeddingMap<1, 1>& map_1,
                 const EmbeddingMap<1, 1>& map_2,
                 const EmbeddingMap<1, 1>& map_3);
  ProductOf3Maps() = default;
  ~ProductOf3Maps() override = default;
  ProductOf3Maps(const ProductOf3Maps& other);
  ProductOf3Maps& operator=(const ProductOf3Maps&) = delete;
  ProductOf3Maps(ProductOf3Maps&&) noexcept =  // NOLINT
      default;
  ProductOf3Maps& operator=(ProductOf3Maps&&) = default;

  std::unique_ptr<EmbeddingMap<3, 3>> get_clone() const override{
    return std::make_unique<ProductOf3Maps>(*this);
  }

  Point<3, Frame::Grid> operator()(
      const Point<3, Frame::Logical>& xi) const override;

  Point<3, Frame::Logical> inverse(
      const Point<3, Frame::Grid>& x) const override;

  double jacobian(const Point<3, Frame::Logical>& xi, size_t ud,
                  size_t ld) const override;

  double inv_jacobian(const Point<3, Frame::Logical>& xi, size_t ud,
                      size_t ld) const override;

  WRAPPED_PUPable_decl_base_template(  // NOLINT
      SINGLE_ARG(EmbeddingMap<3, 3>), ProductOf3Maps);

  explicit ProductOf3Maps(CkMigrateMessage* /* m */);

  void pup(PUP::er& p) override;

 private:
  std::unique_ptr<EmbeddingMap<1, 1>> map1_{nullptr};
  std::unique_ptr<EmbeddingMap<1, 1>> map2_{nullptr};
  std::unique_ptr<EmbeddingMap<1, 1>> map3_{nullptr};
};
}  // namespace EmbeddingMaps

/// \cond HIDDEN_SYMBOLS
template <size_t Dim1, size_t Dim2>
PUP::able::PUP_ID
    EmbeddingMaps::ProductOf2Maps<Dim1, Dim2>::my_PUP_ID =  // NOLINT
    0;
/// \endcond
