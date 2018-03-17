// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // for tags
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/GrTagsDeclarations.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace EinsteinSolutions {

/*!
 * \ingroup EinsteinSolutionsGroup
 * \brief The Minkowski solution for flat space in Dim spatial dimensions.
 *
 * \details The solution has lapse \f$N(x,t)= 1 \f$, shift \f$N^i(x,t) = 0 \f$
 * and the identity as the spatial metric: \f$g_{ii} = 1 \f$
 */
template <size_t Dim>
class Minkowski {
 public:
  using options = tmpl::list<>;
  static constexpr OptionString help{
      "Minkowski solution to Einstein's Equations"};

  Minkowski() = default;
  Minkowski(const Minkowski& /*rhs*/) = delete;
  Minkowski& operator=(const Minkowski& /*rhs*/) = delete;
  Minkowski(Minkowski&& /*rhs*/) noexcept = default;
  Minkowski& operator=(Minkowski&& /*rhs*/) noexcept = default;
  ~Minkowski() = default;

  template <typename DataType>
  using DerivLapse =
      Tags::deriv<gr::Tags::Lapse<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivShift =
      Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivSpatialMetric =
      Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                  tmpl::size_t<Dim>, Frame::Inertial>;
  template <typename DataType>
  using tags = tmpl::list<
      gr::Tags::Lapse<Dim, Frame::Inertial, DataType>,
      gr::Tags::DtLapse<Dim, Frame::Inertial, DataType>, DerivLapse<DataType>,
      gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
      gr::Tags::DtShift<Dim, Frame::Inertial, DataType>, DerivShift<DataType>,
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
      gr::Tags::DtSpatialMetric<Dim, Frame::Inertial, DataType>,
      DerivSpatialMetric<DataType>>;

  template <typename DataType>
  tuples::TaggedTupleTypelist<tags<DataType>> variables(
      const tnsr::I<DataType, Dim>& x, double t, tags<DataType> /*meta*/) const
      noexcept;

  template <typename T>
  Scalar<T> lapse(const tnsr::I<T, Dim>& x, double t) const noexcept;

  template <typename T>
  Scalar<T> dt_lapse(const tnsr::I<T, Dim>& x, double t) const noexcept;

  template <typename T>
  tnsr::i<T, Dim> deriv_lapse(const tnsr::I<T, Dim>& x, double t) const
      noexcept;

  template <typename T>
  Scalar<T> sqrt_determinant_of_spatial_metric(const tnsr::I<T, Dim>& x,
                                               double t) const noexcept;

  template <typename T>
  Scalar<T> dt_sqrt_determinant_of_spatial_metric(const tnsr::I<T, Dim>& x,
                                                  double t) const noexcept;

  template <typename T>
  tnsr::I<T, Dim> shift(const tnsr::I<T, Dim>& x, double t) const noexcept;

  template <typename T>
  tnsr::I<T, Dim> dt_shift(const tnsr::I<T, Dim>& x, double t) const noexcept;

  template <typename T>
  tnsr::iJ<T, Dim> deriv_shift(const tnsr::I<T, Dim>& x, double t) const
      noexcept;

  template <typename T>
  tnsr::ii<T, Dim> spatial_metric(const tnsr::I<T, Dim>& x, double t) const
      noexcept;

  template <typename T>
  tnsr::ii<T, Dim> dt_spatial_metric(const tnsr::I<T, Dim>& x, double t) const
      noexcept;

  template <typename T>
  tnsr::ijj<T, Dim> deriv_spatial_metric(const tnsr::I<T, Dim>& x,
                                         double t) const noexcept;

  template <typename T>
  tnsr::II<T, Dim> inverse_spatial_metric(const tnsr::I<T, Dim>& x,
                                          double t) const noexcept;

  template <typename T>
  tnsr::ii<T, Dim> extrinsic_curvature(const tnsr::I<T, Dim>& x, double t) const
      noexcept;

  // clang-tidy: google-runtime-references
  void pup(PUP::er& /*p*/) noexcept {}  // NOLINT
};

template <size_t Dim>
inline constexpr bool operator==(const Minkowski<Dim>& /*lhs*/,
                                 const Minkowski<Dim>& /*rhs*/) noexcept {
  return true;
}

template <size_t Dim>
inline constexpr bool operator!=(const Minkowski<Dim>& /*lhs*/,
                                 const Minkowski<Dim>& /*rhs*/) noexcept {
  return false;
}
}  // namespace EinsteinSolutions
