// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cstddef>

#include "DataStructures/Tensor/Tensor.hpp"
#include "NumericalAlgorithms/LinearOperators/PartialDerivatives.hpp"  // for tags
#include "Options/Options.hpp"
#include "PointwiseFunctions/GeneralRelativity/Tags.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

// IWYU pragma: no_forward_declare Tags::deriv

/// \cond
namespace PUP {
class er;
}  // namespace PUP
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
/// \endcond

namespace gr {
namespace Solutions {

/*!
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
  static constexpr size_t volume_dim = Dim;

  Minkowski() = default;
  Minkowski(const Minkowski& /*rhs*/) noexcept = default;
  Minkowski& operator=(const Minkowski& /*rhs*/) noexcept = default;
  Minkowski(Minkowski&& /*rhs*/) noexcept = default;
  Minkowski& operator=(Minkowski&& /*rhs*/) noexcept = default;
  ~Minkowski() = default;

  template <typename DataType>
  using DerivLapse = ::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                                   Frame::Inertial>;
  template <typename DataType>
  using DerivShift =
      ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial>;
  template <typename DataType>
  using DerivSpatialMetric =
      ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial>;
  template <typename DataType>
  using tags = tmpl::list<
      gr::Tags::Lapse<DataType>, ::Tags::dt<gr::Tags::Lapse<DataType>>,
      DerivLapse<DataType>, gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
      ::Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>,
      DerivShift<DataType>,
      gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
      ::Tags::dt<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>>,
      DerivSpatialMetric<DataType>, gr::Tags::SqrtDetSpatialMetric<DataType>,
      gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataType>,
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>>;
  template <typename DataType, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(const tnsr::I<DataType, Dim>& x,
                                         double t,
                                         tmpl::list<Tags...> /*meta*/) const
      noexcept {
    static_assert(sizeof...(Tags) > 1,
                  "Unrecognized tag requested.  See the function parameters "
                  "for the tag.");
    return {get<Tags>(variables(x, t, tmpl::list<Tags>{}))...};
  }

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::Lapse<DataType>> variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<gr::Tags::Lapse<DataType>> /*meta*/) const noexcept;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::dt<gr::Tags::Lapse<DataType>>> variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<::Tags::dt<gr::Tags::Lapse<DataType>>> /*meta*/) const
      noexcept;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::deriv<gr::Tags::Lapse<DataType>,
                                    tmpl::size_t<Dim>, Frame::Inertial>>
  variables(
      const tnsr::I<DataType, Dim>& x, double t,
      tmpl::list<::Tags::deriv<gr::Tags::Lapse<DataType>, tmpl::size_t<Dim>,
                               Frame::Inertial>> /*meta*/) const noexcept;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::Shift<Dim, Frame::Inertial, DataType>> /*meta*/)
      const noexcept;

  template <typename DataType>
  tuples::TaggedTuple<
      ::Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<
          ::Tags::dt<gr::Tags::Shift<Dim, Frame::Inertial, DataType>>> /*meta*/)
      const noexcept;

  template <typename DataType>
  tuples::TaggedTuple<
      ::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<::Tags::deriv<gr::Tags::Shift<Dim, Frame::Inertial, DataType>,
                               tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/)
      const noexcept;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>>
  variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
            tmpl::list<gr::Tags::SpatialMetric<Dim, Frame::Inertial,
                                               DataType>> /*meta*/) const
      noexcept;

  template <typename DataType>
  tuples::TaggedTuple<
      ::Tags::dt<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>>>
  variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
            tmpl::list<::Tags::dt<gr::Tags::SpatialMetric<
                Dim, Frame::Inertial, DataType>>> /*meta*/) const noexcept;

  template <typename DataType>
  tuples::TaggedTuple<
      ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                    tmpl::size_t<Dim>, Frame::Inertial>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<
          ::Tags::deriv<gr::Tags::SpatialMetric<Dim, Frame::Inertial, DataType>,
                        tmpl::size_t<Dim>, Frame::Inertial>> /*meta*/) const
      noexcept;

  template <typename DataType>
  tuples::TaggedTuple<
      gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial, DataType>>
  variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
            tmpl::list<gr::Tags::InverseSpatialMetric<Dim, Frame::Inertial,
                                                      DataType>> /*meta*/) const
      noexcept;

  template <typename DataType>
  tuples::TaggedTuple<
      gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial, DataType>>
  variables(const tnsr::I<DataType, Dim>& x, double /*t*/,
            tmpl::list<gr::Tags::ExtrinsicCurvature<Dim, Frame::Inertial,
                                                    DataType>> /*meta*/) const
      noexcept;

  template <typename DataType>
  tuples::TaggedTuple<gr::Tags::SqrtDetSpatialMetric<DataType>> variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<gr::Tags::SqrtDetSpatialMetric<DataType>> /*meta*/) const
      noexcept;

  template <typename DataType>
  tuples::TaggedTuple<::Tags::dt<gr::Tags::SqrtDetSpatialMetric<DataType>>>
  variables(
      const tnsr::I<DataType, Dim>& x, double /*t*/,
      tmpl::list<::Tags::dt<gr::Tags::SqrtDetSpatialMetric<DataType>>> /*meta*/)
      const noexcept;

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
}  // namespace Solutions
}  // namespace gr
