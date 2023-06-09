// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <array>
#include <limits>
#include <pup.h>

#include "DataStructures/CachedTempBuffer.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "DataStructures/Tensor/IndexType.hpp"
#include "DataStructures/Tensor/TypeAliases.hpp"
#include "Options/Context.hpp"
#include "Options/String.hpp"
#include "PointwiseFunctions/AnalyticData/AnalyticData.hpp"
#include "PointwiseFunctions/AnalyticData/GeneralRelativity/AnalyticData.hpp"
#include "PointwiseFunctions/GeneralRelativity/TagsDeclarations.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace Tags {
template <typename Tag>
struct dt;
}  // namespace Tags
namespace gsl {
template <class T>
class not_null;
}  // namespace gsl
/// \endcond

namespace gr::AnalyticData {
/*!
 * \brief Brill Lindquist data \cite Brill1963yv corresponding to two black
 * holes momentarily at rest
 *
 * The spatial metric is given by \f$\gamma_{ij} = \psi^4 \delta_{ij}\f$
 * where the conformal factor is given by
 * \f$\psi = 1 + \frac{m_A}{2 r_A} + \frac{m_B}{2 r_B}\f$ where
 * \f$m_{A,B}\f$ are the masses of the black holes and \f$r_{A,B}\f$ are the
 * positions of a point relative to the center of each black hole
 *
 * The data is time symmetric (\f$K_{ij} = 0\f$) and we arbitrarily choose
 * unit lapse and zero shift.
 */
class BrillLindquist : public AnalyticDataBase<3>, public MarkAsAnalyticData {
 public:
  struct MassA {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole A"};
    static type lower_bound() { return 0.; }
  };
  struct MassB {
    using type = double;
    static constexpr Options::String help = {"Mass of the black hole B"};
    static type lower_bound() { return 0.; }
  };
  struct CenterA {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The [x,y,z] center of the black hole A"};
  };
  struct CenterB {
    using type = std::array<double, 3>;
    static constexpr Options::String help = {
        "The [x,y,z] center of the black hole B"};
  };
  using options = tmpl::list<MassA, MassB, CenterA, CenterB>;
  static constexpr Options::String help{
      "Brill-Lindquist data for two black holes"};

  BrillLindquist(double mass_a, double mass_b,
                 const std::array<double, 3>& center_a,
                 const std::array<double, 3>& center_b,
                 const Options::Context& context = {});
  explicit BrillLindquist(CkMigrateMessage* /*unused*/);

  BrillLindquist() = default;
  BrillLindquist(const BrillLindquist& /*rhs*/) = default;
  BrillLindquist& operator=(const BrillLindquist& /*rhs*/) = default;
  BrillLindquist(BrillLindquist&& /*rhs*/) = default;
  BrillLindquist& operator=(BrillLindquist&& /*rhs*/) = default;
  ~BrillLindquist() = default;

  /*!
   * \brief Computes and returns spacetime quantities for BrillLindquist data at
   * a specific Cartesian position
   *
   * \param x Cartesian coordinates of the position at which to compute
   * spacetime quantities
   */
  template <typename DataType, typename Frame, typename... Tags>
  tuples::TaggedTuple<Tags...> variables(
      const tnsr::I<DataType, volume_dim, Frame>& x,
      tmpl::list<Tags...> /*meta*/) const {
    static_assert(
        tmpl2::flat_all_v<
            tmpl::list_contains_v<tags<DataType, Frame>, Tags>...>,
        "At least one of the requested tags is not supported. The requested "
        "tags are listed as template parameters of the `variables` function.");
    IntermediateVars<DataType, Frame> cache(get_size(*x.begin()));
    IntermediateComputer<DataType, Frame> computer(*this, x);
    return {cache.get_var(computer, Tags{})...};
  }

  // NOLINTNEXTLINE(google-runtime-references)
  void pup(PUP::er& p);

  /*!
   * \brief Return the mass of black hole A
   */
  SPECTRE_ALWAYS_INLINE double mass_a() const { return mass_a_; }
  /*!
   * \brief Return the mass of black hole B
   */
  SPECTRE_ALWAYS_INLINE double mass_b() const { return mass_b_; }
  /*!
   * \brief Return the center of black hole A
   */
  SPECTRE_ALWAYS_INLINE const std::array<double, 3>& center_a() const {
    return center_a_;
  }
  /*!
   * \brief Return the center of black hole B
   */
  SPECTRE_ALWAYS_INLINE const std::array<double, 3>& center_b() const {
    return center_b_;
  }

  /*!
   * \brief Tags defined for intermediates specific to BrillLindquist data
   */
  struct internal_tags {
    /*!
     * \brief Tag for the position of a point relative to the center of
     * black hole A
     *
     * \details Defined as \f$X_A^i = \left(x^i - C_A^i\right)\f$, where
     * \f$C_A^i\f$ is the Cartesian coordinates of the center of black hole A
     * and \f$x^i\f$ is the Cartesian coordinates of the point where we're
     * wanting to compute spacetime quantities.
     */
    template <typename DataType, typename Frame>
    using x_minus_center_a = ::Tags::TempI<0, 3, Frame, DataType>;

    /*!
     * \brief Tag for the radius corresponding to the position of a point
     * relative to the center of black hole A
     *
     * \details Defined as \f$r_A = \sqrt{\delta_{ij} X_A^i X_A^j}\f$, where
     * \f$X_A^i\f$ is defined by `internal_tags::x_minus_center_a`.
     */
    template <typename DataType>
    using r_a = ::Tags::TempScalar<1, DataType>;

    /*!
     * \brief Tag for the position of a point relative to the center of
     * black hole B
     *
     * \details Defined as \f$X_B^i = \left(x^i - C_B^i\right)\f$, where
     * \f$C_B^i\f$ is the Cartesian coordinates of the center of black hole B
     * and \f$x^i\f$ is the Cartesian coordinates of the point where we're
     * wanting to compute spacetime quantities.
     */
    template <typename DataType, typename Frame>
    using x_minus_center_b = ::Tags::TempI<2, 3, Frame, DataType>;

    /*!
     * \brief Tag for the radius corresponding to the position of a point
     * relative to the center of black hole B
     *
     * \details Defined as \f$r_B = \sqrt{\delta_{ij} X_B^i X_B^j}\f$, where
     * \f$X_B^i\f$ is defined by `internal_tags::x_minus_center_b`.
     */
    template <typename DataType>
    using r_b = ::Tags::TempScalar<3, DataType>;
    /*!
     * \brief Tag for the conformal factor
     *
     * \details Defined as \f$\psi = 1 + \frac{m_A}{2 r_A} + \frac{m_B}{2
     * r_B}\f$ where \f$m_{A,B}\f$ are the masses of the black holes and
     * \f$r_{A,B}\f$ are the positions of a point relative to the center of each
     * black hole
     */
    template <typename DataType>
    using conformal_factor = ::Tags::TempScalar<4, DataType>;
    /*!
     * \brief Tag for the deriatives of the conformal factor
     *
     * \details Defined as \f$d_i\psi = -\frac{m_A X_A^j}{2 r_A^3} \delta_{ij}
     *  - \frac{m_B X_B^j}{2 r_B^3} \delta_{ij}\f$ where \f$m_{A,B}\f$ are the
     * masses of the black holes and \f$r_{A,B}\f$ are the positions of a point
     * relative to the center of each black hole.  (Note we are free to
     * raise/lower coordinate indices with a Eucledian metric)
     */
    template <typename DataType, typename Frame>
    using deriv_conformal_factor = ::Tags::Tempi<5, 3, Frame, DataType>;
  };

  template <typename DataType, typename Frame>
  using DerivSpatialMetric =
      ::Tags::deriv<gr::Tags::SpatialMetric<DataType, volume_dim, Frame>,
                    tmpl::size_t<volume_dim>, Frame>;

  /*!
   * \brief Buffer for caching computed intermediates and quantities that we do
   * not want to recompute across the solution's implementation
   *
   * \details See `internal_tags` documentation for details on what quantities
   * the internal tags represent
   */
  template <typename DataType, typename Frame>
  using CachedBuffer =
      CachedTempBuffer<internal_tags::x_minus_center_a<DataType, Frame>,
                       internal_tags::r_a<DataType>,
                       internal_tags::x_minus_center_b<DataType, Frame>,
                       internal_tags::r_b<DataType>,
                       internal_tags::conformal_factor<DataType>,
                       internal_tags::deriv_conformal_factor<DataType, Frame>,
                       gr::Tags::SpatialMetric<DataType, 3, Frame>,
                       DerivSpatialMetric<DataType, Frame>>;

  /*!
   * \brief Computes the intermediates and quantities that we do not want to
   * recompute across the solution's implementation
   */
  template <typename DataType, typename Frame>
  class IntermediateComputer {
   public:
    using CachedBuffer = BrillLindquist::CachedBuffer<DataType, Frame>;

    /*!
     * \brief Constructs a computer for spacetime quantities of a given
     * `gr::AnalyticData::BrillLindquist` solution at at a specific
     * Cartesian position
     *
     * \param analytic_data the given `gr::AnalyticData::BrillLindquist` data
     * \param x Cartesian coordinates of the position at which to compute
     * spacetime quantities
     */
    IntermediateComputer(const BrillLindquist& analytic_data,
                         const tnsr::I<DataType, 3, Frame>& x);

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::x_minus_center_a`
     */
    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center_a,
        gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_minus_center_a<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Computes the radius defined by `internal_tags::r_a`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> r_a,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r_a<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::x_minus_center_b`
     */
    void operator()(
        gsl::not_null<tnsr::I<DataType, 3, Frame>*> x_minus_center_b,
        gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::x_minus_center_b<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Computes the radius defined by `internal_tags::r_b`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> r_b,
                    gsl::not_null<CachedBuffer*> cache,
                    internal_tags::r_b<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::conformal_factor`
     */
    void operator()(gsl::not_null<Scalar<DataType>*> conformal_factor,
                    gsl::not_null<CachedBuffer*> /*cache*/,
                    internal_tags::conformal_factor<DataType> /*meta*/) const;

    /*!
     * \brief Computes the intermediate defined by
     * `internal_tags::deriv_conformal_factor`
     */
    void operator()(
        gsl::not_null<tnsr::i<DataType, 3, Frame>*> deriv_conformal_factor,
        gsl::not_null<CachedBuffer*> /*cache*/,
        internal_tags::deriv_conformal_factor<DataType, Frame> /*meta*/) const;

    /*!
     * \brief Computes the spatial metric
     *
     * \details Computed as \f$\gamma_{ij} = \delta_{ij} \psi^4\f$
     * where \f$\psi\f$ is the conformal factor defined by
     * `internal_tags::conformal_factor`.
     */
    void operator()(gsl::not_null<tnsr::ii<DataType, 3, Frame>*> spatial_metric,
                    gsl::not_null<CachedBuffer*> cache,
                    gr::Tags::SpatialMetric<DataType, 3, Frame> /*meta*/) const;

    /*!
     * \brief Computes the spatial derivative of the spatial metric
     *
     * \details Computed as \f$\partial_k \gamma_{ij} = 4 \psi^3 \partial_k
     * \psi \delta_{ij} \f$ where \f$\psi\f$ is the conformal factor defined by
     * `internal_tags::conformal_factor`.
     */
    void operator()(
        gsl::not_null<tnsr::ijj<DataType, 3, Frame>*> deriv_spatial_metric,
        gsl::not_null<CachedBuffer*> cache,
        DerivSpatialMetric<DataType, Frame> /*meta*/) const;

   private:
    /*!
     * \brief The BrillLindquist data
     */
    const BrillLindquist& analytic_data_;
    /*!
     * \brief Cartesian coordinates of the position at which to compute
     * spacetime quantities
     */
    const tnsr::I<DataType, 3, Frame>& x_;
  };

  /*!
   * \brief Computes and returns spacetime quantities of interest
   */
  template <typename DataType, typename Frame>
  class IntermediateVars : public CachedBuffer<DataType, Frame> {
   public:
    using CachedBuffer = BrillLindquist::CachedBuffer<DataType, Frame>;
    using CachedBuffer::CachedBuffer;
    using CachedBuffer::get_var;

    /*!
     * \brief Returns the lapse, which is 1
     */
    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::Lapse<DataType> /*meta*/);

    /*!
     * \brief Returns the time derivative of the lapse, which is 0
     */
    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Lapse<DataType>> /*meta*/);

    /*!
     * \brief Returns the spatial derivative of the lapse, which is 0
     */
    tnsr::i<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        DerivLapse<DataType, Frame> /*meta*/);

    /*!
     * \brief Returns the shift, which is 0
     */
    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::Shift<DataType, 3, Frame> /*meta*/);

    /*!
     * \brief Returns the time derivative of the shift, which is 0
     */
    tnsr::I<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::Shift<DataType, 3, Frame>> /*meta*/);

    /*!
     * \brief Returns the spatial derivative of the shift, which is 0
     */
    tnsr::iJ<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        DerivShift<DataType, Frame> /*meta*/);

    /*!
     * \brief Returns the time derivative of the spatial metric, which is 0
     */
    tnsr::ii<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        ::Tags::dt<gr::Tags::SpatialMetric<DataType, 3, Frame>> /*meta*/);

    /*!
     * \brief Computes and returns the square root of the determinant of the
     * spatial metric
     */
    Scalar<DataType> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::SqrtDetSpatialMetric<DataType> /*meta*/);

    /*!
     * \brief Computes and returns the inverse spatial metric
     */
    tnsr::II<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::InverseSpatialMetric<DataType, 3, Frame> /*meta*/);

    /*!
     * \brief Computes and returns the extrinsic curvature which is 0
     */
    tnsr::ii<DataType, 3, Frame> get_var(
        const IntermediateComputer<DataType, Frame>& computer,
        gr::Tags::ExtrinsicCurvature<DataType, 3, Frame> /*meta*/);
  };

 private:
  /*!
   * \brief Mass of black hole A
   */
  double mass_a_{std::numeric_limits<double>::signaling_NaN()};
  /*!
   * \brief Mass of black hole B
   */
  double mass_b_{std::numeric_limits<double>::signaling_NaN()};
  /*!
   * \brief Center of black hole A
   */
  std::array<double, 3> center_a_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
  /*!
   * \brief Center of black hole B
   */
  std::array<double, 3> center_b_ =
      make_array<3>(std::numeric_limits<double>::signaling_NaN());
};

/*!
 * \brief Return whether two BrillLindquist data are equivalent
 */
SPECTRE_ALWAYS_INLINE bool operator==(const BrillLindquist& lhs,
                                      const BrillLindquist& rhs);

/*!
 * \brief Return whether two BrillLindquist data are not equivalent
 */
SPECTRE_ALWAYS_INLINE bool operator!=(const BrillLindquist& lhs,
                                      const BrillLindquist& rhs);
}  // namespace gr::AnalyticData
