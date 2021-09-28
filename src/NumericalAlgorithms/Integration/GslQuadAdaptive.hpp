// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <gsl/gsl_integration.h>
#include <limits>
#include <memory>
#include <vector>

/// \cond
namespace PUP {
class er;
}  // namespace PUP
/// \endcond

/// \ingroup NumericalAlgorithmsGroup
/// Numerical integration algorithms
namespace integration {

namespace detail {

// The GSL functions require the integrand to have this particular function
// signature. In particular, any extra parameters to the functions must be
// passed as a void* and re-interpreted appropriately.
template <typename IntegrandType>
double integrand(const double x, void* const params) {
  // NOLINTNEXTLINE(cppcoreguidelines-pro-type-reinterpret-cast)
  auto* const function = reinterpret_cast<IntegrandType*>(params);
  return (*function)(x);
}

class GslQuadAdaptiveImpl {
 public:
  explicit GslQuadAdaptiveImpl(size_t max_intervals);

  GslQuadAdaptiveImpl() = default;
  GslQuadAdaptiveImpl(const GslQuadAdaptiveImpl&) = delete;
  GslQuadAdaptiveImpl& operator=(const GslQuadAdaptiveImpl&) = delete;
  GslQuadAdaptiveImpl(GslQuadAdaptiveImpl&&) = default;
  GslQuadAdaptiveImpl& operator=(GslQuadAdaptiveImpl&& rhs) = default;
  ~GslQuadAdaptiveImpl() = default;

  void pup(PUP::er& p);  // NOLINT(google-runtime-references)

 protected:
  template <typename IntegrandType>
  gsl_function* gsl_integrand(IntegrandType&& integrand) const {
    gsl_integrand_.function = &detail::integrand<IntegrandType>;
    gsl_integrand_.params = &integrand;
    return &gsl_integrand_;
  }

  struct GslIntegrationWorkspaceDeleter {
    void operator()(gsl_integration_workspace* workspace) const;
  };

  size_t max_intervals_ = 0;
  std::unique_ptr<gsl_integration_workspace, GslIntegrationWorkspaceDeleter>
      workspace_{};

 private:
  void initialize();

  mutable gsl_function gsl_integrand_{};
};

void check_status_code(int status_code);

void disable_gsl_error_handling();

}  // namespace detail

/// Each type specifies which algorithm from GSL should be used. It should be
/// chosen according to the problem.
enum class GslIntegralType {
  /// gsl_integration_qag()
  StandardGaussKronrod,
  /// gsl_integration_qags()
  IntegrableSingularitiesPresent,
  /// gsl_integration_qagp()
  IntegrableSingularitiesKnown,
  /// gsl_integration_qagi()
  InfiniteInterval,
  /// gsl_integration_qagiu()
  UpperBoundaryInfinite,
  /// gsl_integration_qagil()
  LowerBoundaryInfinite
};

/*!
 * \brief A wrapper around the GSL adaptive integration procedures
 *
 * All templates take upper bounds to the absolute and relative error;
 * `tolerance_abs` and `tolerance_rel(default = 0.0)`, respectively. To compute
 * to a specified absolute error, set `tolerance_rel` to zero. To compute to a
 * specified relative error, set `tolerance_abs` to zero. For details on the
 * algorithm see the GSL documentation on `gsl_integration`.
 *
 * Here is an example how to use this class. For the function:
 *
 * \snippet Test_GslQuadAdaptive.cpp integrated_function
 *
 * the integration should look like:
 *
 * \snippet Test_GslQuadAdaptive.cpp integration_example
 */
template <GslIntegralType TheIntegralType>
class GslQuadAdaptive;

/*!
 * \brief Use Gauss-Kronrod rule to integrate a 1D-function
 *
 * The algorithm for "StandardGaussKronrod" uses the QAG algorithm to employ an
 * adaptive Gauss-Kronrod n-points integration rule. Its function takes a
 * `lower_boundary` and `upper_boundary` to the integration region, an upper
 * bound for the absolute error `tolerance_abs` and a `key`. The latter is an
 * index to the array [15, 21, 31, 41, 51, 61], where each element denotes how
 * many function evaluations take place in each subinterval. A higher-order rule
 * serves better for smooth functions, whereas a lower-order rule saves time for
 * functions with local difficulties, such as discontinuities.
 */
template <>
class GslQuadAdaptive<GslIntegralType::StandardGaussKronrod>
    : public detail::GslQuadAdaptiveImpl {
 public:
  using detail::GslQuadAdaptiveImpl::GslQuadAdaptiveImpl;
  template <typename IntegrandType>
  double operator()(IntegrandType&& integrand, const double lower_boundary,
                    const double upper_boundary, const double tolerance_abs,
                    const int key, const double tolerance_rel = 0.) const {
    double result = std::numeric_limits<double>::signaling_NaN();
    double error = std::numeric_limits<double>::signaling_NaN();
    detail::disable_gsl_error_handling();
    const int status_code = gsl_integration_qag(
        gsl_integrand(std::forward<IntegrandType>(integrand)), lower_boundary,
        upper_boundary, tolerance_abs, tolerance_rel, this->max_intervals_, key,
        this->workspace_.get(), &result, &error);
    detail::check_status_code(status_code);
    return result;
  }
};

/*!
 * \brief Integrates a 1D-function with singularities
 *
 * The algorithm for "IntegrableSingularitiesPresent" concentrates new,
 * increasingly smaller subintervals around an unknown singularity and makes
 * successive approximations to the integral which should converge towards a
 * limit. The integration region is defined by `lower_boundary` and
 * `upper_boundary`.
 */
template <>
class GslQuadAdaptive<GslIntegralType::IntegrableSingularitiesPresent>
    : public detail::GslQuadAdaptiveImpl {
 public:
  using detail::GslQuadAdaptiveImpl::GslQuadAdaptiveImpl;
  template <typename IntegrandType>
  double operator()(IntegrandType&& integrand, const double lower_boundary,
                    const double upper_boundary, const double tolerance_abs,
                    const double tolerance_rel = 0.) const {
    double result = std::numeric_limits<double>::signaling_NaN();
    double error = std::numeric_limits<double>::signaling_NaN();
    detail::disable_gsl_error_handling();
    const int status_code = gsl_integration_qags(
        gsl_integrand(std::forward<IntegrandType>(integrand)), lower_boundary,
        upper_boundary, tolerance_abs, tolerance_rel, this->max_intervals_,
        this->workspace_.get(), &result, &error);
    detail::check_status_code(status_code);
    return result;
  }
};

/*!
 * \brief Integrates a 1D-function where singularities are known
 *
 * The algorithm for "IntegrableSingularitiesKnown" uses user-defined
 * subintervals given by a vector of doubles `points`, where each element
 * denotes an interval boundary.
 */
template <>
class GslQuadAdaptive<GslIntegralType::IntegrableSingularitiesKnown>
    : public detail::GslQuadAdaptiveImpl {
 public:
  using detail::GslQuadAdaptiveImpl::GslQuadAdaptiveImpl;
  template <typename IntegrandType>
  double operator()(IntegrandType&& integrand,
                    const std::vector<double>& points,
                    const double tolerance_abs,
                    const double tolerance_rel = 0.) const {
    double result = std::numeric_limits<double>::signaling_NaN();
    double error = std::numeric_limits<double>::signaling_NaN();
    detail::disable_gsl_error_handling();
    // The const_cast is necessary because GSL has a weird interface where
    // the number of singularities does not change, but it doesn't take
    // the argument as a `const double*`. However, the first thing
    // `gsl_integration_qagp` does internally is forward to another function
    // that does take `points` by `const double*`. If `gsl_integration_qagp`
    // were to change the size of `points` this code would be severely broken
    // because `std::vector` allocates with `new`, while GSL would likely use
    // `malloc` (or some other C allocator). Mixing (de)allocators in such a way
    // is undefined behavior.
    const int status_code = gsl_integration_qagp(
        gsl_integrand(std::forward<IntegrandType>(integrand)),
        const_cast<double*>(points.data()),  // NOLINT
        points.size(), tolerance_abs, tolerance_rel, this->max_intervals_,
        this->workspace_.get(), &result, &error);
    detail::check_status_code(status_code);
    return result;
  }
};

/*!
 * \brief Integrates a 1D-function over the interval \f$ (-\infty, +\infty) \f$
 *
 * The algorithm for "InfiniteInterval" uses the semi-open interval
 * \f$ (0, 1] \f$ to map to an infinite interval \f$ (-\infty, +\infty) \f$. Its
 * function takes no parameters other than limits `tolerance_abs` and optional
 * `tolerance_rel` for the absolute error and the relative error.
 */
template <>
class GslQuadAdaptive<GslIntegralType::InfiniteInterval>
    : public detail::GslQuadAdaptiveImpl {
 public:
  using detail::GslQuadAdaptiveImpl::GslQuadAdaptiveImpl;
  template <typename IntegrandType>
  double operator()(IntegrandType&& integrand, const double tolerance_abs,
                    const double tolerance_rel = 0.) const {
    double result = std::numeric_limits<double>::signaling_NaN();
    double error = std::numeric_limits<double>::signaling_NaN();
    detail::disable_gsl_error_handling();
    const int status_code = gsl_integration_qagi(
        gsl_integrand(std::forward<IntegrandType>(integrand)), tolerance_abs,
        tolerance_rel, this->max_intervals_, this->workspace_.get(), &result,
        &error);
    detail::check_status_code(status_code);
    return result;
  }
};

/*!
 * \brief Integrates a 1D-function over the interval \f$ [a, +\infty) \f$
 *
 * The algorithm for "UpperBoundaryInfinite" maps the semi-open interval
 * \f$ (0, 1] \f$ to a semi-infinite interval \f$ [a, +\infty) \f$, where
 * \f$ a \f$ is given by `lower_boundary`.
 */
template <>
class GslQuadAdaptive<GslIntegralType::UpperBoundaryInfinite>
    : public detail::GslQuadAdaptiveImpl {
 public:
  using detail::GslQuadAdaptiveImpl::GslQuadAdaptiveImpl;
  template <typename IntegrandType>
  double operator()(IntegrandType&& integrand, const double lower_boundary,
                    const double tolerance_abs,
                    const double tolerance_rel = 0.) const {
    double result = std::numeric_limits<double>::signaling_NaN();
    double error = std::numeric_limits<double>::signaling_NaN();
    detail::disable_gsl_error_handling();
    const int status_code = gsl_integration_qagiu(
        gsl_integrand(std::forward<IntegrandType>(integrand)), lower_boundary,
        tolerance_abs, tolerance_rel, this->max_intervals_,
        this->workspace_.get(), &result, &error);
    detail::check_status_code(status_code);
    return result;
  }
};

/*!
 * \brief Integrates a 1D-function over the interval \f$ (-\infty, b] \f$
 *
 * The algorithm for "LowerBoundaryInfinite" maps the semi-open interval
 * \f$ (0, 1] \f$ to a semi-infinite interval \f$ (-\infty, b] \f$, where
 * \f$ b \f$ is given by `upper_boundary`.
 */
template <>
class GslQuadAdaptive<GslIntegralType::LowerBoundaryInfinite>
    : public detail::GslQuadAdaptiveImpl {
 public:
  using detail::GslQuadAdaptiveImpl::GslQuadAdaptiveImpl;
  template <typename IntegrandType>
  double operator()(IntegrandType&& integrand, const double upper_boundary,
                    const double tolerance_abs,
                    const double tolerance_rel = 0.) const {
    double result = std::numeric_limits<double>::signaling_NaN();
    double error = std::numeric_limits<double>::signaling_NaN();
    detail::disable_gsl_error_handling();
    const int status_code = gsl_integration_qagil(
        gsl_integrand(std::forward<IntegrandType>(integrand)), upper_boundary,
        tolerance_abs, tolerance_rel, this->max_intervals_,
        this->workspace_.get(), &result, &error);
    detail::check_status_code(status_code);
    return result;
  }
};
}  // namespace integration
