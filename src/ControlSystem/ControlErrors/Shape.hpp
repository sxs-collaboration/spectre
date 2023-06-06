// Distributed under the MIT License.
// See LICENSE.txt for details.

#pragma once

#include <cmath>
#include <cstddef>
#include <pup.h>
#include <string>

#include "ControlSystem/Protocols/ControlError.hpp"
#include "ControlSystem/Tags/QueueTags.hpp"
#include "ControlSystem/Tags/SystemTags.hpp"
#include "DataStructures/DataVector.hpp"
#include "Domain/Structure/ObjectLabel.hpp"
#include "NumericalAlgorithms/SphericalHarmonics/SpherepackIterator.hpp"
#include "Options/String.hpp"
#include "Parallel/GlobalCache.hpp"
#include "Utilities/EqualWithinRoundoff.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ForceInline.hpp"
#include "Utilities/ProtocolHelpers.hpp"
#include "Utilities/TMPL.hpp"
#include "Utilities/TaggedTuple.hpp"

/// \cond
namespace domain::Tags {
template <size_t Dim>
struct Domain;
struct FunctionsOfTime;
}  // namespace domain::Tags
namespace Frame {
struct Distorted;
}  // namespace Frame
/// \endcond

namespace control_system {
namespace ControlErrors {
namespace detail {
template <::domain::ObjectLabel Horizon>
std::string excision_sphere_name() {
  return "ExcisionSphere"s + ::domain::name(Horizon);
}

template <::domain::ObjectLabel Horizon>
std::string size_name() {
  return "Size"s + ::domain::name(Horizon);
}
}  // namespace detail

/*!
 * \brief Control error in the
 * \link domain::CoordinateMaps::TimeDependent::Shape Shape \endlink coordinate
 * map
 *
 * \details Computes the error for each \f$ l,m \f$ mode of the shape map except
 * for \f$ l=0 \f$ and \f$ l=1 \f$. This is because the \f$ l=0 \f$ mode is
 * controlled by characteristic speed control, and the \f$ l=1 \f$ mode is
 * redundant with the \link
 * domain::CoordinateMaps::TimeDependent::Translation Translation \endlink map.
 * The equation for the control error is Eq. (77) from \cite Hemberger2012jz
 * which is
 *
 * \f{align}
 * Q_{lm} &= -\frac{r_\mathrm{EB} -
 * Y_{00}\lambda_{00}(t)}{\sqrt{\frac{\pi}{2}} Y_{00}S_{00}} S_{lm} -
 * \lambda_{lm}(t), \quad l>=2 \label{eq:shape_control_error}
 * \f}
 *
 * where \f$ r_\mathrm{EB} \f$ is the radius of the excision boundary in the
 * grid frame, \f$ \lambda_{00}(t) \f$ is the size map parameter, \f$
 * \lambda_{lm}(t) \f$ for \f$ l>=2 \f$ are the shape map parameters, and \f$
 * S_{lm}\f$ are the coefficients of the harmonic expansion of the apparent
 * horizon. The coefficients \f$ \lambda_{lm}(t) \f$ (*not* including \f$ l=0
 * \f$) and \f$ S_{lm}\f$ (including \f$ l=0 \f$) are stored as the real-valued
 * coefficients \f$ a_{lm} \f$ and \f$ b_{lm} \f$ of the SPHEREPACK
 * spherical-harmonic expansion (in ylm::Spherepack). The $\lambda_{00}(t)$
 * coefficient, on the other hand, is stored as the complex coefficient \f$
 * A_{00} \f$ of the standard \f$ Y_{lm} \f$ decomposition. Because \f$ a_{00} =
 * \sqrt{\frac{2}{\pi}}A_{00} \f$, there is an extra factor of \f$
 * \sqrt{\frac{\pi}{2}} \f$ in the above formula in the denominator of the
 * fraction multiplying the \f$ S_{00}\f$ component so it is represented in the
 * \f$ Y_{lm} \f$ decomposition just like \f$ r_{EB} \f$ and \f$ \lambda_{00}
 * \f$ are). That way, we ensure the numerator and denominator are represented
 * in the same way before we take their ratio.
 *
 * Requirements:
 * - This control error requires that there be at least one excision surface in
 *   the simulation
 * - Currently this control error can only be used with the \link
 *   control_system::Systems::Shape Shape \endlink control system
 */
template <::domain::ObjectLabel Horizon>
struct Shape : tt::ConformsTo<protocols::ControlError> {
  static constexpr size_t expected_number_of_excisions = 1;

  // Shape needs an excision sphere
  using object_centers = domain::object_list<Horizon>;

  using options = tmpl::list<>;
  static constexpr Options::String help{
      "Computes the control error for shape control. This should not take any "
      "options."};

  void pup(PUP::er& /*p*/) {}

  template <typename Metavariables, typename... TupleTags>
  DataVector operator()(const Parallel::GlobalCache<Metavariables>& cache,
                        const double time,
                        const std::string& function_of_time_name,
                        const tuples::TaggedTuple<TupleTags...>& measurements) {
    const auto& domain = get<domain::Tags::Domain<3>>(cache);
    const auto& functions_of_time = get<domain::Tags::FunctionsOfTime>(cache);
    const DataVector lambda_lm_coefs =
        functions_of_time.at(function_of_time_name)->func(time)[0];
    const double lambda_00_coef =
        functions_of_time.at(detail::size_name<Horizon>())->func(time)[0][0];

    const auto& ah =
        get<control_system::QueueTags::Horizon<Frame::Distorted>>(measurements);
    const auto& ah_coefs = ah.coefficients();

    ASSERT(lambda_lm_coefs.size() == ah_coefs.size(),
           "Number of coefficients for shape map '"
               << function_of_time_name << "' (" << lambda_lm_coefs.size()
               << ") does not match the number of coefficients in the AH "
                  "Strahlkorper ("
               << ah_coefs.size() << ").");

    const auto& excision_spheres = domain.excision_spheres();

    ASSERT(domain.excision_spheres().count(
               detail::excision_sphere_name<Horizon>()) == 1,
           "Excision sphere " << detail::excision_sphere_name<Horizon>()
                              << " not in the domain but is needed to "
                                 "compute Shape control error.");

    const double radius_excision_sphere_grid_frame =
        excision_spheres.at(detail::excision_sphere_name<Horizon>()).radius();

    const double Y00 = sqrt(0.25 / M_PI);
    SpherepackIterator iter{ah.l_max(), ah.m_max()};
    // See above docs for why we have the sqrt(pi/2) in the denominator
    const double relative_size_factor =
        (radius_excision_sphere_grid_frame / Y00 - lambda_00_coef) /
        (sqrt(0.5 * M_PI) * ah_coefs[iter.set(0, 0)()]);

    // The map parameters are in terms of SPHEREPACK coefficients (just like
    // strahlkorper coefficients), *not* spherical harmonic coefficients, thus
    // the control error for each l,m is in terms of SPHEREPACK coefficients
    // and no extra factors of sqrt(2/pi) are needed
    DataVector Q = -relative_size_factor * ah_coefs - lambda_lm_coefs;

    // Shape control is only for l > 1 so enforce that Q=0 for l=0,l=1. These
    // components of the control error won't be 0 automatically because the AH
    // can freely have nonzero l=0 and l=1 coefficients so we have to set the
    // control error components to be 0 manually.
    Q[iter.set(0, 0)()] = 0.0;
    Q[iter.set(1, -1)()] = 0.0;
    Q[iter.set(1, 0)()] = 0.0;
    Q[iter.set(1, 1)()] = 0.0;

    return Q;
  }
};
}  // namespace ControlErrors
}  // namespace control_system
