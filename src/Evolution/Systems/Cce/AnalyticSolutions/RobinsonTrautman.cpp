// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/Cce/AnalyticSolutions/RobinsonTrautman.hpp"

#include <algorithm>
#include <complex>
#include <cstddef>
#include <memory>
#include <type_traits>
#include <utility>
#include <vector>

#include "DataStructures/ComplexDataVector.hpp"
#include "DataStructures/SpinWeighted.hpp"
#include "DataStructures/Tags/TempTensor.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/SphericalMetricData.hpp"
#include "Evolution/Systems/Cce/AnalyticSolutions/WorldtubeData.hpp"
#include "Evolution/Systems/Cce/Tags.hpp"
#include "NumericalAlgorithms/OdeIntegration/OdeIntegration.hpp"
#include "NumericalAlgorithms/Spectral/SwshCoefficients.hpp"
#include "NumericalAlgorithms/Spectral/SwshCollocation.hpp"
#include "NumericalAlgorithms/Spectral/SwshDerivatives.hpp"
#include "NumericalAlgorithms/Spectral/SwshFiltering.hpp"
#include "NumericalAlgorithms/Spectral/SwshTransform.hpp"
#include "Options/Options.hpp"
#include "Utilities/ErrorHandling/Assert.hpp"
#include "Utilities/ErrorHandling/Error.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/TMPL.hpp"

namespace Cce::Solutions {

RobinsonTrautman::RobinsonTrautman(
    std::vector<std::complex<double>> initial_modes,
    const double extraction_radius, const size_t l_max, const double tolerance,
    const double start_time, const Options::Context& context)
    : SphericalMetricData{extraction_radius},
      dense_output_rt_scalar_{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)},
      dense_output_du_rt_scalar_{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max)},
      l_max_{l_max},
      tolerance_{tolerance},
      start_time_{start_time},
      initial_modes_{std::move(initial_modes)} {
  if (initial_modes_.size() > square(l_max + 1)) {
    PARSE_ERROR(context,
                "There must not be more than (l_max + 1)^2 modes specified for "
                "InitialModes");
  }
  if (tolerance == 0.0) {
    PARSE_ERROR(context,
                "A target tolerance of 0.0 is not permitted, as it will cause "
                "the time-stepper to enter an infinite loop.");
  }
  initialize_stepper_from_start();
}

void RobinsonTrautman::initialize_stepper_from_start() noexcept {
  // create the initial data
  SpinWeighted<ComplexModalVector, 0> goldberg_modes{square(l_max_ + 1), 0.0};
  for (size_t i = 0; i < std::min(initial_modes_.size(), goldberg_modes.size());
       ++i) {
    goldberg_modes.data()[i] = gsl::at(initial_modes_, i);
  }

  SpinWeighted<ComplexModalVector, 0> libsharp_modes{
      Spectral::Swsh::size_of_libsharp_coefficient_vector(l_max_)};
  const auto& coefficients_metadata =
      Spectral::Swsh::cached_coefficients_metadata(l_max_);
  for (const auto& coefficient : coefficients_metadata) {
    Spectral::Swsh::goldberg_modes_to_libsharp_modes_single_pair(
        coefficient, make_not_null(&libsharp_modes), 0,
        goldberg_modes.data()[Spectral::Swsh::goldberg_mode_index(
            l_max_, coefficient.l, static_cast<int>(coefficient.m), 0)],
        goldberg_modes.data()[Spectral::Swsh::goldberg_mode_index(
            l_max_, coefficient.l, -static_cast<int>(coefficient.m), 0)]);
  }

  {
    // re-use the `dense_output_rt_scalar_` buffers to reduce allocations
    auto& initial_rt_scalar = get(dense_output_rt_scalar_);
    auto& initial_du_rt_scalar = get(dense_output_du_rt_scalar_);

    Spectral::Swsh::inverse_swsh_transform(
        l_max_, 1, make_not_null(&initial_rt_scalar), libsharp_modes);
    // Add the modes specified in the input file to the leading-order solution
    // of 1.0 for the Robinson-Trautman scalar.
    initial_rt_scalar.data() =
        1.0 + std::complex<double>(1.0, 0.0) * real(initial_rt_scalar.data());
    du_rt_scalar(make_not_null(&initial_du_rt_scalar), initial_rt_scalar);
    stepper_ = boost::numeric::odeint::make_dense_output(
        tolerance_ * 0.01, tolerance_,
        boost::numeric::odeint::runge_kutta_dopri5<ComplexDataVector>{});
    const double rt_scalar_scale = max(abs(initial_rt_scalar.data()));
    const double du_rt_scalar_scale = max(abs(initial_du_rt_scalar.data()));

    // Take as an initial step guess a conservative factor multiplied by the
    // rough scaling of the grid spacing. Typically, the initial_step should be
    // set by the guess based on the scale of the fields inside the `if`, but
    // this guess is typically acceptable when the field scale is not usable for
    // step estimation.
    double initial_step = extraction_radius_ * 1.0e-4 / square(l_max_);
    if (rt_scalar_scale > 1.0e-5 and du_rt_scalar_scale > 1.0e-5) {
      // set initial step size according to the first couple of steps in section
      // II.4 of Solving Ordinary Differential equations by Hairer, Norsett, and
      // Wanner
      initial_step =
          0.1 * rt_scalar_scale / (du_rt_scalar_scale * square(l_max_));
    }
    stepper_.initialize(initial_rt_scalar.data(), start_time_, initial_step);
    // scoped to destruct buffer reference
  }
  const auto rt_system = [this](const ComplexDataVector& local_rt_scalar,
                                ComplexDataVector& local_du_rt_scalar,
                                const double /*t*/) noexcept {
    local_du_rt_scalar.destructive_resize(local_rt_scalar.size());
    const SpinWeighted<ComplexDataVector, 0> rt_scalar_reference;
    make_const_view(make_not_null(&rt_scalar_reference.data()), local_rt_scalar,
                    0, local_rt_scalar.size());
    SpinWeighted<ComplexDataVector, 0> du_rt_scalar_reference;
    du_rt_scalar_reference.set_data_ref(local_du_rt_scalar.data(),
                                        local_du_rt_scalar.size());
    du_rt_scalar(make_not_null(&du_rt_scalar_reference), rt_scalar_reference);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&du_rt_scalar_reference), l_max_, l_max_ - 3);
  };
  step_range_ = stepper_.do_step(rt_system);
}

std::unique_ptr<WorldtubeData> RobinsonTrautman::get_clone() const noexcept {
  return std::make_unique<RobinsonTrautman>(*this);
}

void RobinsonTrautman::prepare_solution(const size_t l_max,
                                        const double time) const noexcept {
  ASSERT(l_max == l_max_,
         "The Robinson-Trautman solution only supports the l_max resolution "
         "specified at construction, as it must internally store the evolved "
         "scalar at the desired resolution");
  if (step_range_.first > time) {
    ERROR(
        "The Robinson-Trautman solution does not support stepping backwards in "
        "time during the internal evolution. This error likely means that the "
        "corresponding evolution test should be using a multistep time-stepper "
        "so that the requested boundary data is monotonic.");
  }
  // step until the target time is within the current timestep
  const auto rt_system = [this](const ComplexDataVector& local_rt_scalar,
                                ComplexDataVector& local_du_rt_scalar,
                                const double /*t*/) noexcept {
    local_du_rt_scalar.destructive_resize(local_rt_scalar.size());
    const SpinWeighted<ComplexDataVector, 0> rt_scalar_reference;
    make_const_view(make_not_null(&rt_scalar_reference.data()), local_rt_scalar,
                    0, local_rt_scalar.size());
    SpinWeighted<ComplexDataVector, 0> du_rt_scalar_reference;
    du_rt_scalar_reference.set_data_ref(local_du_rt_scalar.data(),
                                        local_du_rt_scalar.size());
    du_rt_scalar(make_not_null(&du_rt_scalar_reference), rt_scalar_reference);
    Spectral::Swsh::filter_swsh_boundary_quantity(
        make_not_null(&du_rt_scalar_reference), l_max_, l_max_ - 3);
  };
  while (step_range_.second < time) {
    step_range_ = stepper_.do_step(rt_system);
  }
  stepper_.calc_state(time, get(dense_output_rt_scalar_).data());
  du_rt_scalar(make_not_null(&get(dense_output_du_rt_scalar_)),
               get(dense_output_rt_scalar_));
  Spectral::Swsh::filter_swsh_boundary_quantity(
      make_not_null(&get(dense_output_du_rt_scalar_)), l_max_, l_max_ - 3);
  prepared_time_ = time;
}

void RobinsonTrautman::variables_impl(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, -2>>*> news,
    const size_t l_max, const double time,
    tmpl::type_<Tags::News> /*meta*/) const noexcept {
  ASSERT(time == prepared_time_,
         "The Robinson-Trautman solution is being calculated in an "
         "inconsistent state. The public interface should always call "
         "`prepare_solution` before calculating any metric components, so this "
         "may indicate inconsistent use of the internal Robinson-Trautman "
         "calculation functions.");
  ASSERT(l_max == l_max_,
         "The Robinson-Trautman solution only supports the l_max resolution "
         "specified at construction, as it must internally store the evolved "
         "scalar at the desired resolution");
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::EthbarEthbar>>(
      l_max_, 1, make_not_null(&get(*news)), get(dense_output_rt_scalar_));
  get(*news).data() /= get(dense_output_rt_scalar_).data();
}

void RobinsonTrautman::du_rt_scalar(
    const gsl::not_null<SpinWeighted<ComplexDataVector, 0>*> local_du_rt_scalar,
    const SpinWeighted<ComplexDataVector, 0>& rt_scalar) const noexcept {
  using rt_tag = ::Tags::SpinWeighted<::Tags::TempScalar<0, ComplexDataVector>,
                                      std::integral_constant<int, 0>>;
  using ethbar_ethbar_rt_tag =
      Spectral::Swsh::Tags::Derivative<rt_tag,
                                       Spectral::Swsh::Tags::EthbarEthbar>;
  using eth_eth_ethbar_ethbar_rt_tag =
      Spectral::Swsh::Tags::Derivative<ethbar_ethbar_rt_tag,
                                       Spectral::Swsh::Tags::EthEth>;
  Variables<tmpl::list<ethbar_ethbar_rt_tag, eth_eth_ethbar_ethbar_rt_tag>>
      temporary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max_)};
  auto& ethbar_ethbar_rt_scalar =
      get<ethbar_ethbar_rt_tag>(temporary_variables);
  auto& eth_eth_ethbar_ethbar_rt_scalar =
      get<eth_eth_ethbar_ethbar_rt_tag>(temporary_variables);
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::EthbarEthbar>>(
      l_max_, 1, make_not_null(&get(ethbar_ethbar_rt_scalar)), rt_scalar);
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::EthEth>>(
      l_max_, 1, make_not_null(&get(eth_eth_ethbar_ethbar_rt_scalar)),
      get(ethbar_ethbar_rt_scalar));
  *local_du_rt_scalar =
      (-pow<4>(rt_scalar) * get(eth_eth_ethbar_ethbar_rt_scalar) +
       pow<3>(rt_scalar) * get(ethbar_ethbar_rt_scalar) *
           conj(get(ethbar_ethbar_rt_scalar))) /
      12.0;
}

void RobinsonTrautman::bondi_u(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 1>>*> bondi_u,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_scalar)
    const noexcept {
  Spectral::Swsh::angular_derivatives<tmpl::list<Spectral::Swsh::Tags::Eth>>(
      l_max_, 1, make_not_null(&get(*bondi_u)), get(rt_scalar));
  get(*bondi_u).data() /= extraction_radius_;
}

void RobinsonTrautman::bondi_w(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_scalar)
    const noexcept {
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::EthEthbar>>(
      l_max_, 1, make_not_null(&get(*bondi_w)), get(rt_scalar));
  get(*bondi_w) = (get(rt_scalar) + get(*bondi_w) - 1.0) / extraction_radius_ -
                  2.0 / square(extraction_radius_ * get(rt_scalar));
}

void RobinsonTrautman::dr_bondi_w(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> dr_bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_scalar)
    const noexcept {
  SpinWeighted<ComplexDataVector, 0> bondi_w{get(*dr_bondi_w).size()};
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::EthEthbar>>(
      l_max_, 1, make_not_null(&bondi_w), get(rt_scalar));
  get(*dr_bondi_w) =
      -(get(rt_scalar) + bondi_w - 1.0) / square(extraction_radius_) +
      4.0 / (pow<3>(extraction_radius_) * square(get(rt_scalar)));
}

void RobinsonTrautman::du_bondi_w(
    const gsl::not_null<Scalar<SpinWeighted<ComplexDataVector, 0>>*> du_bondi_w,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& local_du_rt_scalar,
    const Scalar<SpinWeighted<ComplexDataVector, 0>>& rt_scalar)
    const noexcept {
  Spectral::Swsh::angular_derivatives<
      tmpl::list<Spectral::Swsh::Tags::EthEthbar>>(
      l_max_, 1, make_not_null(&get(*du_bondi_w)), get(local_du_rt_scalar));
  get(*du_bondi_w) =
      (get(local_du_rt_scalar) + get(*du_bondi_w)) / extraction_radius_ +
      4.0 * get(local_du_rt_scalar) /
          (square(extraction_radius_) * pow<3>(get(rt_scalar)));
}

void RobinsonTrautman::spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        spherical_metric,
    const size_t l_max, const double time) const noexcept {
  ASSERT(time == prepared_time_,
         "The Robinson-Trautman solution is being calculated in an "
         "inconsistent state. The public interface should always call "
         "`prepare_solution` before calculating any metric components, so this "
         "may indicate inconsistent use of the internal Robinson-Trautman "
         "calculation functions.");
  ASSERT(l_max == l_max_,
         "The Robinson-Trautman solution only supports the l_max resolution "
         "specified at construction, as it must internally store the evolved "
         "scalar at the desired resolution");
  Variables<tmpl::list<Tags::BondiW, Tags::BondiU>> temporary_variables{
      Spectral::Swsh::number_of_swsh_collocation_points(l_max_)};
  auto& local_bondi_u = get<Tags::BondiU>(temporary_variables);
  auto& local_bondi_w = get<Tags::BondiW>(temporary_variables);

  bondi_u(make_not_null(&local_bondi_u), dense_output_rt_scalar_);
  bondi_w(make_not_null(&local_bondi_w), dense_output_rt_scalar_);

  get<0, 0>(*spherical_metric) =
      -real(get(dense_output_rt_scalar_).data() *
                (extraction_radius_ * get(local_bondi_w).data() + 1.0) -
            square(extraction_radius_) * get(local_bondi_u).data() *
                conj(get(local_bondi_u).data()));
  get<0, 1>(*spherical_metric) =
      -get<0, 0>(*spherical_metric) - real(get(dense_output_rt_scalar_).data());
  get<1, 1>(*spherical_metric) =
      get<0, 0>(*spherical_metric) +
      2.0 * real(get(dense_output_rt_scalar_).data());

  get<0, 2>(*spherical_metric) =
      square(extraction_radius_) * real(get(local_bondi_u).data());
  get<1, 2>(*spherical_metric) =
      -square(extraction_radius_) * real(get(local_bondi_u).data());
  // note: using 'pfaffian' components, because otherwise the sin(theta)s lose
  // precision at the poles
  get<0, 3>(*spherical_metric) =
      square(extraction_radius_) * imag(get(local_bondi_u).data());
  get<1, 3>(*spherical_metric) =
      -square(extraction_radius_) * imag(get(local_bondi_u).data());

  // note: pfaffian components
  get<2, 2>(*spherical_metric) = square(extraction_radius_);
  get<2, 3>(*spherical_metric) = 0.0;
  get<3, 3>(*spherical_metric) = square(extraction_radius_);
}

void RobinsonTrautman::dr_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dr_spherical_metric,
    const size_t l_max, const double time) const noexcept {
  ASSERT(time == prepared_time_,
         "The Robinson-Trautman solution is being calculated in an "
         "inconsistent state. The public interface should always call "
         "`prepare_solution` before calculating any metric components, so this "
         "may indicate inconsistent use of the internal Robinson-Trautman "
         "calculation functions.");
  ASSERT(l_max == l_max_,
         "The Robinson-Trautman solution only supports the l_max resolution "
         "specified at construction, as it must internally store the evolved "
         "scalar at the desired resolution");
  Variables<tmpl::list<Tags::BondiW, Tags::Dr<Tags::BondiW>, Tags::BondiU>>
      temporary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max_)};
  auto& local_dr_bondi_w = get<Tags::Dr<Tags::BondiW>>(temporary_variables);
  auto& local_bondi_w = get<Tags::BondiW>(temporary_variables);
  auto& local_bondi_u = get<Tags::BondiU>(temporary_variables);

  bondi_u(make_not_null(&local_bondi_u), dense_output_rt_scalar_);
  dr_bondi_w(make_not_null(&local_dr_bondi_w), dense_output_rt_scalar_);
  bondi_w(make_not_null(&local_bondi_w), dense_output_rt_scalar_);

  dt_spherical_metric(dr_spherical_metric, l_max, time);

  // note: bondi_u is proportional to 1/r, so when comparing to the values in
  // `spherical_metric`, this (correctly) doesn't have a factor of two where you
  // might naively expect one.
  get<0, 0>(*dr_spherical_metric) =
      -get<0, 0>(*dr_spherical_metric) -
      real(get(dense_output_rt_scalar_).data() *
           (extraction_radius_ * get(local_dr_bondi_w).data() +
            get(local_bondi_w).data()));
  get<0, 1>(*dr_spherical_metric) =
      real(get(dense_output_rt_scalar_).data() *
           (extraction_radius_ * get(local_dr_bondi_w).data() +
            get(local_bondi_w).data())) -
      get<0, 1>(*dr_spherical_metric);
  get<1, 1>(*dr_spherical_metric) =
      -real(get(dense_output_rt_scalar_).data() *
            (extraction_radius_ * get(local_dr_bondi_w).data() +
             get(local_bondi_w).data())) -
      get<1, 1>(*dr_spherical_metric);

  get<0, 2>(*dr_spherical_metric) =
      -get<0, 2>(*dr_spherical_metric) +
      extraction_radius_ * real(get(local_bondi_u).data());
  get<1, 2>(*dr_spherical_metric) =
      -get<1, 2>(*dr_spherical_metric) +
      -extraction_radius_ * real(get(local_bondi_u).data());
  // note: using 'pfaffian' components, because otherwise the factors of
  // sin(theta) lose precision at the poles
  get<0, 3>(*dr_spherical_metric) =
      -get<0, 3>(*dr_spherical_metric) +
      extraction_radius_ * imag(get(local_bondi_u).data());
  get<1, 3>(*dr_spherical_metric) =
      -get<1, 3>(*dr_spherical_metric) +
      -extraction_radius_ * imag(get(local_bondi_u).data());

  // note: pfaffian components
  get<2, 2>(*dr_spherical_metric) =
      -get<2, 2>(*dr_spherical_metric) + 2.0 * extraction_radius_;
  get<2, 3>(*dr_spherical_metric) = 0.0;
  get<3, 3>(*dr_spherical_metric) =
      -get<3, 3>(*dr_spherical_metric) + 2.0 * extraction_radius_;
}

void RobinsonTrautman::dt_spherical_metric(
    const gsl::not_null<
        tnsr::aa<DataVector, 3, ::Frame::Spherical<::Frame::Inertial>>*>
        dt_spherical_metric,
    const size_t l_max, const double time) const noexcept {
  ASSERT(time == prepared_time_,
         "The Robinson-Trautman solution is being calculated in an "
         "inconsistent state. The public interface should always call "
         "`prepare_solution` before calculating any metric components, so this "
         "may indicate inconsistent use of the internal Robinson-Trautman "
         "calculation functions.");
  ASSERT(l_max == l_max_,
         "The Robinson-Trautman solution only supports the l_max resolution "
         "specified at construction, as it must internally store the evolved "
         "scalar at the desired resolution");
  Variables<tmpl::list<Tags::BondiW, Tags::Du<Tags::BondiW>, Tags::BondiU,
                       Tags::Du<Tags::BondiU>>>
      temporary_variables{
          Spectral::Swsh::number_of_swsh_collocation_points(l_max_)};
  auto& local_bondi_w = get<Tags::BondiW>(temporary_variables);
  auto& local_bondi_u = get<Tags::BondiU>(temporary_variables);
  auto& local_du_bondi_w = get<Tags::Du<Tags::BondiW>>(temporary_variables);
  auto& local_du_bondi_u = get<Tags::Du<Tags::BondiU>>(temporary_variables);
  bondi_u(make_not_null(&local_du_bondi_u), dense_output_du_rt_scalar_);
  bondi_u(make_not_null(&local_bondi_u), dense_output_rt_scalar_);
  du_bondi_w(make_not_null(&local_du_bondi_w), dense_output_du_rt_scalar_,
             dense_output_rt_scalar_);
  bondi_w(make_not_null(&local_bondi_w), dense_output_rt_scalar_);

  get<0, 0>(*dt_spherical_metric) = -real(
      get(dense_output_du_rt_scalar_).data() *
          (extraction_radius_ * get(local_bondi_w).data() + 1.0) +
      get(dense_output_rt_scalar_).data() * extraction_radius_ *
          get(local_du_bondi_w).data() -
      square(extraction_radius_) *
          (get(local_du_bondi_u).data() * conj(get(local_bondi_u).data()) +
           conj(get(local_du_bondi_u).data()) * get(local_bondi_u).data()));
  get<0, 1>(*dt_spherical_metric) =
      -get<0, 0>(*dt_spherical_metric) -
      real(get(dense_output_du_rt_scalar_).data());
  get<1, 1>(*dt_spherical_metric) =
      get<0, 0>(*dt_spherical_metric) +
      2.0 * real(get(dense_output_du_rt_scalar_).data());

  get<0, 2>(*dt_spherical_metric) =
      square(extraction_radius_) * real(get(local_du_bondi_u).data());
  get<1, 2>(*dt_spherical_metric) =
      -square(extraction_radius_) * real(get(local_du_bondi_u).data());
  get<0, 3>(*dt_spherical_metric) =
      square(extraction_radius_) * imag(get(local_du_bondi_u).data());
  get<1, 3>(*dt_spherical_metric) =
      -square(extraction_radius_) * imag(get(local_du_bondi_u).data());
  for (size_t A = 0; A < 2; ++A) {
    for (size_t B = A; B < 2; ++B) {
      dt_spherical_metric->get(A + 2, B + 2) = 0.0;
    }
  }
}

void RobinsonTrautman::pup(PUP::er& p) noexcept {
  SphericalMetricData::pup(p);
  p | tolerance_;
  p | start_time_;
  p | dense_output_rt_scalar_;
  p | dense_output_du_rt_scalar_;
  p | l_max_;
  p | initial_modes_;
  // it is difficult to serialize the stepper, but because it will usually not
  // migrate, it should be fine to just re-initialize it with the candidate
  // starting step size at the starting time. It will need to 'catch up' to the
  // current evolution each time it is migrated.
  // This procedure should be reconsidered if this analytic solution is used in
  // a context (e.g. in a chare array) where it may need to migrate frequently.
  if (p.isUnpacking()) {
    initialize_stepper_from_start();
  }
}

PUP::able::PUP_ID RobinsonTrautman::my_PUP_ID = 0;
}  // namespace Cce::Solutions
