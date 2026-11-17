#pragma once

#include <cmath>
#include <algorithm>
#include <memory>
#include "DataStructures/DataVector.hpp"
#include "ControlSystem/FunctionsOfTime/FunctionOfTime.hpp"

namespace GeneralizedHarmonic::ControlSystem {

class DampingGaugeFeedback {
 public:
  struct TimeConstant {
    using type = double;
    static constexpr Options::String help = "Time constant (1/nu_r) in units of M";
  };
  struct MaxScaleFactor {
    using type = double;
    static constexpr Options::String help = "Maximum allowed scale factor (s_max)";
  };
  struct ConstraintThreshold {
    using type = double;
    static constexpr Options::String help = "Baseline constraint energy threshold (c_hi)";
  };

  using options = tmpl::list<TimeConstant, MaxScaleFactor, ConstraintThreshold>;
  
  static constexpr Options::String help = 
      "Feedback loop regulating damping and gauge amplitudes.";

  DampingGaugeFeedback(double time_constant, double max_scale_factor, double constraint_threshold)
    : time_constant_(time_constant), 
      max_scale_factor_(max_scale_factor), 
      constraint_threshold_(constraint_threshold) {}

  void update_scale_factors(
      gsl::not_null<double*> s_gamma,
      gsl::not_null<double*> s_mu,
      const double current_constraint_norm,
      const double current_volume_theta,
      const double min_outflow_speed,
      const double baseline_outflow_speed,
      const double dt_c,
      const double kappa,
      const double nu_r,
      const double nu_gamma,
      const double nu_mu) const {
      
    const double f_m = 0.95; 
    // Strict Interlock: freezes both relaxation and gain when outflow speed drops
    if (min_outflow_speed < f_m * baseline_outflow_speed) {
        return; 
    }

    const double log_c_ratio = std::log(current_constraint_norm / constraint_threshold_);
    const double G_c = std::tanh(kappa * std::max(0.0, log_c_ratio));
    
    const double theta_hi = 0.5; 
    const double log_theta_ratio = std::log(current_volume_theta / theta_hi);
    const double G_t = std::tanh(kappa * std::max(0.0, log_theta_ratio));

    double ln_s_gamma = std::log(*s_gamma);
    double ln_s_mu = std::log(*s_mu);

    ln_s_gamma += dt_c * (-nu_r * ln_s_gamma + nu_gamma * G_c);
    ln_s_mu    += dt_c * (-nu_r * ln_s_mu    + nu_mu    * G_t);

    *s_gamma = std::min(max_scale_factor_, std::exp(ln_s_gamma));
    *s_mu    = std::min(max_scale_factor_, std::exp(ln_s_mu));
  }

 private:
  double time_constant_;
  double max_scale_factor_;
  double constraint_threshold_;
};

} // namespace GeneralizedHarmonic::ControlSystem
