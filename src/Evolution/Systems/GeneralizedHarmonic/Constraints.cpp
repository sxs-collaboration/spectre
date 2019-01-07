// Distributed under the MIT License.
// See LICENSE.txt for details.

#include "Evolution/Systems/GeneralizedHarmonic/Constraints.hpp"

#include <cstddef>

#include "DataStructures/LeviCivitaIterator.hpp"
#include "DataStructures/Tensor/Tensor.hpp"  // IWYU pragma: keep
#include "Utilities/ConstantExpressions.hpp"
#include "Utilities/ContainerHelpers.hpp"
#include "Utilities/GenerateInstantiations.hpp"
#include "Utilities/Gsl.hpp"
#include "Utilities/MakeWithValue.hpp"

namespace {
// Functions to compute generalized-harmonic 2-index constraint, where
// function arguments are in the order that each quantity first
// appears in the corresponding term in Eq. (44) of
// https://arXiv.org/abs/gr-qc/0512093v3
template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_1_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        for (size_t k = 0; k < SpatialDim; ++k) {
          constraint->get(i, a) +=
              inverse_spatial_metric.get(j, k) * d_phi.get(j, i, k + 1, a);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_2_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          if (a > 0) {
            constraint->get(i, a) -= 0.5 * inverse_spacetime_metric.get(c, d) *
                                     d_phi.get(a - 1, i, c, d);
          }
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(i, a) -= 0.5 * spacetime_normal_vector.get(j + 1) *
                                     spacetime_normal_one_form.get(a) *
                                     inverse_spacetime_metric.get(c, d) *
                                     d_phi.get(j, i, c, d);
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_3_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        constraint->get(i, a) +=
            spacetime_normal_vector.get(b) * d_pi.get(i, b, a);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_4_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          constraint->get(i, a) -= 0.5 * spacetime_normal_one_form.get(a) *
                                   inverse_spacetime_metric.get(c, d) *
                                   d_pi.get(i, c, d);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_5_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      constraint->get(i, a) += d_gauge_function.get(i, a);
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_6_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            for (size_t f = 0; f < SpatialDim + 1; ++f) {
              if (a > 0) {
                constraint->get(i, a) += 0.5 * phi.get(a - 1, c, d) *
                                         phi.get(i, e, f) *
                                         inverse_spacetime_metric.get(c, e) *
                                         inverse_spacetime_metric.get(d, f);
              }
              for (size_t j = 0; j < SpatialDim; ++j) {
                constraint->get(i, a) +=
                    0.5 * spacetime_normal_vector.get(j + 1) *
                    spacetime_normal_one_form.get(a) * phi.get(j, c, d) *
                    phi.get(i, e, f) * inverse_spacetime_metric.get(c, e) *
                    inverse_spacetime_metric.get(d, f);
              }
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_7_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::a<DataType, SpatialDim, Frame>&
        spacetime_normal_one_form) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        for (size_t k = 0; k < SpatialDim; ++k) {
          for (size_t c = 0; c < SpatialDim + 1; ++c) {
            for (size_t d = 0; d < SpatialDim + 1; ++d) {
              for (size_t e = 0; e < SpatialDim + 1; ++e) {
                constraint->get(i, a) +=
                    0.5 * inverse_spatial_metric.get(j, k) * phi.get(j, c, d) *
                    phi.get(i, k + 1, e) * inverse_spacetime_metric.get(c, d) *
                    spacetime_normal_vector.get(e) *
                    spacetime_normal_one_form.get(a);
              }
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_8_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        for (size_t k = 0; k < SpatialDim; ++k) {
          for (size_t m = 0; m < SpatialDim; ++m) {
            for (size_t n = 0; n < SpatialDim; ++n) {
              constraint->get(i, a) -= inverse_spatial_metric.get(j, k) *
                                       inverse_spatial_metric.get(m, n) *
                                       phi.get(j, m + 1, a) *
                                       phi.get(i, k + 1, n + 1);
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_9_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>&
        spacetime_normal_vector) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        for (size_t c = 0; c < SpatialDim + 1; ++c) {
          for (size_t d = 0; d < SpatialDim + 1; ++d) {
            for (size_t e = 0; e < SpatialDim + 1; ++e) {
              constraint->get(i, a) +=
                  0.5 * phi.get(i, c, d) * pi.get(b, e) *
                  spacetime_normal_one_form.get(a) *
                  (inverse_spacetime_metric.get(c, b) *
                       inverse_spacetime_metric.get(d, e) +
                   0.5 * inverse_spacetime_metric.get(b, e) *
                       spacetime_normal_vector.get(c) *
                       spacetime_normal_vector.get(d));
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_10_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t b = 0; b < SpatialDim + 1; ++b) {
        for (size_t c = 0; c < SpatialDim + 1; ++c) {
          for (size_t d = 0; d < SpatialDim + 1; ++d) {
            constraint->get(i, a) -= phi.get(i, c, d) * pi.get(b, a) *
                                     spacetime_normal_vector.get(c) *
                                     (inverse_spacetime_metric.get(b, d) +
                                      0.5 * spacetime_normal_vector.get(b) *
                                          spacetime_normal_vector.get(d));
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint_add_term_11_of_11(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const Scalar<DataType>& gamma2,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept {
  for (size_t i = 0; i < SpatialDim; ++i) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t c = 0; c < SpatialDim + 1; ++c) {
          constraint->get(i, a) += 0.5 * get(gamma2) *
                                   spacetime_normal_one_form.get(a) *
                                   inverse_spacetime_metric.get(c, d) *
                                   three_index_constraint.get(i, c, d);
        }
        constraint->get(i, a) -= get(gamma2) * spacetime_normal_vector.get(d) *
                                 three_index_constraint.get(i, a, d);
      }
    }
  }
}

// Functions to compute generalized-harmonic F constraint, where
// function arguments are in the order that each quantity first
// appears in the corresponding term in Eq. (43) of
// https://arXiv.org/abs/gr-qc/0512093v3
template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_1_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        if (a > 0) {
          constraint->get(a) +=
              0.5 * inverse_spacetime_metric.get(b, c) * d_pi.get(a - 1, b, c);
        }
        for (size_t i = 0; i < SpatialDim; ++i) {
          constraint->get(a) += 0.5 * spacetime_normal_vector.get(i + 1) *
                                spacetime_normal_one_form.get(a) *
                                inverse_spacetime_metric.get(b, c) *
                                d_pi.get(i, b, c);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_2_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        constraint->get(a) -=
            inverse_spatial_metric.get(i, j) * d_pi.get(i, j + 1, a);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_3_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        for (size_t b = 0; b < SpatialDim + 1; ++b) {
          constraint->get(a) -= inverse_spatial_metric.get(i, j) *
                                spacetime_normal_vector.get(b) *
                                d_phi.get(i, j, b, a);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_4_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) += 0.5 * spacetime_normal_one_form.get(a) *
                                  inverse_spacetime_metric.get(b, c) *
                                  inverse_spatial_metric.get(i, j) *
                                  d_phi.get(i, j, b, c);
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_5_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        constraint->get(a) += spacetime_normal_one_form.get(a) *
                              inverse_spatial_metric.get(i, j) *
                              d_gauge_function.get(i, j + 1);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_6_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            for (size_t k = 0; k < SpatialDim; ++k) {
              if (a > 0) {
                constraint->get(a) += phi.get(a - 1, j + 1, b) *
                                      inverse_spatial_metric.get(j, k) *
                                      phi.get(k, c, d) *
                                      inverse_spacetime_metric.get(b, d) *
                                      spacetime_normal_vector.get(c);
              }
              for (size_t i = 0; i < SpatialDim; ++i) {
                constraint->get(a) +=
                    spacetime_normal_one_form.get(a) *
                    spacetime_normal_vector.get(i + 1) * phi.get(i, j + 1, b) *
                    inverse_spatial_metric.get(j, k) * phi.get(k, c, d) *
                    inverse_spacetime_metric.get(b, d) *
                    spacetime_normal_vector.get(c);
              }
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_7_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            for (size_t k = 0; k < SpatialDim; ++k) {
              if (a > 0) {
                constraint->get(a) -= 0.5 * phi.get(a - 1, j + 1, b) *
                                      inverse_spatial_metric.get(j, k) *
                                      phi.get(k, c, d) *
                                      inverse_spacetime_metric.get(c, d) *
                                      spacetime_normal_vector.get(b);
              }
              for (size_t i = 0; i < SpatialDim; ++i) {
                constraint->get(a) -=
                    0.5 * spacetime_normal_one_form.get(a) *
                    spacetime_normal_vector.get(i + 1) * phi.get(i, j + 1, b) *
                    inverse_spatial_metric.get(j, k) * phi.get(k, c, d) *
                    inverse_spacetime_metric.get(c, d) *
                    spacetime_normal_vector.get(b);
              }
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_8_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      if (a > 0) {
        constraint->get(a) -=
            spacetime_normal_vector.get(b) * d_gauge_function.get(a - 1, b);
      }
      for (size_t i = 0; i < SpatialDim; ++i) {
        constraint->get(a) -= spacetime_normal_one_form.get(a) *
                              spacetime_normal_vector.get(i + 1) *
                              spacetime_normal_vector.get(b) *
                              d_gauge_function.get(i, b);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_9_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>&
        spacetime_normal_vector) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t i = 0; i < SpatialDim; ++i) {
            for (size_t j = 0; j < SpatialDim; ++j) {
              constraint->get(a) += inverse_spatial_metric.get(i, j) *
                                    phi.get(i, c, d) * phi.get(j, b, a) *
                                    inverse_spacetime_metric.get(b, c) *
                                    spacetime_normal_vector.get(d);
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_10_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            for (size_t m = 0; m < SpatialDim; ++m) {
              for (size_t n = 0; n < SpatialDim; ++n) {
                constraint->get(a) -=
                    0.5 * spacetime_normal_one_form.get(a) *
                    inverse_spatial_metric.get(i, j) *
                    inverse_spatial_metric.get(m, n) * phi.get(i, m + 1, c) *
                    phi.get(n, j + 1, d) * inverse_spacetime_metric.get(c, d);
              }
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_11_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            for (size_t i = 0; i < SpatialDim; ++i) {
              for (size_t j = 0; j < SpatialDim; ++j) {
                constraint->get(a) -= 0.25 * spacetime_normal_one_form.get(a) *
                                      inverse_spatial_metric.get(i, j) *
                                      phi.get(i, c, d) * phi.get(j, b, e) *
                                      inverse_spacetime_metric.get(c, b) *
                                      inverse_spacetime_metric.get(d, e);
              }
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_12_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            constraint->get(a) += 0.25 * spacetime_normal_one_form.get(a) *
                                  pi.get(c, d) * pi.get(b, e) *
                                  inverse_spacetime_metric.get(c, b) *
                                  inverse_spacetime_metric.get(d, e);
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_13_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        constraint->get(a) -= inverse_spatial_metric.get(i, j) *
                              gauge_function.get(i + 1) * pi.get(j + 1, a);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_14_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        for (size_t j = 0; j < SpatialDim; ++j) {
          constraint->get(a) -= spacetime_normal_vector.get(b) *
                                inverse_spatial_metric.get(i, j) *
                                pi.get(b, i + 1) * pi.get(j + 1, a);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_15_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            if (a > 0) {
              constraint->get(a) -=
                  0.25 * phi.get(a - 1, c, d) * spacetime_normal_vector.get(c) *
                  spacetime_normal_vector.get(d) * pi.get(b, e) *
                  inverse_spacetime_metric.get(b, e);
            }
            for (size_t i = 0; i < SpatialDim; ++i) {
              constraint->get(a) -=
                  0.25 * spacetime_normal_one_form.get(a) *
                  spacetime_normal_vector.get(i + 1) * phi.get(i, c, d) *
                  spacetime_normal_vector.get(c) *
                  spacetime_normal_vector.get(d) * pi.get(b, e) *
                  inverse_spacetime_metric.get(b, e);
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_16_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>&
        spacetime_normal_vector) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            constraint->get(a) +=
                0.5 * spacetime_normal_one_form.get(a) * pi.get(c, d) *
                pi.get(b, e) * inverse_spacetime_metric.get(c, e) *
                spacetime_normal_vector.get(d) * spacetime_normal_vector.get(b);
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_17_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          for (size_t e = 0; e < SpatialDim + 1; ++e) {
            if (a > 0) {
              constraint->get(a) += phi.get(a - 1, c, d) * pi.get(b, e) *
                                    spacetime_normal_vector.get(c) *
                                    spacetime_normal_vector.get(b) *
                                    inverse_spacetime_metric.get(d, e);
            }
            for (size_t i = 0; i < SpatialDim; ++i) {
              constraint->get(a) += spacetime_normal_one_form.get(a) *
                                    spacetime_normal_vector.get(i + 1) *
                                    phi.get(i, c, d) * pi.get(b, e) *
                                    spacetime_normal_vector.get(c) *
                                    spacetime_normal_vector.get(b) *
                                    inverse_spacetime_metric.get(d, e);
            }
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_18_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t e = 0; e < SpatialDim + 1; ++e) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) -=
                inverse_spatial_metric.get(i, j) * phi.get(i, b, a) *
                spacetime_normal_vector.get(b) * pi.get(j + 1, e) *
                spacetime_normal_vector.get(e);
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_19_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) -=
                0.5 * inverse_spatial_metric.get(i, j) * phi.get(i, c, d) *
                spacetime_normal_vector.get(c) *
                spacetime_normal_vector.get(d) * pi.get(j + 1, a);
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_20_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::A<DataType, SpatialDim, Frame>&
        spacetime_normal_vector) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        for (size_t j = 0; j < SpatialDim; ++j) {
          constraint->get(a) -= inverse_spatial_metric.get(i, j) *
                                gauge_function.get(i + 1) * phi.get(j, b, a) *
                                spacetime_normal_vector.get(b);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_21_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          if (a > 0) {
            constraint->get(a) += phi.get(a - 1, c, d) * gauge_function.get(b) *
                                  inverse_spacetime_metric.get(b, c) *
                                  spacetime_normal_vector.get(d);
          }
          for (size_t i = 0; i < SpatialDim; ++i) {
            constraint->get(a) += spacetime_normal_one_form.get(a) *
                                  spacetime_normal_vector.get(i + 1) *
                                  phi.get(i, c, d) * gauge_function.get(b) *
                                  inverse_spacetime_metric.get(b, c) *
                                  spacetime_normal_vector.get(d);
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_22_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const Scalar<DataType>& gamma2,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::a<DataType, SpatialDim, Frame>&
        spacetime_normal_one_form) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t d = 0; d < SpatialDim + 1; ++d) {
      for (size_t i = 0; i < SpatialDim; ++i) {
        constraint->get(a) += get(gamma2) *
                              inverse_spacetime_metric.get(i + 1, d) *
                              three_index_constraint.get(i, d, a);
        constraint->get(a) += get(gamma2) * spacetime_normal_vector.get(i + 1) *
                              spacetime_normal_vector.get(d) *
                              three_index_constraint.get(i, d, a);
      }

      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        if (a > 0) {
          constraint->get(a) -= 0.5 * get(gamma2) *
                                inverse_spacetime_metric.get(c, d) *
                                three_index_constraint.get(a - 1, c, d);
        }
        for (size_t i = 0; i < SpatialDim; ++i) {
          constraint->get(a) -= 0.5 * get(gamma2) *
                                spacetime_normal_one_form.get(a) *
                                spacetime_normal_vector.get(i + 1) *
                                inverse_spacetime_metric.get(c, d) *
                                three_index_constraint.get(i, c, d);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_23_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::A<DataType, SpatialDim, Frame>&
        spacetime_normal_vector) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        for (size_t d = 0; d < SpatialDim + 1; ++d) {
          constraint->get(a) +=
              0.5 * spacetime_normal_one_form.get(a) * pi.get(c, d) *
              inverse_spacetime_metric.get(c, d) * gauge_function.get(b) *
              spacetime_normal_vector.get(b);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_24_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) -= spacetime_normal_one_form.get(a) *
                                  inverse_spatial_metric.get(i, j) *
                                  phi.get(i, j + 1, c) * gauge_function.get(d) *
                                  inverse_spacetime_metric.get(c, d);
          }
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint_add_term_25_of_25(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::AA<DataType, SpatialDim, Frame>&
        inverse_spacetime_metric) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t c = 0; c < SpatialDim + 1; ++c) {
      for (size_t d = 0; d < SpatialDim + 1; ++d) {
        for (size_t i = 0; i < SpatialDim; ++i) {
          for (size_t j = 0; j < SpatialDim; ++j) {
            constraint->get(a) += 0.5 * spacetime_normal_one_form.get(a) *
                                  inverse_spatial_metric.get(i, j) *
                                  gauge_function.get(i + 1) * phi.get(j, c, d) *
                                  inverse_spacetime_metric.get(c, d);
          }
        }
      }
    }
  }
}
}  // namespace

namespace GeneralizedHarmonic {
template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> three_index_constraint(
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto constraint =
      make_with_value<tnsr::iaa<DataType, SpatialDim, Frame>>(phi, 0.0);
  three_index_constraint<SpatialDim, Frame, DataType>(&constraint,
                                                      d_spacetime_metric, phi);
  return constraint;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void three_index_constraint(
    const gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_spacetime_metric,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  // Declare iterators for d_spacetime_metric and phi outside the for loop,
  // because they are const but constraint is not
  auto d_spacetime_metric_it = d_spacetime_metric.begin(), phi_it = phi.begin();

  for (auto constraint_it = (*constraint).begin();
       constraint_it != (*constraint).end();
       ++constraint_it, (void)++d_spacetime_metric_it, (void)++phi_it) {
    *constraint_it = *d_spacetime_metric_it - *phi_it;
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> gauge_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  auto constraint =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(pi, 0.0);
  gauge_constraint<SpatialDim, Frame, DataType>(
      &constraint, gauge_function, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric, inverse_spacetime_metric,
      pi, phi);
  return constraint;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void gauge_constraint(
    const gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi) noexcept {
  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    (*constraint).get(a) = gauge_function.get(a);
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        (*constraint).get(a) +=
            inverse_spatial_metric.get(i, j) * phi.get(i, j + 1, a);
      }
    }
    for (size_t b = 0; b < SpatialDim + 1; ++b) {
      (*constraint).get(a) += spacetime_normal_vector.get(b) * pi.get(b, a);
      for (size_t c = 0; c < SpatialDim + 1; ++c) {
        (*constraint).get(a) -= 0.5 * spacetime_normal_one_form.get(a) *
                                pi.get(b, c) *
                                inverse_spacetime_metric.get(b, c);
        if (a > 0) {
          (*constraint).get(a) -=
              0.5 * phi.get(a - 1, b, c) * inverse_spacetime_metric.get(b, c);
        }
        for (size_t i = 0; i < SpatialDim; ++i) {
          (*constraint).get(a) -= 0.5 * spacetime_normal_one_form.get(a) *
                                  spacetime_normal_vector.get(i + 1) *
                                  phi.get(i, b, c) *
                                  inverse_spacetime_metric.get(b, c);
        }
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::ia<DataType, SpatialDim, Frame> two_index_constraint(
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept {
  auto constraint =
      make_with_value<tnsr::ia<DataType, SpatialDim, Frame>>(pi, 0.0);
  two_index_constraint<SpatialDim, Frame, DataType>(
      &constraint, d_gauge_function, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric, inverse_spacetime_metric,
      pi, phi, d_pi, d_phi, gamma2, three_index_constraint);
  return constraint;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void two_index_constraint(
    const gsl::not_null<tnsr::ia<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept {
  for (auto& component : *constraint) {
    component = 0.0;
  }

  two_index_constraint_add_term_1_of_11(constraint, inverse_spatial_metric,
                                        d_phi);
  two_index_constraint_add_term_2_of_11(constraint, spacetime_normal_vector,
                                        spacetime_normal_one_form,
                                        inverse_spacetime_metric, d_phi);
  two_index_constraint_add_term_3_of_11(constraint, spacetime_normal_vector,
                                        d_pi);
  two_index_constraint_add_term_4_of_11(constraint, spacetime_normal_one_form,
                                        inverse_spacetime_metric, d_pi);
  two_index_constraint_add_term_5_of_11(constraint, d_gauge_function);
  two_index_constraint_add_term_6_of_11(constraint, spacetime_normal_vector,
                                        spacetime_normal_one_form, phi,
                                        inverse_spacetime_metric);
  two_index_constraint_add_term_7_of_11(
      constraint, inverse_spatial_metric, phi, inverse_spacetime_metric,
      spacetime_normal_vector, spacetime_normal_one_form);
  two_index_constraint_add_term_8_of_11(constraint, inverse_spatial_metric,
                                        phi);
  two_index_constraint_add_term_9_of_11(
      constraint, phi, pi, spacetime_normal_one_form, inverse_spacetime_metric,
      spacetime_normal_vector);
  two_index_constraint_add_term_10_of_11(
      constraint, phi, pi, spacetime_normal_vector, inverse_spacetime_metric);
  two_index_constraint_add_term_11_of_11(
      constraint, gamma2, spacetime_normal_one_form, inverse_spacetime_metric,
      spacetime_normal_vector, three_index_constraint);
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::iaa<DataType, SpatialDim, Frame> four_index_constraint(
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept {
  static_assert(
      SpatialDim == 3,
      "four_index_constraint() currently only supports 3 spatial dimensions");
  auto constraint =
      make_with_value<tnsr::iaa<DataType, SpatialDim, Frame>>(d_phi, 0.0);
  four_index_constraint<SpatialDim, Frame, DataType>(&constraint, d_phi);
  return constraint;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void four_index_constraint(
    const gsl::not_null<tnsr::iaa<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi) noexcept {
  static_assert(
      SpatialDim == 3,
      "four_index_constraint() currently only supports 3 spatial dimensions");
  if (get_size(get<0, 0, 0>(*constraint)) != get_size(get<0, 0, 0, 0>(d_phi))) {
    *constraint =
        tnsr::iaa<DataType, SpatialDim, Frame>{get<0, 0, 0, 0>(d_phi)};
  }
  std::fill(constraint->begin(), constraint->end(), 0.0);

  for (LeviCivitaIterator<SpatialDim> it; it; ++it) {
    for (size_t a = 0; a < SpatialDim + 1; ++a) {
      // Constraint is symmetric on a,b. Avoid double-counting in summation
      // by starting the inner loop at b = a.
      for (size_t b = a; b < SpatialDim + 1; ++b) {
        constraint->get(it[0], a, b) +=
            it.sign() * d_phi.get(it[1], it[2], a, b);
      }
    }
  }
}

template <size_t SpatialDim, typename Frame, typename DataType>
tnsr::a<DataType, SpatialDim, Frame> f_constraint(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept {
  auto constraint =
      make_with_value<tnsr::a<DataType, SpatialDim, Frame>>(pi, 0.0);
  f_constraint<SpatialDim, Frame, DataType>(
      &constraint, gauge_function, d_gauge_function, spacetime_normal_one_form,
      spacetime_normal_vector, inverse_spatial_metric, inverse_spacetime_metric,
      pi, phi, d_pi, d_phi, gamma2, three_index_constraint);
  return constraint;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void f_constraint(
    gsl::not_null<tnsr::a<DataType, SpatialDim, Frame>*> constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_function,
    const tnsr::ia<DataType, SpatialDim, Frame>& d_gauge_function,
    const tnsr::a<DataType, SpatialDim, Frame>& spacetime_normal_one_form,
    const tnsr::A<DataType, SpatialDim, Frame>& spacetime_normal_vector,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const tnsr::AA<DataType, SpatialDim, Frame>& inverse_spacetime_metric,
    const tnsr::aa<DataType, SpatialDim, Frame>& pi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& phi,
    const tnsr::iaa<DataType, SpatialDim, Frame>& d_pi,
    const tnsr::ijaa<DataType, SpatialDim, Frame>& d_phi,
    const Scalar<DataType>& gamma2,
    const tnsr::iaa<DataType, SpatialDim, Frame>&
        three_index_constraint) noexcept {
  if (get_size(get<0>(*constraint)) != get_size(get<0, 0>(pi))) {
    *constraint = tnsr::a<DataType, SpatialDim, Frame>{get<0, 0>(pi)};
  }
  std::fill(constraint->begin(), constraint->end(), 0.0);

  f_constraint_add_term_1_of_25(constraint, spacetime_normal_one_form,
                                spacetime_normal_vector,
                                inverse_spacetime_metric, d_pi);
  f_constraint_add_term_2_of_25(constraint, inverse_spatial_metric, d_pi);
  f_constraint_add_term_3_of_25(constraint, inverse_spatial_metric,
                                spacetime_normal_vector, d_phi);
  f_constraint_add_term_4_of_25(constraint, spacetime_normal_one_form,
                                inverse_spacetime_metric,
                                inverse_spatial_metric, d_phi);
  f_constraint_add_term_5_of_25(constraint, spacetime_normal_one_form,
                                inverse_spatial_metric, d_gauge_function);
  f_constraint_add_term_6_of_25(
      constraint, spacetime_normal_one_form, spacetime_normal_vector, phi,
      inverse_spatial_metric, inverse_spacetime_metric);
  f_constraint_add_term_7_of_25(
      constraint, spacetime_normal_one_form, spacetime_normal_vector, phi,
      inverse_spatial_metric, inverse_spacetime_metric);
  f_constraint_add_term_8_of_25(constraint, spacetime_normal_one_form,
                                spacetime_normal_vector, d_gauge_function);
  f_constraint_add_term_9_of_25(constraint, inverse_spatial_metric, phi,
                                inverse_spacetime_metric,
                                spacetime_normal_vector);
  f_constraint_add_term_10_of_25(constraint, spacetime_normal_one_form,
                                 inverse_spatial_metric, phi,
                                 inverse_spacetime_metric);
  f_constraint_add_term_11_of_25(constraint, spacetime_normal_one_form,
                                 inverse_spatial_metric, phi,
                                 inverse_spacetime_metric);
  f_constraint_add_term_12_of_25(constraint, spacetime_normal_one_form, pi,
                                 inverse_spacetime_metric);
  f_constraint_add_term_13_of_25(constraint, inverse_spatial_metric,
                                 gauge_function, pi);
  f_constraint_add_term_14_of_25(constraint, spacetime_normal_vector,
                                 inverse_spatial_metric, pi);
  f_constraint_add_term_15_of_25(constraint, inverse_spacetime_metric,
                                 spacetime_normal_one_form,
                                 spacetime_normal_vector, phi, pi);
  f_constraint_add_term_16_of_25(constraint, spacetime_normal_one_form, pi,
                                 inverse_spacetime_metric,
                                 spacetime_normal_vector);
  f_constraint_add_term_17_of_25(constraint, inverse_spacetime_metric,
                                 spacetime_normal_one_form,
                                 spacetime_normal_vector, phi, pi);
  f_constraint_add_term_18_of_25(constraint, inverse_spatial_metric, phi,
                                 spacetime_normal_vector, pi);
  f_constraint_add_term_19_of_25(constraint, inverse_spatial_metric, phi,
                                 spacetime_normal_vector, pi);
  f_constraint_add_term_20_of_25(constraint, inverse_spatial_metric,
                                 gauge_function, phi, spacetime_normal_vector);
  f_constraint_add_term_21_of_25(constraint, spacetime_normal_one_form,
                                 spacetime_normal_vector, phi, gauge_function,
                                 inverse_spacetime_metric);
  f_constraint_add_term_22_of_25(
      constraint, gamma2, inverse_spacetime_metric, spacetime_normal_vector,
      three_index_constraint, spacetime_normal_one_form);
  f_constraint_add_term_23_of_25(constraint, spacetime_normal_one_form, pi,
                                 inverse_spacetime_metric, gauge_function,
                                 spacetime_normal_vector);
  f_constraint_add_term_24_of_25(constraint, spacetime_normal_one_form,
                                 inverse_spatial_metric, phi, gauge_function,
                                 inverse_spacetime_metric);
  f_constraint_add_term_25_of_25(constraint, spacetime_normal_one_form,
                                 inverse_spatial_metric, gauge_function, phi,
                                 inverse_spacetime_metric);
}

template <size_t SpatialDim, typename Frame, typename DataType>
Scalar<DataType> constraint_energy(
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& f_constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& two_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& four_index_constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& spatial_metric_determinant,
    double gauge_constraint_multiplier, double two_index_constraint_multiplier,
    double three_index_constraint_multiplier,
    double four_index_constraint_multiplier) noexcept {
  auto energy = make_with_value<Scalar<DataType>>(gauge_constraint, 0.0);
  constraint_energy<SpatialDim, Frame, DataType>(
      &energy, gauge_constraint, f_constraint, two_index_constraint,
      three_index_constraint, four_index_constraint, inverse_spatial_metric,
      spatial_metric_determinant, gauge_constraint_multiplier,
      two_index_constraint_multiplier, three_index_constraint_multiplier,
      four_index_constraint_multiplier);
  return energy;
}

template <size_t SpatialDim, typename Frame, typename DataType>
void constraint_energy(
    gsl::not_null<Scalar<DataType>*> energy,
    const tnsr::a<DataType, SpatialDim, Frame>& gauge_constraint,
    const tnsr::a<DataType, SpatialDim, Frame>& f_constraint,
    const tnsr::ia<DataType, SpatialDim, Frame>& two_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& three_index_constraint,
    const tnsr::iaa<DataType, SpatialDim, Frame>& four_index_constraint,
    const tnsr::II<DataType, SpatialDim, Frame>& inverse_spatial_metric,
    const Scalar<DataType>& spatial_metric_determinant,
    double gauge_constraint_multiplier, double two_index_constraint_multiplier,
    double three_index_constraint_multiplier,
    double four_index_constraint_multiplier) noexcept {
  if (get_size(get(*energy)) != get_size(get<0>(f_constraint))) {
    *energy = Scalar<DataType>{get_size(get<0>(f_constraint))};
  }

  get(*energy) = gauge_constraint_multiplier * square(gauge_constraint.get(0)) +
                 two_index_constraint_multiplier * square(f_constraint.get(0));
  for (size_t a = 1; a < SpatialDim + 1; ++a) {
    get(*energy) +=
        gauge_constraint_multiplier * square(gauge_constraint.get(a)) +
        two_index_constraint_multiplier * square(f_constraint.get(a));
  }

  for (size_t a = 0; a < SpatialDim + 1; ++a) {
    for (size_t i = 0; i < SpatialDim; ++i) {
      for (size_t j = 0; j < SpatialDim; ++j) {
        get(*energy) +=
            two_index_constraint_multiplier * two_index_constraint.get(i, a) *
            two_index_constraint.get(j, a) * inverse_spatial_metric.get(i, j);

        for (size_t b = 0; b < SpatialDim + 1; ++b) {
          get(*energy) += three_index_constraint_multiplier *
                              three_index_constraint.get(i, a, b) *
                              three_index_constraint.get(j, a, b) *
                              inverse_spatial_metric.get(i, j) +
                          2.0 * four_index_constraint_multiplier *
                              get(spatial_metric_determinant) *
                              four_index_constraint.get(i, a, b) *
                              four_index_constraint.get(j, a, b) *
                              inverse_spatial_metric.get(i, j);
        }
      }
    }
  }
}
}  // namespace GeneralizedHarmonic

// Explicit Instantiations
/// \cond
#define DIM(data) BOOST_PP_TUPLE_ELEM(0, data)
#define DTYPE(data) BOOST_PP_TUPLE_ELEM(1, data)
#define FRAME(data) BOOST_PP_TUPLE_ELEM(2, data)

#define INSTANTIATE(_, data)                                                 \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                    \
  GeneralizedHarmonic::three_index_constraint(                               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          d_spacetime_metric,                                                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;   \
  template void GeneralizedHarmonic::three_index_constraint(                 \
      gsl::not_null<tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>*>         \
          constraint,                                                        \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          d_spacetime_metric,                                                \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;   \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::gauge_constraint(                                     \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,    \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_one_form,                                         \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_vector,                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                            \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;   \
  template void GeneralizedHarmonic::gauge_constraint(                       \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>     \
          constraint,                                                        \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,    \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_one_form,                                         \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_vector,                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                            \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi) noexcept;   \
  template tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>                     \
  GeneralizedHarmonic::two_index_constraint(                                 \
      const tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>& d_gauge_function, \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_one_form,                                         \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_vector,                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                            \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,            \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,          \
      const Scalar<DTYPE(data)>& gamma2,                                     \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          three_index_constraint) noexcept;                                  \
  template void GeneralizedHarmonic::two_index_constraint(                   \
      const gsl::not_null<tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>*>    \
          constraint,                                                        \
      const tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>& d_gauge_function, \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_one_form,                                         \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_vector,                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                            \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,            \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,          \
      const Scalar<DTYPE(data)>& gamma2,                                     \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          three_index_constraint) noexcept;                                  \
  template tnsr::a<DTYPE(data), DIM(data), FRAME(data)>                      \
  GeneralizedHarmonic::f_constraint(                                         \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,    \
      const tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>& d_gauge_function, \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_one_form,                                         \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_vector,                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                            \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,            \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,          \
      const Scalar<DTYPE(data)>& gamma2,                                     \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          three_index_constraint) noexcept;                                  \
  template void GeneralizedHarmonic::f_constraint(                           \
      const gsl::not_null<tnsr::a<DTYPE(data), DIM(data), FRAME(data)>*>     \
          constraint,                                                        \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_function,    \
      const tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>& d_gauge_function, \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_one_form,                                         \
      const tnsr::A<DTYPE(data), DIM(data), FRAME(data)>&                    \
          spacetime_normal_vector,                                           \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                            \
      const tnsr::AA<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spacetime_metric,                                          \
      const tnsr::aa<DTYPE(data), DIM(data), FRAME(data)>& pi,               \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& phi,             \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>& d_pi,            \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi,          \
      const Scalar<DTYPE(data)>& gamma2,                                     \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          three_index_constraint) noexcept;                                  \
  template Scalar<DTYPE(data)> GeneralizedHarmonic::constraint_energy(       \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_constraint,  \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& f_constraint,      \
      const tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>&                   \
          two_index_constraint,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          three_index_constraint,                                            \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          four_index_constraint,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                            \
      const Scalar<DTYPE(data)>& spatial_metric_determinant,                 \
      double gauge_constraint_multiplier,                                    \
      double two_index_constraint_multiplier,                                \
      double three_index_constraint_multiplier,                              \
      double four_index_constraint_multiplier) noexcept;                     \
  template void GeneralizedHarmonic::constraint_energy(                      \
      gsl::not_null<Scalar<DTYPE(data)>*> energy,                            \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& gauge_constraint,  \
      const tnsr::a<DTYPE(data), DIM(data), FRAME(data)>& f_constraint,      \
      const tnsr::ia<DTYPE(data), DIM(data), FRAME(data)>&                   \
          two_index_constraint,                                              \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          three_index_constraint,                                            \
      const tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>&                  \
          four_index_constraint,                                             \
      const tnsr::II<DTYPE(data), DIM(data), FRAME(data)>&                   \
          inverse_spatial_metric,                                            \
      const Scalar<DTYPE(data)>& spatial_metric_determinant,                 \
      double gauge_constraint_multiplier,                                    \
      double two_index_constraint_multiplier,                                \
      double three_index_constraint_multiplier,                              \
      double four_index_constraint_multiplier) noexcept;

#define INSTANTIATE_ONLY_3D(_, data)                                          \
  template tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>                     \
  GeneralizedHarmonic::four_index_constraint(                                 \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi) noexcept; \
  template void GeneralizedHarmonic::four_index_constraint(                   \
      const gsl::not_null<tnsr::iaa<DTYPE(data), DIM(data), FRAME(data)>*>    \
          constraint,                                                         \
      const tnsr::ijaa<DTYPE(data), DIM(data), FRAME(data)>& d_phi) noexcept;

GENERATE_INSTANTIATIONS(INSTANTIATE, (1, 2, 3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))
GENERATE_INSTANTIATIONS(INSTANTIATE_ONLY_3D, (3), (double, DataVector),
                        (Frame::Grid, Frame::Inertial))
#undef DIM
#undef DTYPE
#undef FRAME
#undef INSTANTIATE
/// \endcond
