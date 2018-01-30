#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

#include <cmath>

using namespace tensorflow;


#define KEPLER_MAX_ITER 2000
#define KEPLER_TOL      1.234e-10

// Kepler solver
template <typename T>
inline T kepler (const T& M, const T& e) {
  T E0 = M, E = M;
  for (int i = 0; i < KEPLER_MAX_ITER; ++i) {
    T g = E0 - e * sin(E0) - M, gp = 1.0 - e * cos(E0);
    E = E0 - g / gp;
    if (std::abs((E - E0) / E) <= T(KEPLER_TOL)) {
      return E;
    }
    E0 = E;
  }

  // If we get here, we didn't converge, but return the best estimate.
  return E;
}

#undef KEPLER_MAX_ITER
#undef KEPLER_TOL

// Forward op

REGISTER_OP("Kepler")
  .Attr("T: {float, double}")
  .Input("manom: T")
  .Input("eccen: T")
  .Output("eanom: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle e;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &e));
    c->set_output(0, c->input(0));
    return Status::OK();
  });

template <typename T>
class KeplerOp : public OpKernel {
 public:
  explicit KeplerOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& M_tensor = context->input(0);
    const Tensor& e_tensor = context->input(1);

    // Dimensions
    const int64 N = M_tensor.NumElements();

    // Output
    Tensor* E_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, M_tensor.shape(), &E_tensor));

    // Access the data
    const auto M = M_tensor.template flat<T>();
    const auto e = e_tensor.template scalar<T>()(0);
    auto E = E_tensor->template flat<T>();

    for (int64 n = 0; n < N; ++n) {
      E(n) = kepler<T>(M(n), e);
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Kepler").Device(DEVICE_CPU).TypeConstraint<type>("T"),         \
      KeplerOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
