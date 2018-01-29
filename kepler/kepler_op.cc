#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor_shape.h"
#include "tensorflow/core/platform/default/logging.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/util/work_sharder.h"

#include <cmath>

using namespace tensorflow;


#define KEPLER_MAX_ITER 200
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
    shape_inference::ShapeHandle M, e;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &M));
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
    const auto e = e_tensor.template flat<T>();
    auto E = E_tensor->template flat<T>();

    for (int64 n = 0; n < N; ++n) {
      E(n) = kepler<T>(M(n), e(0));
    }
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("Kepler").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      KeplerOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL

// Reverse op

REGISTER_OP("KeplerGrad")
  .Attr("T: {float, double}")
  .Input("manom: T")
  .Input("eccen: T")
  .Input("eanom: T")
  .Input("beanom: T")
  .Output("bmanom: T")
  .Output("beccen: T")
  .SetShapeFn([](shape_inference::InferenceContext* c) {
    shape_inference::ShapeHandle M, e, E, bE;
    TF_RETURN_IF_ERROR(c->WithRank(c->input(0), 1, &M));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &e));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 1, &E));
    TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &bE));
    TF_RETURN_IF_ERROR(c->Merge(M, E, &M));
    TF_RETURN_IF_ERROR(c->Merge(E, bE, &E));
    c->set_output(0, c->input(0));
    c->set_output(1, c->input(1));
    return Status::OK();
  });

template <typename T>
class KeplerGradOp : public OpKernel {
 public:
  explicit KeplerGradOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Inputs
    const Tensor& M_tensor = context->input(0);
    const Tensor& e_tensor = context->input(1);
    const Tensor& E_tensor = context->input(2);
    const Tensor& bE_tensor = context->input(3);

    // Dimensions
    const int64 N = M_tensor.NumElements();
    OP_REQUIRES(context, E_tensor.dim_size(0) == N, errors::InvalidArgument("dimension mismatch"));
    OP_REQUIRES(context, bE_tensor.dim_size(0) == N, errors::InvalidArgument("dimension mismatch"));

    // Output
    Tensor* bM_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, M_tensor.shape(), &bM_tensor));
    Tensor* be_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(1, e_tensor.shape(), &be_tensor));

    // Access the data
    const auto e = e_tensor.template flat<T>();
    const auto E = E_tensor.template flat<T>();
    const auto bE = bE_tensor.template flat<T>();
    auto bM = bM_tensor->template flat<T>();
    auto be = be_tensor->template flat<T>();

    T e_value = e(0), be_value = T(0.0);
    for (int64 n = 0; n < N; ++n) {
      bM(n) = bE(n) / (T(1.0) - e_value * cos(E(n)));
      be_value += sin(E(n)) * bM(n);
    }
    be(0) = be_value;
  }
};


#define REGISTER_KERNEL(type)                                              \
  REGISTER_KERNEL_BUILDER(                                                 \
      Name("KeplerGrad").Device(DEVICE_CPU).TypeConstraint<type>("T"), \
      KeplerGradOp<type>)

REGISTER_KERNEL(float);
REGISTER_KERNEL(double);

#undef REGISTER_KERNEL
