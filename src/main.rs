use burn::backend::autodiff::grads::Gradients;
use burn::backend::{wgpu::WgpuDevice, Autodiff, Fusion, Wgpu};

use burn::tensor::{Distribution, Float, Tensor, TensorKind};

type Backend = Autodiff<Fusion<Wgpu>>;

type WgpuTensor<const D: usize = 2, K = Float> = Tensor<Backend, D, K>;

const NUM_DATA: usize = 1024;
const BATCH_SIZE: usize = 128;
const DEVICE: WgpuDevice = WgpuDevice::BestAvailable;

trait TensorClone
where
    Self: Sized,
{
    fn tclone(&self) -> Self;
}

impl<B, const D: usize, K> TensorClone for Tensor<B, D, K>
where
    B: burn::tensor::backend::Backend,
    K: TensorKind<B>,
    Self: Sized,
{
    fn tclone(&self) -> Self {
        <Self as Clone>::clone(self)
    }
}

fn main() {
    let real_w = WgpuTensor::from_data([2.0, -3.4], &DEVICE).reshape([2, 1]);
    let real_b = WgpuTensor::from_data([[4.2]], &DEVICE);

    let (features, labels) = {
        let x = WgpuTensor::random(
            [NUM_DATA, real_w.dims()[0]],
            Distribution::Normal(0.0, 1.0),
            &DEVICE,
        );
        let y = x.tclone().matmul(real_w.tclone()) + real_b.tclone();
        let y = y.tclone() + y.random_like(Distribution::Normal(0.0, 0.01));

        (x, y.reshape([-1, 1]))
    };

    let mut next = {
        let mut counter = 0;
        let features = features.tclone();
        let labels = labels.tclone();
        move || {
            let r = (
                features.tclone().slice([counter..counter + 1, 0..2]),
                labels.tclone().slice([counter..counter + 1, 0..1]),
            );
            counter += 1;
            counter %= NUM_DATA;
            r
        }
    };

    let mut w = WgpuTensor::random([2, 1], Distribution::Normal(0.0, 0.01), &DEVICE).require_grad();
    let mut b = WgpuTensor::zeros([1, 1], &DEVICE).require_grad();
    let lr = 0.03;

    for counter in 0..128 {
        for (x, y) in (0..BATCH_SIZE).map(|_| next()) {
            let grad = squared_loss(y, linreg(x, w.tclone(), b.tclone()))
                .sum()
                .backward();
            sdg([&mut w, &mut b], grad, lr);
        }

        println!(
            "第{counter}次 : {}, w : {:?}, b : {}",
            squared_loss(
                labels.tclone(),
                linreg(features.tclone(), w.tclone(), b.tclone()),
            )
            .mean()
            .into_scalar(),
            w.tclone().into_data().value,
            b.tclone().into_scalar()
        )
    }
}

#[inline]
fn linreg(x: WgpuTensor, w: WgpuTensor, b: WgpuTensor) -> WgpuTensor {
    x.matmul(w) + b
}

#[inline]
fn squared_loss(y: WgpuTensor, y_hat: WgpuTensor) -> WgpuTensor {
    (y - y_hat).powi_scalar(2) / 2.0
}

#[inline]
fn sdg<const C: usize>(params: [&mut WgpuTensor; C], mut grad: Gradients, lr: f64) {
    for param in params {
        let delta_param =
            WgpuTensor::from_data(param.grad_remove(&mut grad).unwrap().into_data(), &DEVICE) * lr
                / 8;

        *param = param
            .tclone()
            .set_require_grad(false)
            .sub(delta_param)
            .require_grad();
    }
}
