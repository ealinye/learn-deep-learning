use burn::backend::autodiff::grads::Gradients;
use burn::backend::{wgpu::WgpuDevice, Autodiff, Fusion, Wgpu};
use burn::tensor::{Distribution, Float, Tensor};

type Backend = Autodiff<Fusion<Wgpu>>;

type WgpuTensor<const D: usize = 2, K = Float> = Tensor<Backend, D, K>;

const NUM_DATA: usize = 1024;
const BATCH_SIZE: usize = 128;
const DEVICE: WgpuDevice = WgpuDevice::BestAvailable;

fn main() {
    let real_w = WgpuTensor::from_data([2.0, -3.4], &DEVICE).reshape([2, 1]);
    let real_b = WgpuTensor::from_data([[4.2]], &DEVICE);

    let (features, labels): (WgpuTensor, WgpuTensor) = {
        let x = WgpuTensor::random(
            [NUM_DATA, real_w.dims()[0]],
            Distribution::Normal(0.0, 1.0),
            &DEVICE,
        );
        let y = x.clone().matmul(real_w.clone()) + real_b.clone();
        let y = y.clone() + y.random_like(Distribution::Normal(0.0, 0.01));

        (x, y.reshape([-1, 1]))
    };

    let mut next = {
        let mut counter = 0;
        let features = features.clone();
        let labels = labels.clone();
        move || {
            let r = (
                features.clone().slice([counter..counter + 1, 0..2]),
                labels.clone().slice([counter..counter + 1, 0..1]),
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
            let grad = squared_loss(y, linreg(x, w.clone(), b.clone()))
                .sum()
                .backward();
            sdg([&mut w, &mut b], grad, lr);
        }

        println!(
            "第{counter}次 : {}, w : {:?}, b : {}",
            squared_loss(
                labels.clone(),
                linreg(features.clone(), w.clone(), b.clone()),
            )
            .mean()
            .into_scalar(),
            w.clone().into_data().value,
            b.clone().into_scalar()
        )
    }
}

#[inline]
fn linreg(x: WgpuTensor, w: WgpuTensor, b: WgpuTensor) -> WgpuTensor {
    x.matmul(w) + b
}

#[inline]
fn squared_loss(y: WgpuTensor, y_hat: WgpuTensor) -> WgpuTensor {
    (y - y_hat).powf(WgpuTensor::from_data([[2.0]], &WgpuDevice::BestAvailable)) / 2.0
}

#[inline]
fn sdg<const C: usize>(params: [&mut WgpuTensor; C], mut grad: Gradients, lr: f64) {
    for param in params {
        let delta_param =
            WgpuTensor::from_data(param.grad_remove(&mut grad).unwrap().into_data(), &DEVICE) * lr
                / 8;

        *param = param
            .clone()
            .set_require_grad(false)
            .sub(delta_param)
            .require_grad();
    }
}
