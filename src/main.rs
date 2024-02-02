use burn::backend::autodiff::grads::Gradients;
use burn::backend::{wgpu::WgpuDevice, Autodiff, Fusion, Wgpu};
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::{MNISTDataset, MNISTItem};
use burn::data::dataset::Dataset;
use burn::tensor::{Data, Distribution, ElementConversion, Float, Int, Tensor, TensorKind};

type Backend = Autodiff<Fusion<Wgpu>>;
type WgpuTensor<const D: usize = 2, K = Float> = Tensor<Backend, D, K>;

const BATCH_SIZE: usize = 512;
const DEVICE: WgpuDevice = WgpuDevice::BestAvailable;

const NUM_INPUTS: usize = 28 * 28;
const NUM_OUTPUTS: usize = 10;

fn main() {
    let mut w = WgpuTensor::random(
        [NUM_INPUTS, NUM_OUTPUTS],
        Distribution::Normal(0.0, 0.01),
        &DEVICE,
    )
    .require_grad();
    let mut b = WgpuTensor::zeros([1, NUM_OUTPUTS], &DEVICE).require_grad();
    let lr = 0.03;

    let dataloader_train = DataLoaderBuilder::new(MyBatcher)
        .batch_size(BATCH_SIZE)
        .build(MNISTDataset::train());

    for (counter, Batch { images, targets }) in dataloader_train.iter().enumerate() {
        let x = images;
        let y = targets;

        let y_hat = net(x.tclone(), w.tclone(), b.tclone());
        let l = cross_entropy(y.tclone(), y_hat.tclone());
        let grad = l.mean().backward();

        sdg([&mut w, &mut b], grad, lr);
        println!("第[{}\t]次的准确度：{:.8}", counter + 1, accuracy(y, y_hat));
    }

    {
        let Batch { images, targets } =
            MyBatcher.batch(MNISTDataset::test().iter().collect::<Vec<_>>());
        println!(
            "最终准确度：{:.8}",
            accuracy(targets, net(images, w.tclone(), b.tclone()))
        )
    }
}

fn softmax(x: WgpuTensor) -> WgpuTensor {
    let x_exp = x.tclone().exp();
    let partition = x_exp.tclone().sum_dim(1);
    x_exp.tclone() / partition
}

/// y : [num_inputs,1]
///
/// y_hat : [num_inputs,num_outputs]
fn cross_entropy(y: WgpuTensor<2, Int>, y_hat: WgpuTensor) -> WgpuTensor {
    let tensors = y
        .iter_dim(0)
        .zip(y_hat.iter_dim(0))
        .map(|(y, y_hat)| {
            let index = y.into_scalar() as usize;
            y_hat.slice([0..1, index..index + 1])
        })
        .collect();
    -WgpuTensor::cat(tensors, 0).log()
}

fn sdg<const C: usize>(params: [&mut WgpuTensor; C], mut grad: Gradients, lr: f64) {
    for param in params {
        let grad = param.grad_remove(&mut grad).unwrap();
        let delta_param = WgpuTensor::from_data(grad.into_data(), &DEVICE) * lr / 8;

        *param = param
            .tclone()
            .set_require_grad(false)
            .sub(delta_param)
            .require_grad();
    }
}

fn accuracy(y: WgpuTensor<2, Int>, y_hat: WgpuTensor) -> f32 {
    let y_hat = y_hat.argmax(1);
    y.tclone().equal(y_hat).float().sum().into_scalar() / y.dims()[0] as f32
}

fn net(x: WgpuTensor<3>, w: WgpuTensor, b: WgpuTensor) -> WgpuTensor {
    softmax(x.reshape([-1, NUM_INPUTS as i32]).matmul(w) + b)
}

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

#[derive(Debug, Clone, Copy)]
struct MyBatcher;

#[derive(Debug, Clone)]
struct Batch {
    pub images: WgpuTensor<3>,
    pub targets: WgpuTensor<2, Int>,
}

impl Batcher<MNISTItem, Batch> for MyBatcher {
    fn batch(&self, items: Vec<MNISTItem>) -> Batch {
        let images = items
            .iter()
            .map(|data| Data::<f32, 2>::from(data.image).convert())
            .map(|data| WgpuTensor::<2>::from_data(data, &DEVICE))
            .map(|tensor| ((tensor.reshape([1, 28, 28]) / 255) - 0.1307) / 0.3081)
            .collect::<Vec<_>>();
        let targets = items
            .iter()
            .map(|target| {
                WgpuTensor::<1, Int>::from_data(Data::from([(target.label as i64).elem()]), &DEVICE)
                    .reshape([1, 1])
            })
            .collect::<Vec<_>>();
        Batch {
            images: WgpuTensor::cat(images, 0),
            targets: WgpuTensor::cat(targets, 0),
        }
    }
}
