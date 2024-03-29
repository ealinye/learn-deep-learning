use burn::backend::{candle::CandleDevice, Autodiff, Candle};
use burn::config::Config;
use burn::data::dataloader::batcher::Batcher;
use burn::data::dataloader::DataLoaderBuilder;
use burn::data::dataset::vision::{MNISTDataset, MNISTItem};
use burn::module::Module;
use burn::nn::loss::CrossEntropyLossConfig;
use burn::nn::{Linear, LinearConfig, ReLU};
use burn::optim::SgdConfig;
use burn::record::CompactRecorder;
use burn::record::Recorder;
use burn::tensor::backend::{AutodiffBackend, Backend};
use burn::tensor::{Data, ElementConversion, Int, Tensor};
use burn::train::metric::{AccuracyMetric, LossMetric};
use burn::train::LearnerBuilder;
use burn::train::{ClassificationOutput, TrainOutput, TrainStep, ValidStep};

fn main() {
    let artifact_dir = "./train";
    let device = CandleDevice::Cuda(0);
    let config = Config::load(format!("{artifact_dir}/config.json")).unwrap_or_else(|_e| {
        std::fs::create_dir(artifact_dir).ok();
        let config = TrainingConfig::new(ModelConfig::new(), SgdConfig::new());
        config
            .save(format!("{artifact_dir}/config.json"))
            .expect("???");
        config
    });
    train::<Autodiff<Candle>>("./train", config, &device);
}

#[derive(Debug, Clone, Copy, Default)]
struct MNISTBatcher<B: Backend> {
    device: B::Device,
}

impl<B: Backend> MNISTBatcher<B> {
    fn new(device: B::Device) -> Self {
        Self { device }
    }
}

#[derive(Debug, Clone)]
struct MNISTBatch<B: Backend> {
    pub images: Tensor<B, 3>,
    pub targets: Tensor<B, 1, Int>,
}

impl<B: Backend> Batcher<MNISTItem, MNISTBatch<B>> for MNISTBatcher<B> {
    fn batch(&self, items: Vec<MNISTItem>) -> MNISTBatch<B> {
        let images = items
            .iter()
            .map(|data| Data::<f32, 2>::from(data.image).convert())
            .map(|data| Tensor::<B, 2>::from_data(data, &self.device))
            .map(|tensor| ((tensor.reshape([1, 28, 28]) / 255) - 0.1307) / 0.3081)
            .collect::<Vec<_>>();
        let targets = items
            .iter()
            .map(|target| {
                Tensor::<B, 1, Int>::from_data(
                    Data::from([(target.label as i64).elem()]),
                    &self.device,
                )
            })
            .collect::<Vec<_>>();
        MNISTBatch {
            images: Tensor::cat(images, 0),
            targets: Tensor::cat(targets, 0),
        }
    }
}

#[derive(Module, Debug)]
struct Model<B: Backend> {
    linear1: Linear<B>,
    relu: ReLU,
    linear2: Linear<B>,
}

impl<B: Backend> Model<B> {
    pub fn forward(&self, images: Tensor<B, 3>) -> Tensor<B, 2> {
        let [batch_size, width, height] = images.dims();
        let images = images.reshape([batch_size, width * height]);

        let x = self.linear1.forward(images);
        let x = self.relu.forward(x);
        self.linear2.forward(x)
    }
    pub fn forward_classification(
        &self,
        images: Tensor<B, 3>,
        targets: Tensor<B, 1, Int>,
    ) -> ClassificationOutput<B> {
        let output = self.forward(images);
        let loss = CrossEntropyLossConfig::new()
            .init(&output.device())
            .forward(output.clone(), targets.clone());

        ClassificationOutput::new(loss, output, targets)
    }
}

impl<B: AutodiffBackend> TrainStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> burn::train::TrainOutput<ClassificationOutput<B>> {
        let item = self.forward_classification(batch.images, batch.targets);
        TrainOutput::new::<B, Model<B>>(self, item.loss.backward(), item)
    }
}

impl<B: Backend> ValidStep<MNISTBatch<B>, ClassificationOutput<B>> for Model<B> {
    fn step(&self, batch: MNISTBatch<B>) -> ClassificationOutput<B> {
        self.forward_classification(batch.images, batch.targets)
    }
}

#[derive(Config)]
struct ModelConfig {
    #[config(default = 784 /* 28 * 28 */)]
    num_inputs: usize,
    #[config(default = 10)]
    num_classes: usize,
    #[config(default = 16)]
    hidden_size: usize,
}

impl ModelConfig {
    pub fn init<B: Backend>(&self, device: &B::Device) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.num_inputs, self.hidden_size).init(device),
            relu: ReLU::new(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes).init(device),
        }
    }

    pub fn init_with<B: Backend>(&self, record: ModelRecord<B>) -> Model<B> {
        Model {
            linear1: LinearConfig::new(self.num_inputs, self.hidden_size).init_with(record.linear1),
            relu: ReLU::new(),
            linear2: LinearConfig::new(self.hidden_size, self.num_classes)
                .init_with(record.linear2),
        }
    }
}

#[derive(Config)]
struct TrainingConfig {
    pub model: ModelConfig,
    pub optimizer: SgdConfig,
    #[config(default = 10)]
    pub num_epochs: usize,
    #[config(default = 500)]
    pub batch_size: usize,
    #[config(default = 8)]
    pub num_workers: usize,
    #[config(default = 42)]
    pub seed: u64,
    #[config(default = 1.0e-4)]
    pub learning_rate: f64,
}

fn train<B: AutodiffBackend>(artifact_dir: &str, config: TrainingConfig, device: &B::Device) {
    // B::seed(config.seed);

    let batcher_train = MNISTBatcher::<B>::new(device.clone());
    let batcher_valid = MNISTBatcher::<B::InnerBackend>::new(device.clone());

    let dataloader_train = DataLoaderBuilder::new(batcher_train)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::train());
    let dataloader_test = DataLoaderBuilder::new(batcher_valid)
        .batch_size(config.batch_size)
        .shuffle(config.seed)
        .num_workers(config.num_workers)
        .build(MNISTDataset::test());

    let model = CompactRecorder::new()
        .load(format!("{artifact_dir}/model").into(), device)
        .map(|record| config.model.init_with::<B>(record))
        .unwrap_or_else(|_e| config.model.init(device));

    let model = LearnerBuilder::new("./train")
        .metric_train_numeric(AccuracyMetric::new())
        .metric_valid_numeric(AccuracyMetric::new())
        .metric_train_numeric(LossMetric::new())
        .metric_valid_numeric(LossMetric::new())
        .with_file_checkpointer(CompactRecorder::new())
        .devices(vec![device.clone()])
        .num_epochs(config.num_epochs)
        .build(model, config.optimizer.init(), config.learning_rate)
        .fit(dataloader_train, dataloader_test);

    model
        .save_file(format!("{artifact_dir}/model"), &CompactRecorder::new())
        .unwrap();
}
