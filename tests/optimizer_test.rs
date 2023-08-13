#![cfg_attr(
    all(feature = "test-integrations", feature = "nightly"),
    feature(generic_const_exprs)
)]
use dfdx::{
    prelude::{Bias2D, Conv2D, MaxPool2D, ReLU, Sigmoid, Upscale2DBy},
    tensor_ops::NearestNeighbor,
};

type Model = (
    (Conv2D<3, 10, 3, 1, 1>, Bias2D<10>, ReLU, MaxPool2D<2, 2>),
    (Conv2D<10, 5, 3, 1, 1>, Bias2D<5>, ReLU, MaxPool2D<2, 2>),
    (
        Upscale2DBy<2, 2, NearestNeighbor>,
        Conv2D<5, 10, 3, 1, 1>,
        Bias2D<10>,
        ReLU,
    ),
    (
        Upscale2DBy<2, 2, NearestNeighbor>,
        Conv2D<10, 3, 3, 1, 1>,
        Bias2D<3>,
        ReLU,
    ),
);

#[test]
#[cfg(all(feature = "test-integrations", feature = "nightly"))]
fn test_optim() {
    use dfdx::{
        optim::Adam,
        prelude::*,
        tensor::{Cuda, Gradients},
        tensor_ops::AdamConfig,
    };
    let dev: Cuda = Default::default();

    let mut model = dev.build_module::<Model, f32>();

    let mut optim: Adam<_, f32, Cuda> = Adam::new(
        &model,
        AdamConfig {
            ..Default::default()
        },
    );

    type SI = Rank4<10, 3, 20, 20>;
    type SO = SI; // Rank4<10, 5, 5, 5>;
    fn assert_close<T1, T2>(
        t1: &Tensor<SO, f32, Cuda, T1>,
        t2: &Tensor<SO, f32, Cuda, T2>,
        msg: &'static str,
    ) {
        for (t1c, t2c) in t1.array().iter().zip(t2.array().iter()) {
            for (t1p, t2p) in t1c.iter().zip(t2c.iter()) {
                for (t1r, t2r) in t1p.iter().zip(t2p.iter()) {
                    for (t1v, t2v) in t1r.iter().zip(t2r.iter()) {
                        assert!(
                            (t1v - t2v).abs() < 1e-5,
                            "Differing values, {} != {} in {}",
                            t1v,
                            t2v,
                            msg
                        );
                    }
                }
            }
        }
    }

    model.load("./tests/model.npz").unwrap();

    let mut x: Tensor<SI, f32, _> = dev.zeros();
    x.load_from_npy("./tests/optim_x.npy").unwrap();
    let mut y0: Tensor<SO, f32, _> = dev.zeros();
    y0.load_from_npy("./tests/optim_y0.npy").unwrap();
    let mut target: Tensor<SO, f32, _> = dev.zeros();
    target.load_from_npy("./tests/optim_target.npy").unwrap();
    let mut y1: Tensor<SO, f32, _> = dev.zeros();
    y1.load_from_npy("./tests/optim_y1.npy").unwrap();

    let mut grad = model.alloc_grads();

    for i in 0..10 {
        let model_y = model.forward(x.clone().traced(grad));
        if i == 0 {
            assert_close(&y0, &model_y, "initial outputs differ");
        }
        let loss = mse_loss(model_y, target.clone());
        grad = loss.backward();
        optim.update(&mut model, &grad).unwrap();
        model.zero_grads(&mut grad);
    }

    let model_y1 = model.forward(x);
    assert_close(&model_y1, &y1, "outputs after update differ");
}
