import pytest
param = pytest.mark.parametrize

import torch

@param('small', (False, True))
@param('stereo', (False, True))
@param('use_gru', (False, True))
def test_model(
    small,
    stereo,
    use_gru
):
    from hs_tasnet.hs_tasnet import HSTasNet

    model = HSTasNet(
        dim = 512,
        small = small,
        stereo = stereo,
        use_gru = use_gru
    )

    shape = (2, 1024 * 12) if stereo else (1024 * 12,)

    audio = torch.randn(3, *shape)
    targets = torch.rand(3, 4, *shape)

    loss = model(audio, targets = targets)
    loss.backward()

    # after much training

    chunk = torch.randn(shape)[..., :512].numpy()

    fn = model.init_stream_fn(
        return_reduced_sources = [0, 2] # say we only want drum and vocals, filtering out the bass and other
    )

    out1 = fn(chunk)
    out2 = fn(chunk)
    out3 = fn(chunk)

    prec_shape = (2,) if stereo else ()
    assert out3.shape == (*prec_shape, 512)

@param('with_eval', (False, True))
def test_trainer(
    with_eval
):
    from hs_tasnet.hs_tasnet import HSTasNet
    from hs_tasnet.trainer import Trainer

    from torch.utils.data import Dataset

    model = HSTasNet(small = True)

    class MusicSepDataset(Dataset):
        def __len__(self):
            return 20

        def __getitem__(self, idx):
            audio = torch.randn(1024 * 10)
            targets = torch.rand(4, 1024 * 10)
            return audio, targets

    class EvalMusicSepDataset(Dataset):
        def __len__(self):
            return 5

        def __getitem__(self, idx):
            audio = torch.randn(1024 * 10)
            targets = torch.rand(4, 1024 * 10)
            return audio, targets

    trainer = Trainer(
        model,
        dataset = MusicSepDataset(),
        eval_dataset = EvalMusicSepDataset() if with_eval else None,
        batch_size = 4,
        max_epochs = 3,
        checkpoint_every = 1,
        cpu = True
    )

    trainer()
