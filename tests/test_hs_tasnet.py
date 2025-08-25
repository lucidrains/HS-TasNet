from pathlib import Path

import pytest
param = pytest.mark.parametrize

import torch

@param('small', (False, True))
@param('stereo', (False, True))
@param('use_gru', (False, True))
@param('var_audio_lens', (False, True))
@param('spec_branch_use_phase', (False, True))
@param('torch_compile', (False, True))
def test_model(
    small,
    stereo,
    use_gru,
    var_audio_lens,
    spec_branch_use_phase,
    torch_compile
):
    from hs_tasnet.hs_tasnet import HSTasNet

    model = HSTasNet(
        dim = 512,
        small = small,
        stereo = stereo,
        use_gru = use_gru,
        spec_branch_use_phase = spec_branch_use_phase,
        torch_compile = torch_compile
    )

    shape = (2, 1017 * 12) if stereo else (1017 * 12,)

    audio = torch.randn(3, *shape)
    targets = torch.rand(3, 4, *shape)

    audio_lens = torch.randint(1017 * 2, 1017 * 12, (3,)) if var_audio_lens else None

    loss = model(audio, targets = targets, audio_lens = audio_lens)
    loss.backward()

    # after much training

    chunk = torch.randn(shape)[..., :512].numpy()

    fn = model.init_stateful_transform_fn(
        return_reduced_sources = [0, 2] # say we only want drum and vocals, filtering out the bass and other
    )

    out1 = fn(chunk)
    out2 = fn(chunk)
    out3 = fn(chunk)

    prec_shape = (2,) if stereo else ()
    assert out3.shape == (*prec_shape, 512)

@param('with_eval', (False, True))
@param('with_ema', (False, True))
def test_trainer(
    with_eval,
    with_ema
):
    from hs_tasnet.hs_tasnet import HSTasNet
    from hs_tasnet.trainer import Trainer

    from torch.utils.data import Dataset

    model = HSTasNet(small = True, stereo = False)

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
        cpu = True,
        use_ema = with_ema
    )

    trainer.clear_folders()

    trainer()

    if with_eval:
        eval_folder = Path('./eval-results/1')
        assert eval_folder.exists() and eval_folder.is_dir()

def test_audio_processing():
    import sounddevice as sd

    device_list = sd.query_devices()
    num_devices = len(device_list)

    if num_devices == 0:
        pytest.skip()
        return

    # test sound device stream

    from hs_tasnet.hs_tasnet import HSTasNet

    model = HSTasNet(
        dim = 512,
        small = True,
    )

    model.save('./trained-model.pt')

    model.load('./trained-model.pt')

    model.sounddevice_stream(
        duration_seconds = 2,           # transform the audio with this given neural network coming into the microphone for 2 seconds
        return_reduced_sources = [0, 2] # say we only want drum and vocals, filtering out the bass and other
    )

    model.process_audio_file('./tests/test.mp3', [0, 2], overwrite = True)

def test_save_spectrogram_fig():
    from hs_tasnet.hs_tasnet import HSTasNet

    model = HSTasNet(
        dim = 512,
        stereo = False
    )

    audio = torch.randn(41_000 * 5)

    model.save_spectrogram_figure('./spec.png', audio, overwrite = True)

@param('stereo', (False, True))
def test_musdb(stereo):
    import musdb
    from hs_tasnet.hs_tasnet import HSTasNet
    from hs_tasnet.trainer import Trainer

    mus = musdb.DB(download = True)

    model = HSTasNet(stereo = stereo)

    trainer = Trainer(model, dataset = mus, batch_size = 4, cpu = True, max_steps = 2)
    trainer()
