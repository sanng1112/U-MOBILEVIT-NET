import sys
import torch
from pathlib import Path
_current_dir = Path(__file__).resolve().parent.parent.parent
if str(_current_dir) not in sys.path:
    sys.path.insert(0, str(_current_dir))

try:
    import click
except ImportError:
    print("pip install click"); sys.exit(1)

from cv_nets.pipeline import load_config, ModelBuilder, UnifiedTrainer, Evaluator, SpectralAnalyzer, AblationController, validate_config
from cv_nets.pipeline.registry import BLOCK_REGISTRY
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, random_split


@click.group()
def cli():
    pass


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--device", default="auto")
def train(config_path, device):
    cfg = load_config(config_path)
    errors = validate_config(cfg)
    if errors:
        click.echo("Errors:" + "\n".join(f"  - {e}" for e in errors))
        sys.exit(1)
    model = ModelBuilder(cfg.model).build()
    click.echo(f"Model: {cfg.model.name} | Params: {sum(p.numel() for p in model.parameters()):,}")

    t = transforms.Compose([transforms.ToTensor()])
    if cfg.dataset.name == "mnist":
        full = datasets.MNIST(root=cfg.dataset.root, train=True, transform=t, download=True)
        vs = int(len(full) * cfg.dataset.val_split)
        train_ds, val_ds = random_split(full, [len(full)-vs, vs])
        test_ds = datasets.MNIST(root=cfg.dataset.root, train=False, transform=t, download=True)
    else:
        full = datasets.ImageFolder(root=f"{cfg.dataset.root}/train", transform=t)
        vs = int(len(full) * cfg.dataset.val_split)
        train_ds, val_ds = random_split(full, [len(full)-vs, vs])
        test_ds = datasets.ImageFolder(root=f"{cfg.dataset.root}/test", transform=t)

    train_loader = DataLoader(train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=cfg.dataset.num_workers)
    val_loader = DataLoader(val_ds, batch_size=cfg.training.batch_size, num_workers=cfg.dataset.num_workers)
    test_loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, num_workers=cfg.dataset.num_workers)

    trainer = UnifiedTrainer(model, cfg.training, cfg.dataset, device=device)
    trainer.fit(train_loader, val_loader)

    ev = Evaluator(model, device=device)
    metrics = ev.evaluate(test_loader, num_classes=cfg.dataset.num_classes)
    click.echo(f"Test Accuracy: {metrics['accuracy']:.4f}")

    if cfg.research.enabled:
        analyzer = SpectralAnalyzer(every_n_epochs=cfg.research.visualize_every)
        dummy = torch.randn(2, *cfg.model.input_size)
        analyzer.analyze(model, dummy, epoch=0)
        click.echo(analyzer.summary())


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--checkpoint", default=None)
@click.option("--device", default="auto")
def eval(config_path, checkpoint, device):
    cfg = load_config(config_path)
    model = ModelBuilder(cfg.model).build()
    if checkpoint:
        ckpt = torch.load(checkpoint, map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["model_state_dict"])
    t = transforms.Compose([transforms.ToTensor()])
    if cfg.dataset.name == "mnist":
        test_ds = datasets.MNIST(root=cfg.dataset.root, train=False, transform=t, download=True)
    else:
        test_ds = datasets.ImageFolder(root=f"{cfg.dataset.root}/test", transform=t)
    loader = DataLoader(test_ds, batch_size=cfg.training.batch_size, num_workers=cfg.dataset.num_workers)
    ev = Evaluator(model, device=device)
    metrics = ev.evaluate(loader, num_classes=cfg.dataset.num_classes)
    click.echo(ev.report(metrics, num_classes=cfg.dataset.num_classes))


@cli.command()
@click.argument("config_path", type=click.Path(exists=True))
@click.option("--device", default="auto")
def research(config_path, device):
    cfg = load_config(config_path)
    model = ModelBuilder(cfg.model).build()
    click.echo("Running spectral analysis...")
    analyzer = SpectralAnalyzer(every_n_epochs=1)
    x = torch.randn(1, *cfg.model.input_size)
    analyzer.analyze(model, x, epoch=0)
    click.echo(analyzer.summary())
    click.echo("\nAblation keys:")
    for k in AblationController(model).list_keys():
        click.echo(f"  - {k}")


@cli.command()
def list_blocks():
    click.echo("Available blocks:")
    for name, cls in sorted(BLOCK_REGISTRY.items()):
        doc = (cls.__doc__ or "").strip().split("\n")[0] if cls.__doc__ else ""
        click.echo(f"  {name:30s} {doc}")


if __name__ == "__main__":
    cli()
