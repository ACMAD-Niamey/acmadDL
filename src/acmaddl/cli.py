"""acmadDL command-line interface."""
import subprocess
import sys
import click


@click.group()
def cli():
    """acmadDL — climate data integration CLI."""


@cli.command("datasets")
@click.argument("output", required=False, type=click.Path())
def datasets(output):
    """Render the one-page dataset catalogue (needs matplotlib).

    With OUTPUT, write the figure there; otherwise open a window.
    """
    try:
        from .gallery import show_datasets
        import matplotlib
    except ImportError as e:
        raise click.ClickException(
            f"rendering needs matplotlib (pip install 'acmadDL[demo]'): {e}")
    if output:
        show_datasets(save=output)
        click.echo(f"wrote {output}")
    else:
        import matplotlib.pyplot as plt
        show_datasets()
        plt.show()


@cli.group()
def cache():
    """Inspect and manage the local Nuthatch cache."""


@cache.command("list")
def cache_list():
    """List all cached acmadDL entries."""
    subprocess.run([sys.executable, "-m", "nuthatch", "list", "--namespace", "acmaddl"],
                   check=False)


@cache.command("clear")
@click.option("--product", default=None,
              help="acmadDL product name to clear (e.g. nmme/cfsv2). Omit to clear all.")
@click.option("--yes", is_flag=True, help="Skip confirmation prompt.")
def cache_clear(product, yes):
    """Remove cached acmadDL entries."""
    label = f"product '{product}'" if product else "ALL acmadDL cache entries"
    if not yes:
        click.confirm(f"This will delete {label}. Continue?", abort=True)
    cmd = [sys.executable, "-m", "nuthatch", "delete", "--namespace", "acmaddl", "--force"]
    if product:
        cmd += ["--cache-key", product]
    subprocess.run(cmd, check=False)
