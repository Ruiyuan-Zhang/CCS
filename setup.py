from setuptools import setup, find_packages

# requirements = [
#     'numpy', 'pyyaml', 'trimesh', 'wandb', 'torch', 'pytorch_lightning',
#     'tqdm', 'yacs', 'open3d', 'pytorch3d', 'einops'
# ]

requirements = []


def get_version():
    version_file = 'multi_part_assembly/version.py'
    with open(version_file, 'r') as f:
        exec(compile(f.read(), version_file, 'exec'))
    return locals()['__version__']


setup(
    name="multi_part_assembly",
    version=get_version(),
    description="3D Geometric Shape Assembly with Co-creation Space",
    long_description="Co-creation Space(CSS) is based on Multi-Part Shape Assembly. The corresponding paper, named 'Scalable Geometric Fracture Assembly via Co-creation Space among Assemblers', has been accepted by the AAAI 2024.",
    author="Ruiyuan",
    author_email="zhangruiyuan.0122@gmail.com",
    license="",
    url="",
    keywords="Co-creation Space",
    packages=find_packages(),
    install_requires=requirements,
)
