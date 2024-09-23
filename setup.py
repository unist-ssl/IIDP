import setuptools


def read_requirements(file_path):
    requirements = []
    with open(file_path) as f:
        for line in f:
            if "#" in line:
                line = line[:line.index("#")]
            line = line.strip()
            if line and not line.startswith("-"):
                requirements.append(line)
    return requirements


if __name__ == "__main__":
    setuptools.setup(
        name="iidp",
        version="1.0.0",
        author="The IIDP Authors & Samsung SDS",
        author_email="rugyoon@gmail.com",
        description="IIDP: Independent and Identical Data Parallelism",
        url="https://github.com/unist-ssl/IIDP",
        packages=setuptools.find_packages(include=["iidp", "iidp.*"]),
        python_requires='>=3.6',
        install_requires=read_requirements("requirements.txt")
    )