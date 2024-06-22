import yaml
import os

def convert_environment_to_requirements(environment_file):
    with open(environment_file) as file_handle:
        environment_data = yaml.safe_load(file_handle)

    with open(os.path.join('..', 'requirements.txt'), "w") as requirements_file:
        for dependency in environment_data.get("dependencies", []):
            package_name, package_version = dependency.split("=")
            requirements_file.write(f"{package_name}=={package_version}\n")


convert_environment_to_requirements(os.path.join('..', "environment.yml"))
