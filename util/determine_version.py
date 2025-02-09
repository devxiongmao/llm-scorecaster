import subprocess
import re

class AutoVersionBumper:
    VERSION_FILE = 'VERSION'

    @classmethod
    def bump_version(cls):
        current_version = cls.current_version_from_file() or '0.0.0'
        next_version = cls.calculate_next_version(current_version)

        if next_version:
            cls.update_version_file(next_version)
            print(f"Version bumped to: {next_version}")
        else:
            print("No changes detected; version remains the same.")

    @classmethod
    def current_version_from_file(cls):
        try:
            with open(cls.VERSION_FILE, 'r') as file:
                return file.read().strip()
        except FileNotFoundError:
            return None

    @classmethod
    def calculate_next_version(cls, current_version):
        commits = cls.commits_since_last_version(current_version)
        if not commits:
            return None

        major, minor, patch = map(int, current_version.split('.'))

        if any(re.search(r'\b(breaking|major)\b', msg, re.IGNORECASE) for msg in commits):
            major += 1
            minor = 0
            patch = 0
        elif any(re.search(r'\b(feature|minor)\b', msg, re.IGNORECASE) for msg in commits):
            minor += 1
            patch = 0
        else:
            patch += 1

        return f"{major}.{minor}.{patch}"

    @classmethod
    def commits_since_last_version(cls, current_version):
        tag = f"v{current_version}"
        command = ["git", "log", f"{tag}..HEAD", "--pretty=format:%s"]
        result = subprocess.run(command, capture_output=True, text=True)

        if result.returncode != 0:
            return []

        return result.stdout.split("\n")

    @classmethod
    def update_version_file(cls, version):
        with open(cls.VERSION_FILE, 'w') as file:
            file.write(version)
        print(f"Updated {cls.VERSION_FILE} with version: {version}")

if __name__ == "__main__":
    AutoVersionBumper.bump_version()
