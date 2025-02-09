import subprocess
import sys

def get_latest_tag():
    result = subprocess.run(["git", "describe", "--tags", "--abbrev=0"], capture_output=True, text=True)
    if result.returncode != 0:
        print("No tags found in the repository. Please create a tag to start generating changelogs.")
        sys.exit(1)
    return result.stdout.strip()

def get_changelog(latest_tag):
    command = ["git", "log", f"{latest_tag}..HEAD", "--pretty=format:- %s (%h) by %an"]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"Error fetching changelog: {result.stdout}")
        sys.exit(1)
    return result.stdout.strip()

def generate_changelog():
    latest_tag = get_latest_tag()
    changelog = get_changelog(latest_tag)
    
    if not changelog:
        print(f"No new commits found since the latest tag ({latest_tag}).")
        sys.exit(0)
    
    changelog_content = f"""
    ## Changelog
    ### From {latest_tag} to HEAD

    {changelog}
    """
    
    with open("CHANGELOG.md", "a") as file:
        file.write(changelog_content + "\n")
    
    print("Changelog generated and saved to CHANGELOG.md")

if __name__ == "__main__":
    generate_changelog()
