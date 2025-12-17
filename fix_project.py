import os
import shutil
import subprocess
import sys

# Define the source folder and files to move
BASE_DIR = os.getcwd()
SRC_DIR = os.path.join(BASE_DIR, "src")
FILES_TO_MOVE = [
    "config.py",
    "generate_submission.py",
    "generator.py",
    "indexer.py",
    "ingestion.py",
    "retriever.py"
]

def run_command(command):
    print(f"👉 Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running command: {e}")

def main():
    print("🚀 STARTING AUTOMATED FIX...")

    # 1. Create 'src' directory if it doesn't exist
    if not os.path.exists(SRC_DIR):
        print(f"📂 Creating directory: {SRC_DIR}")
        os.makedirs(SRC_DIR)
    
    # 2. Move Python files to 'src'
    for filename in FILES_TO_MOVE:
        source = os.path.join(BASE_DIR, filename)
        destination = os.path.join(SRC_DIR, filename)
        
        if os.path.exists(source):
            print(f"📦 Moving {filename} -> src/")
            try:
                shutil.move(source, destination)
            except Exception as e:
                print(f"⚠️ Could not move {filename}: {e}")
        elif os.path.exists(destination):
            print(f"✅ {filename} is already in src/")
        else:
            print(f"❓ Warning: {filename} not found in root or src.")

    # 3. Fix Git
    print("\n🔧 FIXING GIT REPOSITORY...")
    run_command("git add .")
    run_command('git commit -m "Fixed folder structure via script"')
    
    print("\n☁️ UPLOADING TO GITHUB...")
    # Using --force to overwrite any conflicts
    run_command("git push -f origin main")

    print("\n✅ DONE! Check your GitHub URL now.")

if __name__ == "__main__":
    main()