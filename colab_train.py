# FILE: colab_train.py
# PURPOSE: All-in-One Google Colab Git Clone/Pull + Vault Restore + Trainer + Drive/Git/PC Export
# USAGE IN COLAB: Paste this entire file into ONE Colab cell and click Run,
#                 OR upload it to Colab and run:  %run colab_train.py

import glob
import os
import shutil
import subprocess
import sys

# =====================================================================
# 1. CONFIGURATION (EDIT THESE SETTINGS ONCE)
# =====================================================================
GITHUB_USER = "xithu01"
GITHUB_REPO = "dating_bot_ai"
USE_PRIVATE_TOKEN = True       # Set True for private repo (save GITHUB_TOKEN in Colab Secrets)

TRAIN_FLAG = ""                 # "" = protect old steps & train only new/missing steps
                                # Or use: "--step1", "--step2", "--step3", "--all"
CLEAR_OLD_TXT = False           # Set True after changing dataset generators to rebuild .txt files

USE_DRIVE_BACKUP = True         # Restores old model_weights.npz from Drive & saves updated copy
PUSH_WEIGHTS_TO_GIT = True     # Set True to push ONLY model_weights.npz to GitHub (needs token)
DOWNLOAD_TO_PC = True           # Auto-downloads model_weights.npz to your PC when finished

REPO_DIR = "/content/dating_bot_ai"
DRIVE_VAULT = "/content/drive/MyDrive/dating_bot_ai/model_weights.npz"
LOCAL_VAULT = os.path.join(REPO_DIR, "model_weights.npz")


def run_cmd(cmd, cwd=None, check=True):
    """Run a shell command with live streaming output in Colab."""
    proc = subprocess.Popen(
        cmd,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    for line in proc.stdout:
        print(line, end="", flush=True)
    proc.wait()
    if check and proc.returncode != 0:
        raise RuntimeError(f"Command failed (exit {proc.returncode}): {' '.join(cmd)}")
    return proc.returncode


def build_repo_url():
    if USE_PRIVATE_TOKEN:
        from google.colab import userdata
        token = userdata.get("GITHUB_TOKEN")
        if not token:
            raise ValueError("Missing GITHUB_TOKEN in Colab Secrets (key icon on left sidebar).")
        return f"https://{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
    return f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"


def main():
    if GITHUB_USER == "YOUR_USERNAME" or GITHUB_REPO == "YOUR_REPO":
        raise ValueError("Please set GITHUB_USER and GITHUB_REPO at the top of this script first.")

    # --- STEP 1: CLONE OR PULL LATEST CODE FROM GIT ---
    repo_url = build_repo_url()
    if os.path.exists(os.path.join(REPO_DIR, ".git")):
        print("[GIT] Repo exists -> Pulling latest code updates...", flush=True)
        run_cmd(["git", "remote", "set-url", "origin", repo_url], cwd=REPO_DIR)
        run_cmd(["git", "pull", "--rebase", "--autostash"], cwd=REPO_DIR)
    else:
        print("[GIT] Cloning repository...", flush=True)
        run_cmd(["git", "clone", repo_url, REPO_DIR])

    os.chdir(REPO_DIR)

    # --- STEP 2: RESTORE OLD model_weights.npz IF IT EXISTS ---
    if USE_DRIVE_BACKUP:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        os.makedirs(os.path.dirname(DRIVE_VAULT), exist_ok=True)
        if os.path.exists(DRIVE_VAULT):
            shutil.copy2(DRIVE_VAULT, LOCAL_VAULT)
            kb = os.path.getsize(LOCAL_VAULT) / 1024.0
            print(f"[VAULT] Loaded existing model_weights.npz from Google Drive ({kb:.2f} KB)", flush=True)
        elif os.path.exists(LOCAL_VAULT):
            kb = os.path.getsize(LOCAL_VAULT) / 1024.0
            print(f"[VAULT] Using existing model_weights.npz from Git ({kb:.2f} KB)", flush=True)
        else:
            print("[VAULT] No old model_weights.npz found -> Training fresh vault.", flush=True)
    elif os.path.exists(LOCAL_VAULT):
        kb = os.path.getsize(LOCAL_VAULT) / 1024.0
        print(f"[VAULT] Using existing model_weights.npz from Git ({kb:.2f} KB)", flush=True)
    else:
        print("[VAULT] No old model_weights.npz found -> Training fresh vault.", flush=True)

    # --- STEP 3: OPTIONAL CLEAN REBUILD OF .TXT DATASETS ---
    if CLEAR_OLD_TXT:
        removed = 0
        for txt_path in glob.glob(os.path.join(REPO_DIR, "*.txt")):
            if os.path.basename(txt_path).lower() != "requirements.txt":
                os.remove(txt_path)
                removed += 1
        print(f"[DATASETS] Cleared {removed} old .txt dataset file(s) for clean rebuild.", flush=True)

    # --- STEP 4: RUN OFFLINE TRAINER (STREAMS LIVE ETA) ---
    train_cmd = [sys.executable, "offline_train.py"]
    if TRAIN_FLAG.strip():
        train_cmd.extend(TRAIN_FLAG.strip().split())
    print(f"[TRAIN] Running: {' '.join(train_cmd)}", flush=True)
    run_cmd(train_cmd, cwd=REPO_DIR)

    if not os.path.exists(LOCAL_VAULT):
        raise FileNotFoundError("model_weights.npz was not generated!")

    final_kb = os.path.getsize(LOCAL_VAULT) / 1024.0
    print(f"[DONE] Vault ready: {LOCAL_VAULT} ({final_kb:.2f} KB)", flush=True)

    # --- STEP 5: BACK UP UPDATED model_weights.npz TO GOOGLE DRIVE ---
    if USE_DRIVE_BACKUP:
        shutil.copy2(LOCAL_VAULT, DRIVE_VAULT)
        print(f"[DRIVE] Saved updated model_weights.npz to Google Drive ({final_kb:.2f} KB)", flush=True)

    # --- STEP 6: PUSH ONLY model_weights.npz TO GIT (OPTIONAL) ---
    if PUSH_WEIGHTS_TO_GIT:
        run_cmd(["git", "config", "user.name", "Colab Trainer"], cwd=REPO_DIR)
        run_cmd(["git", "config", "user.email", "trainer@local"], cwd=REPO_DIR)
        run_cmd(["git", "add", "model_weights.npz"], cwd=REPO_DIR)
        rc = run_cmd(["git", "commit", "-m", "Update model_weights.npz"], cwd=REPO_DIR, check=False)
        if rc == 0:
            run_cmd(["git", "push"], cwd=REPO_DIR)
            print("[GIT] Pushed updated model_weights.npz to GitHub.", flush=True)
        else:
            print("[GIT] model_weights.npz unchanged; skipped push.", flush=True)

    # --- STEP 7: DOWNLOAD model_weights.npz TO PC (OPTIONAL) ---
    if DOWNLOAD_TO_PC:
        from google.colab import files
        print("[DOWNLOAD] Sending model_weights.npz to your PC...", flush=True)
        files.download(LOCAL_VAULT)


if __name__ == "__main__":
    main()