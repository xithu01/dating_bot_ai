# FILE: colab_train.py
# PURPOSE: Final All-in-One Google Colab Git Sync + Vault Restore + Trainer + Drive/PC/Git Export
# USAGE:   Run `%run /content/drive/MyDrive/dating_bot_ai/colab_train.py` in a single Colab cell

import glob
import os
import shutil
import subprocess
import sys

# =====================================================================
# 1. CONFIGURATION (PRE-CONFIGURED FOR xithu01/dating_bot_ai)
# =====================================================================
GITHUB_USER = "xithu01"
GITHUB_REPO = "dating_bot_ai"
USE_PRIVATE_TOKEN = True        # Reads GITHUB_TOKEN from Colab Secrets (supports github_pat_ & ghp_)

TRAIN_FLAG = ""                 # "" = protect old steps & train only new/missing steps
                                # Or set: "--step1", "--step2", "--step3", "--all"
CLEAR_OLD_TXT = False           # Set True after editing dataset builders to rebuild .txt files

USE_DRIVE_BACKUP = True         # Restores old model_weights.npz from Drive & saves updated copy
DOWNLOAD_TO_PC = False           # Downloads model_weights.npz to your PC before git push
PUSH_WEIGHTS_TO_GIT = True      # Commits & pushes ONLY model_weights.npz to GitHub

REPO_DIR = "/content/dating_bot_ai"
DRIVE_DIR = "/content/drive/MyDrive/dating_bot_ai"
DRIVE_VAULT = os.path.join(DRIVE_DIR, "model_weights.npz")
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
        token = (userdata.get("GITHUB_TOKEN") or "").strip()
        if not token:
            raise ValueError("Missing GITHUB_TOKEN in Colab Secrets (Key icon on left sidebar).")
        return f"https://x-access-token:{token}@github.com/{GITHUB_USER}/{GITHUB_REPO}.git"
    return f"https://github.com/{GITHUB_USER}/{GITHUB_REPO}.git"


def main():
    # --- STEP 1: MOUNT DRIVE (IF ENABLED) ---
    if USE_DRIVE_BACKUP:
        from google.colab import drive
        if not os.path.exists("/content/drive/MyDrive"):
            drive.mount("/content/drive")
        os.makedirs(DRIVE_DIR, exist_ok=True)

    # --- STEP 2: CLONE OR PULL LATEST CODE FROM GITHUB ---
    repo_url = build_repo_url()
    if os.path.exists(os.path.join(REPO_DIR, ".git")):
        print("[GIT] Repo exists -> Updating token URL & pulling latest code...", flush=True)
        run_cmd(["git", "remote", "set-url", "origin", repo_url], cwd=REPO_DIR)
        run_cmd(["git", "pull", "--rebase", "--autostash"], cwd=REPO_DIR)
    else:
        print("[GIT] Cloning repository...", flush=True)
        run_cmd(["git", "clone", repo_url, REPO_DIR])

    os.chdir(REPO_DIR)

    # --- STEP 3: RESTORE OLD model_weights.npz IF IT EXISTS ---
    if USE_DRIVE_BACKUP and os.path.exists(DRIVE_VAULT):
        shutil.copy2(DRIVE_VAULT, LOCAL_VAULT)
        kb = os.path.getsize(LOCAL_VAULT) / 1024.0
        print(f"[VAULT] Loaded existing model_weights.npz from Google Drive ({kb:.2f} KB)", flush=True)
    elif os.path.exists(LOCAL_VAULT):
        kb = os.path.getsize(LOCAL_VAULT) / 1024.0
        print(f"[VAULT] Using existing model_weights.npz from Git ({kb:.2f} KB)", flush=True)
    else:
        print("[VAULT] No old model_weights.npz found -> Training fresh vault.", flush=True)

    # --- STEP 4: OPTIONAL CLEAN REBUILD OF .TXT DATASETS ---
    if CLEAR_OLD_TXT:
        removed = 0
        for txt_path in glob.glob(os.path.join(REPO_DIR, "*.txt")):
            if os.path.basename(txt_path).lower() != "requirements.txt":
                os.remove(txt_path)
                removed += 1
        print(f"[DATASETS] Cleared {removed} old .txt dataset file(s) for clean rebuild.", flush=True)

    # --- STEP 5: RUN OFFLINE TRAINER (STREAMS LIVE 1s ETA) ---
    train_cmd = [sys.executable, "offline_train.py"]
    if TRAIN_FLAG.strip():
        train_cmd.extend(TRAIN_FLAG.strip().split())
    print(f"[TRAIN] Running: {' '.join(train_cmd)}", flush=True)
    run_cmd(train_cmd, cwd=REPO_DIR)

    if not os.path.exists(LOCAL_VAULT):
        raise FileNotFoundError("model_weights.npz was not generated!")

    final_kb = os.path.getsize(LOCAL_VAULT) / 1024.0
    print(f"[DONE] Vault ready: {LOCAL_VAULT} ({final_kb:.2f} KB)", flush=True)

    # --- STEP 6: BACK UP UPDATED model_weights.npz TO GOOGLE DRIVE ---
    if USE_DRIVE_BACKUP:
        shutil.copy2(LOCAL_VAULT, DRIVE_VAULT)
        print(f"[DRIVE] Saved updated model_weights.npz to Google Drive ({final_kb:.2f} KB)", flush=True)

    # --- STEP 7: DOWNLOAD model_weights.npz TO PC FIRST ---
    if DOWNLOAD_TO_PC:
        from google.colab import files
        print("[DOWNLOAD] Sending model_weights.npz to your PC...", flush=True)
        files.download(LOCAL_VAULT)

    # --- STEP 8: PUSH ONLY model_weights.npz TO GITHUB ---
    if PUSH_WEIGHTS_TO_GIT:
        run_cmd(["git", "remote", "set-url", "origin", repo_url], cwd=REPO_DIR)
        run_cmd(["git", "config", "user.name", "Colab Trainer"], cwd=REPO_DIR)
        run_cmd(["git", "config", "user.email", "trainer@local"], cwd=REPO_DIR)
        run_cmd(["git", "add", "model_weights.npz"], cwd=REPO_DIR)
        run_cmd(["git", "commit", "-m", "Update model_weights.npz"], cwd=REPO_DIR, check=False)
        rc = run_cmd(["git", "push"], cwd=REPO_DIR, check=False)
        if rc == 0:
            print("[GIT] Pushed updated model_weights.npz to GitHub.", flush=True)
        else:
            print("[GIT WARNING] Push failed (check token Contents=Read/Write), but Drive & PC download succeeded.", flush=True)


if __name__ == "__main__":
    main()