# Running Training in tmux/screen - Complete Guide

## Why Use tmux/screen?

Training takes 3-4 hours. If you disconnect from SSH or close your laptop:
- ❌ **Without tmux/screen**: Training stops
- ✅ **With tmux/screen**: Training continues running on the server

## Quick Start (Recommended)

### Option 1: tmux (Modern, Recommended)

```bash
# Start tmux session
tmux new -s training

# Run the training
bash run_full_retraining.sh

# Detach from session (training continues): Press Ctrl+B, then D
# Or close your laptop - training keeps running!

# Later, reattach to check progress:
tmux attach -t training

# Kill session when done:
tmux kill-session -t training
```

### Option 2: screen (Alternative)

```bash
# Start screen session
screen -S training

# Run the training
bash run_full_retraining.sh

# Detach from session: Press Ctrl+A, then D

# Later, reattach:
screen -r training

# Kill session when done:
screen -XS training quit
```

### Option 3: nohup (Simplest, but less control)

```bash
# Run in background
nohup bash run_full_retraining.sh > training.log 2>&1 &

# Check progress:
tail -f training.log

# Find process:
ps aux | grep run_full_retraining

# Kill if needed:
kill <PID>
```

## Detailed tmux Instructions

### 1. Install tmux (if not installed)

```bash
# Ubuntu/Debian
sudo apt-get install tmux

# CentOS/RHEL
sudo yum install tmux

# Check installation
tmux -V
```

### 2. Start Training in tmux

```bash
# Navigate to project directory
cd /home/dhruv/prashantAi/FSL/few_shot_sketch_recognition

# Start tmux session named "training"
tmux new -s training

# Inside tmux, run the training
bash run_full_retraining.sh
```

### 3. Detach from tmux (Training Continues)

**Press: `Ctrl+B`, then press `D`**

You'll see: `[detached (from session training)]`

Now you can:
- Close your terminal
- Disconnect SSH
- Close your laptop
- Training keeps running on the server!

### 4. Check Progress Later

```bash
# List all tmux sessions
tmux ls

# Reattach to training session
tmux attach -t training

# Or shorthand:
tmux a -t training
```

### 5. View Logs While Training

You can watch logs in real-time:

```bash
# In another tmux window, or separate SSH session:
tail -f logs/ssl/ssl_training.log
tail -f logs/few_shot/few_shot_training.log
```

### 6. Split Screen in tmux (Advanced)

```bash
# Start tmux
tmux new -s training

# Run training in first pane
bash run_full_retraining.sh

# Split screen vertically: Ctrl+B, then %
# Or horizontally: Ctrl+B, then "

# Switch between panes: Ctrl+B, then arrow keys

# In second pane, watch logs:
tail -f logs/ssl/ssl_training.log

# Navigate between panes with: Ctrl+B, then arrow keys
```

## tmux Cheat Sheet

| Action | Command |
|--------|---------|
| Start new session | `tmux new -s training` |
| Detach | `Ctrl+B`, then `D` |
| List sessions | `tmux ls` |
| Attach | `tmux attach -t training` |
| Kill session | `tmux kill-session -t training` |
| Split vertical | `Ctrl+B`, then `%` |
| Split horizontal | `Ctrl+B`, then `"` |
| Navigate panes | `Ctrl+B`, then arrow keys |
| Close pane | `Ctrl+D` or `exit` |
| Scroll up | `Ctrl+B`, then `[`, then arrow keys |
| Exit scroll mode | `q` |

## Complete Workflow Example

```bash
# 1. SSH into server
ssh user@server

# 2. Navigate to project
cd /home/dhruv/prashantAi/FSL/few_shot_sketch_recognition

# 3. Start tmux session
tmux new -s training

# 4. Run training
bash run_full_retraining.sh

# 5. Detach (Ctrl+B, then D)
# Training continues in background

# 6. Close laptop, go home, etc.

# 7. Later, SSH back in
ssh user@server

# 8. Reattach to session
tmux attach -t training

# 9. Check progress, wait for completion

# 10. When done, kill session
tmux kill-session -t training
```

## Monitoring Progress

### While Detached

```bash
# Check if training is still running
tmux ls

# Check GPU usage
nvidia-smi

# Check process
ps aux | grep python

# Check logs
tail -f logs/ssl/ssl_training.log
tail -f logs/few_shot/few_shot_training.log
```

### Inside tmux Session

You can see the training progress directly in the terminal output.

## Troubleshooting

### Problem: tmux not found

```bash
# Install tmux
sudo apt-get install tmux  # Ubuntu/Debian
sudo yum install tmux       # CentOS/RHEL
```

### Problem: Session not found

```bash
# List all sessions
tmux ls

# Attach to correct session name
tmux attach -t <session_name>
```

### Problem: Training stopped unexpectedly

```bash
# Check logs
tail -100 logs/ssl/ssl_training.log
tail -100 logs/few_shot/few_shot_training.log

# Check for errors
grep -i error logs/ssl/ssl_training.log
grep -i error logs/few_shot/few_shot_training.log
```

### Problem: Can't detach

- Make sure you press `Ctrl+B` first, **then** `D`
- Not simultaneously, but sequentially

## Best Practices

1. **Always use tmux/screen for long training**
   - Training takes 3-4 hours
   - Don't risk losing progress

2. **Name your sessions meaningfully**
   ```bash
   tmux new -s ssl_training
   tmux new -s fsl_training
   ```

3. **Monitor GPU usage**
   ```bash
   watch -n 1 nvidia-smi
   ```

4. **Keep logs**
   - Logs are automatically saved in `logs/` directory
   - Check them if training fails

5. **Test first**
   - Run a quick test (10 epochs) before full training
   - Verify everything works

## Single Command (All-in-One)

For the laziest approach 😊:

```bash
# One command to rule them all:
tmux new -s training "bash run_full_retraining.sh; bash"

# This will:
# 1. Start tmux session
# 2. Run training
# 3. Keep shell open when done (so you can see results)
# 4. Detach with: Ctrl+B, then D
```

## Checking Progress from Anywhere

You can even check progress from your phone:

1. Install a terminal app (e.g., Termius, JuiceSSH)
2. SSH into server
3. Run: `tmux attach -t training`
4. See real-time progress

## Summary

### Recommended Setup:

```bash
# Start
cd /home/dhruv/prashantAi/FSL/few_shot_sketch_recognition
tmux new -s training
bash run_full_retraining.sh

# Detach: Ctrl+B, then D
# Come back later: tmux attach -t training
```

This setup ensures:
- ✅ Training continues if you disconnect
- ✅ You can check progress anytime
- ✅ No need to keep laptop on
- ✅ Can monitor from anywhere
- ✅ Logs are saved automatically

**Expected time:**
- SSL training: 2-3 hours
- Few-shot training: 30-45 minutes
- Evaluation: 5 minutes
- **Total: ~3-4 hours**

You can close your laptop and go do something else! 🎉

