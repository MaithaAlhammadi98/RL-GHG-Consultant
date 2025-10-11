"""
Real-time Q-table monitor - Shows live updates to q_table.json
"""
import json
import time
from pathlib import Path
from datetime import datetime

q_file = Path("src/data/q_table.json")

print("=" * 60)
print("Q-TABLE LIVE MONITOR")
print("=" * 60)
print(f"Watching: {q_file.absolute()}")
print("Press Ctrl+C to stop\n")

last_content = None
last_mtime = None

try:
    while True:
        if q_file.exists():
            current_mtime = q_file.stat().st_mtime
            
            # Check if file was modified
            if last_mtime is None or current_mtime > last_mtime:
                try:
                    with open(q_file, 'r') as f:
                        current_content = json.load(f)
                    
                    if last_content is not None:
                        # File was updated!
                        print(f"\n[{datetime.now().strftime('%H:%M:%S')}] FILE UPDATED!")
                        print(f"Total states: {len(current_content)}")
                        
                        # Show what changed
                        for state_key, q_vals in current_content.items():
                            if state_key not in last_content or last_content[state_key] != q_vals:
                                print(f"\n  State: {state_key}")
                                print(f"  Q-values: {q_vals}")
                                
                                if state_key in last_content:
                                    old_vals = last_content[state_key]
                                    for action, new_val in q_vals.items():
                                        old_val = old_vals.get(action, 0.0)
                                        if abs(new_val - old_val) > 0.0001:
                                            print(f"    {action}: {old_val:.4f} -> {new_val:.4f} (change: {new_val-old_val:+.4f})")
                    else:
                        print(f"[{datetime.now().strftime('%H:%M:%S')}] Initial load: {len(current_content)} states")
                    
                    last_content = current_content.copy()
                    last_mtime = current_mtime
                    
                except Exception as e:
                    print(f"[ERROR] Could not read file: {e}")
        
        time.sleep(1)  # Check every second
        
except KeyboardInterrupt:
    print("\n\nMonitor stopped.")
    print("=" * 60)

