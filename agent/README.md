# Digit Motion Agent

An OpenAI-powered agent that combines recorded robot arm motions for individual digits (0-9) to produce any multi-digit number.

## Overview

This agent:
1. Loads motion data from LeRobot parquet files for digits 0-9
2. Uses the **latest folder versions** (S101_x_y takes priority over S101_x)
3. Provides an interactive chat interface to generate combined motions
4. Outputs combined motion data in LeRobot-compatible format

## Data Structure

The agent expects data in the following format:

```
shubhamt0802/
├── S101_0/          # Digit 0 (version 0)
├── S101_0_3/        # Digit 0 (version 3) - USED (latest)
├── S101_1/          # Digit 1 (version 0)
├── S101_1_1/        # Digit 1 (version 1) - USED (latest)
├── S101_2/          # Digit 2 (version 0)
├── S101_2_1/        # Digit 2 (version 1) - USED (latest)
...
```

Each folder contains:
- `data/chunk-000/file-000.parquet` - Motion data
- `meta/info.json` - Metadata (fps, frames, features)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Make sure LeRobot is installed for replay functionality
pip install -e ".[feetech]"
```

## Usage

### Interactive Mode

```bash
# Set your OpenAI API key
export OPENAI_API_KEY="your-api-key"

# Run the agent
python digit_motion_agent.py --data-dir ../shubhamt0802
```

Then interact with the agent:
```
You: What digits do you have available?
Agent: I have motion data for digits 0-9...

You: Generate motion for the number 18
Agent: I'll generate motion data for the number 18...
       Output saved to: ../shubhamt0802/combined_number_18/

You: quit
```

### Non-Interactive Mode

```bash
# Directly generate motion for a number
python digit_motion_agent.py \
    --data-dir ../shubhamt0802 \
    --number 42
```

### Replay Generated Motion

```bash
# Simulate replay (no robot)
python replay_combined_motion.py \
    --motion-dir ../shubhamt0802/combined_number_18 \
    --simulate

# Replay on actual robot
python replay_combined_motion.py \
    --motion-dir ../shubhamt0802/combined_number_18 \
    --robot-port /dev/tty.usbmodem58760431541
```

## API Reference

### DigitMotionLoader

Loads and manages digit motion data from parquet files.

```python
from digit_motion_agent import DigitMotionLoader

loader = DigitMotionLoader("./shubhamt0802")
print(loader.get_available_digits())  # [0, 1, 2, ..., 9]
print(loader.get_digit_summary())     # Description of each digit
```

### MotionCombiner

Combines multiple digit motions into a single sequence.

```python
from digit_motion_agent import DigitMotionLoader, MotionCombiner

loader = DigitMotionLoader("./shubhamt0802")
combiner = MotionCombiner(loader)

# Combine digits 1 and 8 for number "18"
combined_df = combiner.combine_digits([1, 8], pause_frames=30)

# Save in LeRobot format
info = combiner.save_combined_motion(combined_df, "./output/combined_18", number=18)
```

### DigitMotionAgent

OpenAI-powered agent for interactive digit motion generation.

```python
from digit_motion_agent import DigitMotionAgent

agent = DigitMotionAgent(
    data_dir="./shubhamt0802",
    api_key="your-openai-key",
    model="gpt-4o"
)

# Single message
response = agent.chat("Generate motion for 123")

# Interactive session
agent.run_interactive()
```

## Output Format

The agent outputs data in LeRobot v3.0 format:

```
combined_number_18/
├── data/
│   └── chunk-000/
│       └── file-000.parquet
└── meta/
    ├── info.json
    └── stats.json
```

The parquet file contains:
- `action` - Robot joint positions (6 DOF: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper)
- `observation.state` - Same as action
- `timestamp` - Time in seconds
- `frame_index` - Frame number
- `episode_index` - Always 0 for combined motion
- `task_index` - Always 0

## Tools Available to Agent

The OpenAI agent has access to these tools:

1. **generate_number_motion** - Generate combined motion for any number
2. **get_digit_info** - Get info about a specific digit's motion
3. **list_available_digits** - List all available digits

## Folder Selection Logic

The agent automatically selects the **latest version** for each digit:

- `S101_3` = digit 3, version 0
- `S101_3_0` = digit 3, version 0 (same as above)
- `S101_3_2` = digit 3, version 2 (takes priority)

The folder with the highest version number is used.

## Example Session

```
$ python digit_motion_agent.py --data-dir ../shubhamt0802

============================================================
🤖 Digit Motion Agent
============================================================

I can help you generate robot motion data for writing numbers.
Available digits: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]

Type 'quit' or 'exit' to end the session.
============================================================

You: I want to write the number 18

Agent: I'll generate the motion data for the number 18. This will combine 
the recorded motions for digit 1 and digit 8 in sequence.

✅ Motion generated successfully!
- Number: 18
- Digits: [1, 8]
- Total frames: 857
- Duration: 28.57 seconds
- Output: ../shubhamt0802/combined_number_18/

To replay this on your robot:
python replay_combined_motion.py \
    --motion-dir ../shubhamt0802/combined_number_18 \
    --robot-port /dev/tty.usbmodem58760431541

You: quit
Goodbye! 👋
```

## Troubleshooting

### Missing API Key
```
ValueError: OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.
```
Solution: `export OPENAI_API_KEY="sk-..."`

### Missing Digit Data
```
Missing motion data for digits: [7]
```
Solution: Ensure you have recorded data for all digits (S101_0 through S101_9)

### Robot Connection Issues
```
Could not connect to robot on port...
```
Solution: Check USB connection and port name with `lerobot-find-port`

