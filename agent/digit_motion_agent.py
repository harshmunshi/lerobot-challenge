#!/usr/bin/env python
"""
OpenAI Agent for combining LeRobot digit motion data.

This agent:
1. Loads motion data from parquet files for digits 0-9
2. Uses latest folder versions (S101_x_y takes priority over S101_x)
3. Asks user for a number and combines digit motions
4. Outputs combined motion data in lerobot-compatible format

Usage:
    python digit_motion_agent.py --data-dir /path/to/shubhamt0802

Requirements:
    pip install openai pandas pyarrow
"""

import json
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
import pyarrow.parquet as pq
from openai import OpenAI


@dataclass
class DigitMotionData:
    """Container for a single digit's motion data."""
    digit: int
    folder_name: str
    total_frames: int
    fps: int
    actions: pd.DataFrame
    description: str


class DigitMotionLoader:
    """Loads and manages digit motion data from LeRobot parquet files."""
    
    FOLDER_PATTERN = re.compile(r"S101_(\d)(?:_(\d+))?$")
    
    def __init__(self, data_dir: str | Path):
        self.data_dir = Path(data_dir)
        self.digit_data: dict[int, DigitMotionData] = {}
        self._load_all_digits()
    
    def _parse_folder_name(self, folder_name: str) -> tuple[int, int] | None:
        """Parse folder name to extract digit and version.
        
        Returns:
            Tuple of (digit, version) or None if invalid format.
            S101_3 -> (3, 0)
            S101_3_2 -> (3, 2)
        """
        match = self.FOLDER_PATTERN.match(folder_name)
        if not match:
            return None
        digit = int(match.group(1))
        version = int(match.group(2)) if match.group(2) else 0
        return digit, version
    
    def _find_latest_folders(self) -> dict[int, str]:
        """Find the latest folder for each digit (0-9).
        
        Returns:
            Dictionary mapping digit -> folder name
        """
        latest_folders: dict[int, tuple[int, str]] = {}  # digit -> (version, folder_name)
        
        for folder in self.data_dir.iterdir():
            if not folder.is_dir():
                continue
            
            parsed = self._parse_folder_name(folder.name)
            if parsed is None:
                continue
            
            digit, version = parsed
            
            # Check if this folder has valid data
            data_parquet = folder / "data" / "chunk-000" / "file-000.parquet"
            if not data_parquet.exists():
                continue
            
            # Update if this is the latest version
            if digit not in latest_folders or version > latest_folders[digit][0]:
                latest_folders[digit] = (version, folder.name)
        
        return {digit: folder_name for digit, (_, folder_name) in latest_folders.items()}
    
    def _load_motion_data(self, folder_path: Path) -> tuple[pd.DataFrame, dict]:
        """Load motion data from a folder.
        
        Returns:
            Tuple of (actions DataFrame, info dict)
        """
        data_parquet = folder_path / "data" / "chunk-000" / "file-000.parquet"
        info_json = folder_path / "meta" / "info.json"
        
        # Load parquet data
        df = pd.read_parquet(data_parquet)
        
        # Load info
        with open(info_json) as f:
            info = json.load(f)
        
        # The 'action' column contains arrays of 6 floats
        # Extract individual components for easier manipulation
        action_names = info["features"]["action"]["names"]
        if "action" in df.columns:
            # Expand action array into individual columns
            action_arrays = df["action"].tolist()
            for i, name in enumerate(action_names):
                df[f"action_{name}"] = [arr[i] for arr in action_arrays]
        
        if "observation.state" in df.columns:
            # Expand observation.state array into individual columns
            state_arrays = df["observation.state"].tolist()
            for i, name in enumerate(action_names):
                df[f"observation.state_{name}"] = [arr[i] for arr in state_arrays]
        
        return df, info
    
    def _generate_description(self, digit: int, folder_name: str, total_frames: int, fps: int) -> str:
        """Generate a description for the digit motion data."""
        duration = total_frames / fps
        return (
            f"Motion data for digit '{digit}' recorded from SO101 robot arm. "
            f"Source: {folder_name}, Duration: {duration:.2f}s ({total_frames} frames at {fps} FPS). "
            f"This is a recorded trajectory of the robot arm drawing the digit {digit}."
        )
    
    def _load_all_digits(self):
        """Load motion data for all available digits."""
        latest_folders = self._find_latest_folders()
        
        for digit, folder_name in latest_folders.items():
            folder_path = self.data_dir / folder_name
            df, info = self._load_motion_data(folder_path)
            
            # Extract action columns (both original array and expanded)
            action_cols = [col for col in df.columns if col.startswith("action")]
            state_cols = [col for col in df.columns if col.startswith("observation.state")]
            base_cols = ["timestamp", "frame_index", "episode_index", "index", "task_index"]
            
            # Keep columns that exist
            keep_cols = [c for c in action_cols + state_cols + base_cols if c in df.columns]
            actions_df = df[keep_cols].copy()
            
            description = self._generate_description(
                digit=digit,
                folder_name=folder_name,
                total_frames=info["total_frames"],
                fps=info["fps"]
            )
            
            self.digit_data[digit] = DigitMotionData(
                digit=digit,
                folder_name=folder_name,
                total_frames=info["total_frames"],
                fps=info["fps"],
                actions=actions_df,
                description=description
            )
    
    def get_available_digits(self) -> list[int]:
        """Get list of available digits."""
        return sorted(self.digit_data.keys())
    
    def get_digit_summary(self) -> str:
        """Get a summary of all loaded digit data for context."""
        summaries = []
        for digit in sorted(self.digit_data.keys()):
            data = self.digit_data[digit]
            summaries.append(f"- Digit {digit}: {data.description}")
        return "\n".join(summaries)
    
    def get_digit_motion(self, digit: int) -> DigitMotionData | None:
        """Get motion data for a specific digit."""
        return self.digit_data.get(digit)


class MotionCombiner:
    """Combines multiple digit motions into a single sequence."""
    
    def __init__(self, loader: DigitMotionLoader):
        self.loader = loader
    
    def combine_digits(self, digits: list[int], pause_frames: int = 30) -> pd.DataFrame:
        """Combine motion data for multiple digits.
        
        Args:
            digits: List of digits to combine (e.g., [1, 8] for "18")
            pause_frames: Number of frames to pause between digits
            
        Returns:
            Combined DataFrame with all actions
        """
        combined_actions = []
        current_frame = 0
        
        for i, digit in enumerate(digits):
            motion_data = self.loader.get_digit_motion(digit)
            if motion_data is None:
                raise ValueError(f"Digit {digit} not available in the dataset")
            
            # Copy the actions and adjust frame indices
            digit_actions = motion_data.actions.copy()
            digit_actions["frame_index"] = digit_actions["frame_index"] + current_frame
            digit_actions["original_digit"] = digit
            digit_actions["digit_sequence_index"] = i
            
            combined_actions.append(digit_actions)
            current_frame += len(digit_actions) + pause_frames
            
            # Add pause frames (repeat last position) between digits
            if i < len(digits) - 1 and pause_frames > 0:
                last_row = digit_actions.iloc[[-1]].copy()
                pause_df = pd.concat([last_row] * pause_frames, ignore_index=True)
                pause_df["frame_index"] = range(
                    current_frame - pause_frames, 
                    current_frame
                )
                pause_df["original_digit"] = -1  # Marker for pause
                pause_df["digit_sequence_index"] = i
                combined_actions.append(pause_df)
        
        return pd.concat(combined_actions, ignore_index=True)
    
    def save_combined_motion(
        self, 
        combined_df: pd.DataFrame, 
        output_path: Path,
        number: int,
        fps: int = 30
    ) -> dict:
        """Save combined motion data in LeRobot-compatible format.
        
        Args:
            combined_df: Combined DataFrame from combine_digits
            output_path: Directory to save the output
            number: The number being produced
            fps: Frames per second
            
        Returns:
            Info dictionary about the saved data
        """
        import numpy as np
        
        output_path = Path(output_path)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # Create data directory structure
        data_dir = output_path / "data" / "chunk-000"
        data_dir.mkdir(parents=True, exist_ok=True)
        
        meta_dir = output_path / "meta"
        meta_dir.mkdir(parents=True, exist_ok=True)
        
        episodes_dir = meta_dir / "episodes" / "chunk-000"
        episodes_dir.mkdir(parents=True, exist_ok=True)
        
        # Action names for SO101
        action_names = [
            "shoulder_pan.pos",
            "shoulder_lift.pos", 
            "elbow_flex.pos",
            "wrist_flex.pos",
            "wrist_roll.pos",
            "gripper.pos"
        ]
        
        # Create the output DataFrame
        output_df = pd.DataFrame()
        
        # Recalculate frame indices sequentially
        output_df["frame_index"] = range(len(combined_df))
        output_df["episode_index"] = 0
        output_df["index"] = range(len(combined_df))
        output_df["task_index"] = 0
        output_df["timestamp"] = output_df["frame_index"] / fps
        
        # Handle action column - reconstruct as array if expanded
        if "action" in combined_df.columns:
            # Original array column exists
            output_df["action"] = combined_df["action"].values
        else:
            # Reconstruct from expanded columns
            action_arrays = []
            for _, row in combined_df.iterrows():
                arr = []
                for name in action_names:
                    col_name = f"action_{name}"
                    if col_name in combined_df.columns:
                        arr.append(float(row[col_name]))
                    else:
                        arr.append(0.0)
                action_arrays.append(np.array(arr, dtype=np.float32))
            output_df["action"] = action_arrays
        
        # Handle observation.state column
        if "observation.state" in combined_df.columns:
            output_df["observation.state"] = combined_df["observation.state"].values
        else:
            # Reconstruct from expanded columns or use action
            state_arrays = []
            for _, row in combined_df.iterrows():
                arr = []
                for name in action_names:
                    col_name = f"observation.state_{name}"
                    if col_name in combined_df.columns:
                        arr.append(float(row[col_name]))
                    elif f"action_{name}" in combined_df.columns:
                        arr.append(float(row[f"action_{name}"]))
                    else:
                        arr.append(0.0)
                state_arrays.append(np.array(arr, dtype=np.float32))
            output_df["observation.state"] = state_arrays
        
        # Save parquet with proper column order
        output_cols = ["action", "observation.state", "timestamp", "frame_index", 
                       "episode_index", "index", "task_index"]
        output_df = output_df[output_cols]
        
        parquet_path = data_dir / "file-000.parquet"
        output_df.to_parquet(parquet_path, index=False)
        
        # Create episodes metadata parquet
        episodes_df = pd.DataFrame({
            "episode_index": [0],
            "length": [len(output_df)],
            "meta/episodes/chunk_index": [0],
            "meta/episodes/file_index": [0],
            "task": [f"Write number {number}"],
            "task_index": [0]
        })
        episodes_path = episodes_dir / "file-000.parquet"
        episodes_df.to_parquet(episodes_path, index=False)
        
        # Create tasks.parquet
        tasks_df = pd.DataFrame({
            "task_index": [0],
            "task": [f"Write number {number}"]
        })
        tasks_df.to_parquet(meta_dir / "tasks.parquet", index=False)
        
        # Create info.json
        info = {
            "codebase_version": "v3.0",
            "robot_type": "so101_follower",
            "total_episodes": 1,
            "total_frames": len(output_df),
            "total_tasks": 1,
            "chunks_size": 1000,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 0,
            "fps": fps,
            "splits": {"train": "0:1"},
            "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
            "features": {
                "action": {
                    "dtype": "float32",
                    "names": action_names,
                    "shape": [6]
                },
                "observation.state": {
                    "dtype": "float32",
                    "names": action_names,
                    "shape": [6]
                },
                "timestamp": {"dtype": "float32", "shape": [1], "names": None},
                "frame_index": {"dtype": "int64", "shape": [1], "names": None},
                "episode_index": {"dtype": "int64", "shape": [1], "names": None},
                "index": {"dtype": "int64", "shape": [1], "names": None},
                "task_index": {"dtype": "int64", "shape": [1], "names": None}
            },
            "number_produced": number,
            "digits_sequence": [int(d) for d in str(number)]
        }
        
        with open(meta_dir / "info.json", "w") as f:
            json.dump(info, f, indent=4)
        
        # Create stats.json (minimal)
        stats = {}
        with open(meta_dir / "stats.json", "w") as f:
            json.dump(stats, f, indent=4)
        
        return info


class DigitMotionAgent:
    """OpenAI-powered agent for digit motion generation."""
    
    SYSTEM_PROMPT = """You are an AI assistant that helps users generate robot arm motion data for writing numbers.

You have access to pre-recorded motion data for digits 0-9, where each digit was recorded using an SO101 robot arm.

Your capabilities:
1. Explain what motion data is available for each digit
2. Help users generate combined motion sequences for any multi-digit number
3. Provide information about the motion characteristics

When a user asks to produce a number:
1. Parse the number into individual digits
2. Combine the motion sequences for those digits
4. Understand the inherent motion characteristics of the robot arm and the digits to understand the transalation and rotation required to go from one digit data to multiple digits generation.
3. Save the combined motion to a file that can be replayed on the robot

Available digit data summary:
{digit_summary}

When responding:
- Be helpful and explain what you're doing
- If a digit is missing, inform the user
- Provide details about the generated output
"""
    
    def __init__(
        self, 
        data_dir: str | Path,
        api_key: str | None = None,
        model: str = "gpt-4o"
    ):
        self.loader = DigitMotionLoader(data_dir)
        self.combiner = MotionCombiner(self.loader)
        self.data_dir = Path(data_dir)
        
        # Initialize OpenAI client
        api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise ValueError("OpenAI API key required. Set OPENAI_API_KEY env var or pass api_key.")
        
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.conversation_history = []
        
        # Initialize system prompt with digit summary
        self.system_prompt = self.SYSTEM_PROMPT.format(
            digit_summary=self.loader.get_digit_summary()
        )
    
    def _call_openai(self, user_message: str) -> str:
        """Call OpenAI API with conversation history."""
        self.conversation_history.append({
            "role": "user",
            "content": user_message
        })
        
        messages = [
            {"role": "system", "content": self.system_prompt}
        ] + self.conversation_history
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            tools=self._get_tools(),
            tool_choice="auto"
        )
        
        assistant_message = response.choices[0].message
        
        # Handle tool calls
        if assistant_message.tool_calls:
            tool_results = []
            for tool_call in assistant_message.tool_calls:
                result = self._execute_tool(
                    tool_call.function.name,
                    json.loads(tool_call.function.arguments)
                )
                tool_results.append({
                    "tool_call_id": tool_call.id,
                    "role": "tool",
                    "content": json.dumps(result)
                })
            
            # Add assistant message and tool results
            self.conversation_history.append({
                "role": "assistant",
                "content": assistant_message.content,
                "tool_calls": [
                    {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments
                        }
                    }
                    for tc in assistant_message.tool_calls
                ]
            })
            self.conversation_history.extend(tool_results)
            
            # Get final response
            messages = [
                {"role": "system", "content": self.system_prompt}
            ] + self.conversation_history
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages
            )
            final_content = response.choices[0].message.content
        else:
            final_content = assistant_message.content
        
        self.conversation_history.append({
            "role": "assistant",
            "content": final_content
        })
        
        return final_content
    
    def _get_tools(self) -> list[dict]:
        """Define tools available to the agent."""
        return [
            {
                "type": "function",
                "function": {
                    "name": "generate_number_motion",
                    "description": "Generate combined robot motion data for a multi-digit number",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "number": {
                                "type": "integer",
                                "description": "The number to generate motion for (e.g., 18, 42, 123)"
                            },
                            "pause_frames": {
                                "type": "integer",
                                "description": "Number of pause frames between digits (default: 30)",
                                "default": 30
                            }
                        },
                        "required": ["number"]
                    }
                }
            },
            {
                "type": "function", 
                "function": {
                    "name": "get_digit_info",
                    "description": "Get information about a specific digit's motion data",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "digit": {
                                "type": "integer",
                                "description": "The digit (0-9) to get info about"
                            }
                        },
                        "required": ["digit"]
                    }
                }
            },
            {
                "type": "function",
                "function": {
                    "name": "list_available_digits",
                    "description": "List all available digits with their motion data",
                    "parameters": {
                        "type": "object",
                        "properties": {}
                    }
                }
            }
        ]
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> dict:
        """Execute a tool call and return the result."""
        if tool_name == "generate_number_motion":
            return self._generate_number_motion(
                arguments["number"],
                arguments.get("pause_frames", 30)
            )
        elif tool_name == "get_digit_info":
            return self._get_digit_info(arguments["digit"])
        elif tool_name == "list_available_digits":
            return self._list_available_digits()
        else:
            return {"error": f"Unknown tool: {tool_name}"}
    
    def _generate_number_motion(self, number: int, pause_frames: int = 30) -> dict:
        """Generate motion data for a number."""
        digits = [int(d) for d in str(number)]
        
        # Check if all digits are available
        available = self.loader.get_available_digits()
        missing = [d for d in digits if d not in available]
        if missing:
            return {
                "success": False,
                "error": f"Missing motion data for digits: {missing}",
                "available_digits": available
            }
        
        try:
            # Combine digit motions
            combined_df = self.combiner.combine_digits(digits, pause_frames)
            
            # Save to output directory
            output_dir = self.data_dir / f"combined_number_{number}"
            info = self.combiner.save_combined_motion(
                combined_df, 
                output_dir, 
                number
            )
            
            return {
                "success": True,
                "number": number,
                "digits": digits,
                "total_frames": info["total_frames"],
                "duration_seconds": info["total_frames"] / info["fps"],
                "output_path": str(output_dir),
                "replay_command": f"python -c \"from lerobot.datasets.lerobot_dataset import LeRobotDataset; ds = LeRobotDataset(repo_id='local', root='{output_dir}')\""
            }
        except Exception as e:
            return {
                "success": False,
                "error": str(e)
            }
    
    def _get_digit_info(self, digit: int) -> dict:
        """Get information about a digit's motion data."""
        motion_data = self.loader.get_digit_motion(digit)
        if motion_data is None:
            return {
                "available": False,
                "digit": digit,
                "message": f"No motion data available for digit {digit}"
            }
        
        return {
            "available": True,
            "digit": digit,
            "folder": motion_data.folder_name,
            "total_frames": motion_data.total_frames,
            "fps": motion_data.fps,
            "duration_seconds": motion_data.total_frames / motion_data.fps,
            "description": motion_data.description
        }
    
    def _list_available_digits(self) -> dict:
        """List all available digits."""
        digits_info = []
        for digit in self.loader.get_available_digits():
            motion_data = self.loader.get_digit_motion(digit)
            digits_info.append({
                "digit": digit,
                "folder": motion_data.folder_name,
                "frames": motion_data.total_frames,
                "duration": motion_data.total_frames / motion_data.fps
            })
        
        return {
            "available_digits": self.loader.get_available_digits(),
            "details": digits_info
        }
    
    def chat(self, user_message: str) -> str:
        """Send a message to the agent and get a response."""
        return self._call_openai(user_message)
    
    def run_interactive(self):
        """Run an interactive chat session."""
        print("\n" + "="*60)
        print("🤖 Digit Motion Agent")
        print("="*60)
        print("\nI can help you generate robot motion data for writing numbers.")
        print("Available digits:", self.loader.get_available_digits())
        print("\nType 'quit' or 'exit' to end the session.")
        print("="*60 + "\n")
        
        while True:
            try:
                user_input = input("You: ").strip()
                if not user_input:
                    continue
                if user_input.lower() in ["quit", "exit", "q"]:
                    print("\nGoodbye! 👋")
                    break
                
                response = self.chat(user_input)
                print(f"\nAgent: {response}\n")
                
            except KeyboardInterrupt:
                print("\n\nGoodbye! 👋")
                break
            except Exception as e:
                print(f"\nError: {e}\n")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="OpenAI Agent for digit motion generation"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./shubhamt0802",
        help="Directory containing digit motion data folders (S101_x)"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o",
        help="OpenAI model to use"
    )
    parser.add_argument(
        "--number",
        type=int,
        default=None,
        help="Directly generate motion for this number (non-interactive mode)"
    )
    
    args = parser.parse_args()
    
    agent = DigitMotionAgent(
        data_dir=args.data_dir,
        api_key=args.api_key,
        model=args.model
    )
    
    if args.number is not None:
        # Non-interactive mode: directly generate the number
        response = agent.chat(f"Please generate motion data for the number {args.number}")
        print(response)
    else:
        # Interactive mode
        agent.run_interactive()


if __name__ == "__main__":
    main()

