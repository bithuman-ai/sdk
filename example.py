import subprocess
import sys
import argparse
import asyncio
import os
import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from bithuman import AsyncBithuman, VideoFrame
from bithuman.audio import float32_to_int16, load_audio
from bithuman.utils import FPSController

# Package installation handling
def install_required_packages():
    def install_package(package):
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

    required_packages = ['sounddevice', 'aiohttp']
    for package in required_packages:
        try:
            __import__(package.replace('-', '_'))
        except ImportError:
            logger.info(f"Installing {package}...")
            install_package(package)

# Initialize sounddevice
try:
    import sounddevice as sd
except ImportError:
    logger.warning("sounddevice is not installed. Audio will not be played.")
    sd = None

# Configure logger
logger.remove()
logger.add(sys.stdout, level="INFO")


class AudioPlayer:
    """Audio player that uses a buffer and callback for smooth playback."""

    def __init__(self, sample_rate: int = 16000, block_per_second: int = 25):
        self.sample_rate = sample_rate
        self.output_buf = bytearray()
        self.output_lock = threading.Lock()
        self.stream = None
        self.blocksize = self.sample_rate // block_per_second

    def is_started(self) -> bool:
        return self.stream is not None

    def start(self) -> bool:
        if sd is None:
            return False
        self.stream = sd.OutputStream(
            callback=self.output_callback,
            dtype="int16",
            channels=1,
            samplerate=self.sample_rate,
            blocksize=self.blocksize,
        )
        self.stream.start()
        return True

    def stop(self):
        if self.stream is not None:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def add_audio(self, audio_data):
        if isinstance(audio_data, np.ndarray) and audio_data.dtype == np.float32:
            audio_data = (audio_data * 32768.0).astype(np.int16)

        with self.output_lock:
            if isinstance(audio_data, (bytes, bytearray)):
                self.output_buf.extend(audio_data)
            else:
                self.output_buf.extend(audio_data.tobytes())

    def output_callback(self, outdata, frames, time, status):
        with self.output_lock:
            bytes_needed = frames * 2
            if len(self.output_buf) < bytes_needed:
                available_bytes = len(self.output_buf)
                outdata[: available_bytes // 2, 0] = np.frombuffer(
                    self.output_buf, dtype=np.int16, count=available_bytes // 2
                )
                outdata[available_bytes // 2 :, 0] = 0
                del self.output_buf[:available_bytes]
            else:
                chunk = self.output_buf[:bytes_needed]
                outdata[:, 0] = np.frombuffer(chunk, dtype=np.int16, count=frames)
                del self.output_buf[:bytes_needed]


class VideoPlayer:
    """Video player for displaying frames with performance metrics."""

    def __init__(self, window_size: Tuple[int, int], window_name: str = "Bithuman"):
        self.window_name = window_name
        self.start_time = None
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.window_name, window_size[0], window_size[1])

    def start(self):
        self.start_time = asyncio.get_event_loop().time()

    def stop(self):
        cv2.destroyAllWindows()

    async def display_frame(
        self, frame: VideoFrame, fps: float = 0.0, exp_time: float = 0.0
    ) -> int:
        if not frame.has_image:
            await asyncio.sleep(0.01)
            return -1

        image = await self.render_image(frame, fps, exp_time)
        cv2.imshow(self.window_name, image)
        return cv2.waitKey(1) & 0xFF

    async def render_image(
        self, frame: VideoFrame, fps: float = 0.0, exp_time: float = 0.0
    ) -> np.ndarray:
        image = frame.bgr_image.copy()
        self._add_performance_metrics(image, fps, exp_time)
        return image

    def _add_performance_metrics(self, image: np.ndarray, fps: float, exp_time: float):
        cv2.putText(
            image,
            f"FPS: {fps:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
        )

        if self.start_time is not None:
            elapsed = asyncio.get_event_loop().time() - self.start_time
            cv2.putText(
                image,
                f"Time: {elapsed:.1f}s",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )

        if exp_time > 0:
            exp_in_seconds = exp_time - time.time()
            cv2.putText(
                image,
                f"Exp in: {exp_in_seconds:.1f}s",
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
            )


async def push_audio(
    runtime: AsyncBithuman, audio_file: str, delay: float = 0.0
) -> None:
    """Stream audio from file to runtime with specified delay."""
    logger.info(f"Pushing audio file: {audio_file}")
    audio_np, sr = load_audio(audio_file)
    audio_np = float32_to_int16(audio_np)

    await asyncio.sleep(delay)
    chunk_size = sr // 100
    for i in range(0, len(audio_np), chunk_size):
        chunk = audio_np[i : i + chunk_size]
        await runtime.push_audio(chunk.tobytes(), sr, last_chunk=False)

    await runtime.flush()


async def run_bithuman(
    runtime: AsyncBithuman, audio_file: Optional[str] = None
) -> None:
    """Main runtime loop for Bithuman with audio and video processing."""
    audio_player = AudioPlayer()
    video_player = VideoPlayer(window_size=runtime.get_frame_size())
    fps_controller = FPSController(target_fps=25)

    audio_player.start()
    video_player.start()
    await runtime.start()

    push_audio_task = None
    if audio_file:
        push_audio_task = asyncio.create_task(
            push_audio(runtime, audio_file, delay=1.0)
        )

    try:
        async for frame in runtime.run():
            sleep_time = fps_controller.wait_next_frame(sleep=False)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

            exp_time = runtime.get_expiration_time()
            key = await video_player.display_frame(
                frame, fps_controller.average_fps, exp_time
            )

            if frame.audio_chunk and audio_player.is_started():
                audio_player.add_audio(frame.audio_chunk.array)

            if await handle_key_press(key, audio_file, push_audio_task, runtime):
                break

            fps_controller.update()

    except asyncio.CancelledError:
        logger.info("Runtime task cancelled")
    finally:
        await cleanup(push_audio_task, audio_player, video_player, runtime)


async def handle_key_press(key, audio_file, push_audio_task, runtime) -> bool:
    """Handle keyboard input during runtime."""
    if key == ord("1") and audio_file:
        if push_audio_task and not push_audio_task.done():
            logger.info("Cancelling previous push_audio task")
            push_audio_task.cancel()
        push_audio_task = asyncio.create_task(push_audio(runtime, audio_file))
    elif key == ord("2"):
        logger.info("Interrupting")
        if push_audio_task and not push_audio_task.done():
            push_audio_task.cancel()
        runtime.interrupt()
    elif key == ord("q"):
        logger.info("Exiting...")
        return True
    return False


async def cleanup(push_audio_task, audio_player, video_player, runtime):
    """Cleanup resources before exit."""
    if push_audio_task and not push_audio_task.done():
        push_audio_task.cancel()
    audio_player.stop()
    video_player.stop()
    await runtime.stop()


async def main(args: argparse.Namespace) -> None:
    """Application entry point."""
    logger.info(f"Initializing AsyncBithuman with model: {args.model}")
    
    runtime = await AsyncBithuman.create(
        model_path=args.model,
        api_secret=args.api_secret,
        insecure=args.insecure,
    )
    
    try:
        frame_size = runtime.get_frame_size()
        logger.info(f"Model initialized successfully, frame size: {frame_size}")
    except Exception as e:
        logger.error(f"Model initialization verification failed: {e}")
        raise
    
    logger.info("Starting runtime...")
    await run_bithuman(runtime, args.audio_file)


def parse_arguments():
    """Parse and validate command line arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--audio-file", type=str, default=None)
    parser.add_argument(
        "--api-secret",
        type=str,
        default=os.environ.get("BITHUMAN_API_SECRET"),
        help="API Secret for API authentication",
    )
    parser.add_argument(
        "--insecure",
        action="store_true",
        help="Disable SSL certificate verification (not recommended for production use)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    install_required_packages()
    args = parse_arguments()
    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        logger.info("Interrupted by user")
