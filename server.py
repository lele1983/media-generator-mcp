#!/usr/bin/env python3
"""
MCP Server for AI-powered Media Generation.

Generates images using OpenRouter (Nano Banana Pro / Gemini 3 Pro Image Preview)
and animates them into videos using fal.ai (Kling 2.6 Pro).
"""

import os
import json
import base64
import asyncio
import sys
import tempfile
from pathlib import Path
from datetime import datetime
from typing import Optional, Literal
from enum import Enum
from functools import wraps

import httpx
import fal_client
from pydantic import BaseModel, Field, ConfigDict
from mcp.server.fastmcp import FastMCP

# Initialize the MCP server
mcp = FastMCP("media_generator_mcp")

# ============================================================================
# Constants
# ============================================================================

OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
OPENROUTER_MODEL = "google/gemini-3-pro-image-preview"

FAL_MODEL_ID = "fal-ai/kling-video/v2.6/pro/image-to-video"

# Output directory for generated media
OUTPUT_DIR = Path(os.environ.get("MEDIA_OUTPUT_DIR", Path.home() / "media-generator-output"))
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================================
# Environment Variables
# ============================================================================

def get_openrouter_key() -> str:
    """Get OpenRouter API key from environment."""
    key = os.environ.get("OPENROUTER_API_KEY")
    if not key:
        raise ValueError("OPENROUTER_API_KEY environment variable is required")
    return key


def get_fal_key() -> str:
    """Get fal.ai API key from environment."""
    key = os.environ.get("FAL_KEY")
    if not key:
        raise ValueError("FAL_KEY environment variable is required")
    return key


def ensure_fal_auth():
    """Ensure fal.ai client is authenticated."""
    key = get_fal_key()
    # fal_client uses FAL_KEY env var automatically
    os.environ["FAL_KEY"] = key


# ============================================================================
# Enums
# ============================================================================

class AspectRatio(str, Enum):
    """Supported aspect ratios for image generation."""
    SQUARE = "1:1"
    LANDSCAPE = "16:9"
    PORTRAIT = "9:16"
    STANDARD = "4:3"
    WIDE = "21:9"


class ImageSize(str, Enum):
    """Output image resolution."""
    STANDARD = "1K"
    HIGH = "2K"
    ULTRA = "4K"


class VideoDuration(int, Enum):
    """Video duration in seconds."""
    SHORT = 5
    LONG = 10


# ============================================================================
# Input Models
# ============================================================================

class ImageGenerateInput(BaseModel):
    """Input for generating an image from text."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    prompt: str = Field(
        ...,
        description="Detailed description of the image to generate. Be specific about subject, style, lighting, colors, composition.",
        min_length=3,
        max_length=2000
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="Aspect ratio: '1:1' (square), '16:9' (landscape), '9:16' (portrait), '4:3' (standard), '21:9' (wide)"
    )
    image_size: ImageSize = Field(
        default=ImageSize.HIGH,
        description="Output resolution: '1K' (standard), '2K' (high quality), '4K' (ultra high)"
    )
    style_prompt: Optional[str] = Field(
        default=None,
        description="Additional style instructions (e.g., 'photorealistic', 'oil painting', 'anime style')",
        max_length=500
    )


class ImageEditInput(BaseModel):
    """Input for editing an existing image."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    image_url: Optional[str] = Field(
        default=None,
        description="URL of the image to edit (provide either image_url or image_base64)",
        max_length=2000
    )
    image_base64: Optional[str] = Field(
        default=None,
        description="Base64-encoded image data (provide either image_url or image_base64)"
    )
    edit_instructions: str = Field(
        ...,
        description="Instructions for how to modify the image (e.g., 'change the sky to sunset', 'add a cat in the foreground')",
        min_length=3,
        max_length=1000
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.SQUARE,
        description="Aspect ratio for the output image"
    )
    image_size: ImageSize = Field(
        default=ImageSize.HIGH,
        description="Output resolution"
    )


class VideoAnimateInput(BaseModel):
    """Input for animating an image into a video."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    image_url: str = Field(
        ...,
        description="URL of the image to animate (must be publicly accessible)",
        min_length=10,
        max_length=2000
    )
    prompt: str = Field(
        ...,
        description="Description of the motion/animation (e.g., 'camera slowly zooms in', 'wind blowing through hair', 'water rippling')",
        min_length=3,
        max_length=1500
    )
    duration: VideoDuration = Field(
        default=VideoDuration.SHORT,
        description="Video duration: 5 or 10 seconds"
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate synchronized audio (ambient sounds, effects). Costs more but adds realism."
    )
    negative_prompt: Optional[str] = Field(
        default="blur, distort, low quality, jitter, artifacts",
        description="What to avoid in the video",
        max_length=500
    )


class MediaGenerateInput(BaseModel):
    """Input for the complete text-to-video workflow."""
    model_config = ConfigDict(str_strip_whitespace=True, validate_assignment=True, extra='forbid')

    prompt: str = Field(
        ...,
        description="Description of the scene to generate as an image and then animate",
        min_length=3,
        max_length=2000
    )
    animation_prompt: Optional[str] = Field(
        default=None,
        description="Specific animation instructions (if different from main prompt). If not provided, will auto-generate based on the scene.",
        max_length=1500
    )
    duration: VideoDuration = Field(
        default=VideoDuration.SHORT,
        description="Video duration: 5 or 10 seconds"
    )
    generate_audio: bool = Field(
        default=True,
        description="Generate synchronized audio"
    )
    aspect_ratio: AspectRatio = Field(
        default=AspectRatio.LANDSCAPE,
        description="Aspect ratio for the generated media"
    )
    image_size: ImageSize = Field(
        default=ImageSize.HIGH,
        description="Image resolution (affects video quality)"
    )
    style_prompt: Optional[str] = Field(
        default=None,
        description="Style instructions for the image",
        max_length=500
    )


# ============================================================================
# Utility Functions
# ============================================================================

def with_retry(retries: int = 3, delay: int = 10):
    """Decorator to retry async functions with delay."""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            last_exception = None
            for attempt in range(retries + 1):
                try:
                    return await func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    if attempt < retries:
                        print(f"[RETRY] Error in {func.__name__}: {e}. Retrying in {delay}s... (Attempt {attempt + 1}/{retries})", file=sys.stderr)
                        await asyncio.sleep(delay)
            raise last_exception
        return wrapper
    return decorator


def _handle_error(e: Exception, context: str = "") -> str:
    """Format errors consistently."""
    prefix = f"[{context}] " if context else ""

    if isinstance(e, httpx.HTTPStatusError):
        status = e.response.status_code
        if status == 401:
            return f"{prefix}Error: Authentication failed. Check your API keys."
        elif status == 403:
            return f"{prefix}Error: Access forbidden. Check API permissions."
        elif status == 404:
            return f"{prefix}Error: Resource not found."
        elif status == 429:
            return f"{prefix}Error: Rate limit exceeded. Please wait before retrying."
        elif status == 402:
            return f"{prefix}Error: Insufficient credits. Please add funds to your account."
        return f"{prefix}Error: API request failed (HTTP {status})"

    elif isinstance(e, httpx.TimeoutException):
        return f"{prefix}Error: Request timed out. Try again or use a simpler prompt."

    elif isinstance(e, ValueError):
        return f"{prefix}Error: {str(e)}"

    return f"{prefix}Error: {type(e).__name__} - {str(e)}"


# ============================================================================
# OpenRouter API Functions
# ============================================================================

@with_retry(retries=3, delay=10)
async def generate_image_openrouter(
    prompt: str,
    aspect_ratio: str = "1:1",
    image_size: str = "2K",
    existing_image_b64: Optional[str] = None,
    existing_image_url: Optional[str] = None
) -> dict:
    """
    Generate or edit an image using OpenRouter (Nano Banana Pro / Gemini 3 Pro).

    Returns dict with:
        - image_base64: The generated image as base64
        - description: AI description of the image
        - model: Model used
    """
    api_key = get_openrouter_key()

    # Build message content
    content_parts = []

    # If editing, include the existing image first
    if existing_image_b64:
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{existing_image_b64}"
            }
        })
    elif existing_image_url:
        content_parts.append({
            "type": "image_url",
            "image_url": {
                "url": existing_image_url
            }
        })

    # Add the text prompt
    content_parts.append({
        "type": "text",
        "text": prompt
    })

    request_body = {
        "model": OPENROUTER_MODEL,
        "messages": [
            {
                "role": "user",
                "content": content_parts if len(content_parts) > 1 else prompt
            }
        ],
        "modalities": ["image", "text"],
        "image_config": {
            "aspect_ratio": aspect_ratio,
            "image_size": image_size
        },
        "temperature": 1,
        "max_tokens": 1024
    }

    headers = {
        "Authorization": f"Bearer {api_key}",
        "HTTP-Referer": "https://media-generator-mcp.local",
        "X-Title": "Media Generator MCP",
        "Content-Type": "application/json"
    }

    async with httpx.AsyncClient(timeout=180) as client:
        response = await client.post(
            f"{OPENROUTER_API_BASE}/chat/completions",
            json=request_body,
            headers=headers
        )

        if response.status_code != 200:
            error_text = response.text
            try:
                error_json = response.json()
                if "error" in error_json:
                    error_text = json.dumps(error_json["error"])
            except:
                pass
            raise Exception(f"OpenRouter API Error ({response.status_code}): {error_text}")

        result = response.json()
        choices = result.get("choices", [])
        if not choices:
            raise Exception("No response from OpenRouter")

        message = choices[0].get("message", {})
        description = message.get("content", "")
        images = message.get("images", [])

        if not images:
            raise Exception("No image generated. The model may have refused or encountered an error.")

        # Extract base64 from data URL
        image_data_url = images[0].get("image_url", {}).get("url", "")
        if "base64," in image_data_url:
            image_b64 = image_data_url.split("base64,")[1]
        else:
            image_b64 = image_data_url

        return {
            "image_base64": image_b64,
            "description": description,
            "model": OPENROUTER_MODEL,
            "aspect_ratio": aspect_ratio,
            "image_size": image_size
        }


# ============================================================================
# fal.ai API Functions (using official SDK)
# ============================================================================

async def upload_image_to_fal(image_path: str) -> str:
    """
    Upload an image file to fal.ai CDN.

    Returns the public URL of the uploaded image.
    """
    ensure_fal_auth()

    # fal_client.upload_file is synchronous, run in executor
    loop = asyncio.get_event_loop()
    url = await loop.run_in_executor(None, fal_client.upload_file, image_path)
    return url


async def upload_image_bytes_to_fal(image_bytes: bytes, filename: str = "image.png") -> str:
    """
    Upload image bytes to fal.ai CDN by first saving to temp file.

    Returns the public URL of the uploaded image.
    """
    # Save to temp file first
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as f:
        f.write(image_bytes)
        temp_path = f.name

    try:
        url = await upload_image_to_fal(temp_path)
        return url
    finally:
        # Clean up temp file
        try:
            os.unlink(temp_path)
        except:
            pass


@with_retry(retries=2, delay=5)
async def animate_image_fal(
    image_url: str,
    prompt: str,
    duration: int = 5,
    generate_audio: bool = True,
    negative_prompt: str = "blur, distort, low quality",
    max_wait_seconds: int = 300
) -> dict:
    """
    Complete workflow to animate an image using fal.ai Kling 2.6 Pro.

    Uses the official fal_client SDK for reliable async operation.

    Returns dict with video details.
    """
    ensure_fal_auth()

    arguments = {
        "prompt": prompt,
        "image_url": image_url,
        "duration": str(duration),  # fal.ai expects string for duration
        "aspect_ratio": "16:9",  # default aspect ratio
    }

    # Add audio generation if requested
    if generate_audio:
        arguments["with_audio"] = True

    if negative_prompt:
        arguments["negative_prompt"] = negative_prompt

    # Use fal_client.subscribe for async operation with status updates
    loop = asyncio.get_event_loop()

    def run_fal_sync():
        """Run fal.ai request synchronously (will be run in executor)."""
        result = fal_client.subscribe(
            FAL_MODEL_ID,
            arguments=arguments,
            with_logs=True,
            on_queue_update=lambda update: print(f"[fal.ai] Queue update: {update}", file=sys.stderr)
        )
        return result

    # Run in executor to not block the event loop
    result = await loop.run_in_executor(None, run_fal_sync)

    return result


# ============================================================================
# File Storage Helpers
# ============================================================================

def generate_filename(prefix: str, extension: str) -> str:
    """Generate a unique filename with timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


async def save_image_to_file(image_base64: str, filename: Optional[str] = None) -> str:
    """
    Save a base64 image to the output directory.

    Returns the full path to the saved file.
    """
    if not filename:
        filename = generate_filename("image", "png")

    filepath = OUTPUT_DIR / filename

    # Decode and save
    image_bytes = base64.b64decode(image_base64)
    filepath.write_bytes(image_bytes)

    return str(filepath)


async def download_video_to_file(video_url: str, filename: Optional[str] = None) -> str:
    """
    Download a video from URL and save to the output directory.

    Returns the full path to the saved file.
    """
    if not filename:
        filename = generate_filename("video", "mp4")

    filepath = OUTPUT_DIR / filename

    async with httpx.AsyncClient(timeout=120, follow_redirects=True) as client:
        response = await client.get(video_url)
        response.raise_for_status()
        filepath.write_bytes(response.content)

    return str(filepath)


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool(
    name="image_generate",
    annotations={
        "title": "Generate Image from Text",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def image_generate(params: ImageGenerateInput) -> str:
    """Generate an image from a text description using AI (Nano Banana Pro / Gemini 3 Pro).

    This tool creates high-quality images from detailed text prompts.
    It excels at photorealistic images, artistic styles, text rendering, and complex compositions.

    Args:
        params (ImageGenerateInput): Input containing:
            - prompt (str): Detailed description of the image
            - aspect_ratio (AspectRatio): '1:1', '16:9', '9:16', '4:3', '21:9'
            - image_size (ImageSize): '1K', '2K', '4K'
            - style_prompt (Optional[str]): Additional style instructions

    Returns:
        str: JSON with:
            - status: 'success' or 'error'
            - image_path: Local file path to the generated PNG image
            - description: AI description of the generated image
            - metadata: aspect_ratio, image_size, model

    Example:
        prompt: "A serene Japanese garden with a koi pond, cherry blossoms falling, golden hour lighting"
        style_prompt: "photorealistic, 8K, detailed"
    """
    try:
        # Build the full prompt
        full_prompt = f"Generate an image: {params.prompt}"
        if params.style_prompt:
            full_prompt += f"\n\nStyle: {params.style_prompt}"

        result = await generate_image_openrouter(
            prompt=full_prompt,
            aspect_ratio=params.aspect_ratio.value,
            image_size=params.image_size.value
        )

        # Save image to file instead of returning base64
        image_path = await save_image_to_file(result["image_base64"])

        response = {
            "status": "success",
            "image_path": image_path,
            "description": result["description"],
            "metadata": {
                "aspect_ratio": result["aspect_ratio"],
                "image_size": result["image_size"],
                "model": result["model"],
                "output_dir": str(OUTPUT_DIR)
            }
        }

        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        return _handle_error(e, "Image Generation")


@mcp.tool(
    name="image_edit",
    annotations={
        "title": "Edit Existing Image",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def image_edit(params: ImageEditInput) -> str:
    """Edit an existing image based on text instructions using AI.

    This tool modifies an existing image according to your instructions.
    You can change colors, add/remove elements, adjust lighting, transform style, etc.

    Args:
        params (ImageEditInput): Input containing:
            - image_url (Optional[str]): URL of the image to edit
            - image_base64 (Optional[str]): Base64-encoded image data
            - edit_instructions (str): What to change in the image
            - aspect_ratio (AspectRatio): Output aspect ratio
            - image_size (ImageSize): Output resolution

    Returns:
        str: JSON with edited image path and metadata

    Example:
        image_url: "https://example.com/photo.jpg"
        edit_instructions: "Change the sky to a dramatic sunset with orange and purple colors"
    """
    try:
        # Validate input
        if not params.image_url and not params.image_base64:
            return "Error: You must provide either image_url or image_base64"

        # Build the edit prompt
        edit_prompt = f"Edit this image: {params.edit_instructions}"

        result = await generate_image_openrouter(
            prompt=edit_prompt,
            aspect_ratio=params.aspect_ratio.value,
            image_size=params.image_size.value,
            existing_image_b64=params.image_base64,
            existing_image_url=params.image_url
        )

        # Save image to file
        image_path = await save_image_to_file(result["image_base64"])

        response = {
            "status": "success",
            "image_path": image_path,
            "description": result["description"],
            "edit_applied": params.edit_instructions,
            "metadata": {
                "aspect_ratio": result["aspect_ratio"],
                "image_size": result["image_size"],
                "model": result["model"],
                "output_dir": str(OUTPUT_DIR)
            }
        }

        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        return _handle_error(e, "Image Edit")


@mcp.tool(
    name="video_animate",
    annotations={
        "title": "Animate Image to Video",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def video_animate(params: VideoAnimateInput) -> str:
    """Animate a static image into a video using AI (Kling 2.6 Pro).

    This tool takes an image and brings it to life with realistic motion.
    It can add camera movements, subject animations, environmental effects, and synchronized audio.

    Args:
        params (VideoAnimateInput): Input containing:
            - image_url (str): URL of the image to animate (must be publicly accessible)
            - prompt (str): Description of the desired motion/animation
            - duration (VideoDuration): 5 or 10 seconds
            - generate_audio (bool): Add synchronized audio (ambient sounds, effects)
            - negative_prompt (Optional[str]): What to avoid in the video

    Returns:
        str: JSON with:
            - status: 'success' or 'error'
            - video_url: Direct URL to the MP4 video (hosted on fal.ai)
            - video_path: Local file path where video is saved
            - video_metadata: file_name, file_size, content_type
            - animation_details: duration, audio_enabled

    Example:
        image_url: "https://example.com/landscape.jpg"
        prompt: "Gentle camera pan from left to right, clouds moving slowly, birds flying in the distance"
        duration: 5
        generate_audio: true

    Note:
        - Video generation takes 30-120 seconds depending on duration and queue
        - Costs: ~$0.35 for 5s without audio, ~$0.70 for 5s with audio
    """
    try:
        result = await animate_image_fal(
            image_url=params.image_url,
            prompt=params.prompt,
            duration=params.duration.value,
            generate_audio=params.generate_audio,
            negative_prompt=params.negative_prompt or "blur, distort, low quality"
        )

        video_info = result.get("video", {})
        video_url = video_info.get("url")

        # Download video to local file
        video_path = None
        if video_url:
            try:
                video_path = await download_video_to_file(video_url)
            except Exception as e:
                print(f"[WARNING] Could not download video locally: {e}", file=sys.stderr)

        response = {
            "status": "success",
            "video_url": video_url,
            "video_path": video_path,
            "video_metadata": {
                "file_name": video_info.get("file_name", "output.mp4"),
                "file_size": video_info.get("file_size"),
                "content_type": video_info.get("content_type", "video/mp4")
            },
            "animation_details": {
                "duration_seconds": params.duration.value,
                "audio_enabled": params.generate_audio,
                "prompt": params.prompt
            },
            "request_id": result.get("request_id"),
            "output_dir": str(OUTPUT_DIR)
        }

        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        return _handle_error(e, "Video Animation")


@mcp.tool(
    name="media_generate",
    annotations={
        "title": "Generate Video from Text (Full Workflow)",
        "readOnlyHint": False,
        "destructiveHint": False,
        "idempotentHint": False,
        "openWorldHint": True
    }
)
async def media_generate(params: MediaGenerateInput) -> str:
    """Complete text-to-video workflow: Generate an image and animate it into a video.

    This is a convenience tool that combines image generation (Nano Banana Pro)
    and video animation (Kling 2.6) in a single call.

    The workflow:
    1. Generate a high-quality image from your text description
    2. Upload the image to fal.ai CDN
    3. Animate the image into a video with motion and optional audio
    4. Download and save both image and video locally

    Args:
        params (MediaGenerateInput): Input containing:
            - prompt (str): Description of the scene to create
            - animation_prompt (Optional[str]): Specific animation instructions
            - duration (VideoDuration): 5 or 10 seconds
            - generate_audio (bool): Add synchronized audio
            - aspect_ratio (AspectRatio): Output aspect ratio
            - image_size (ImageSize): Image quality (affects video)
            - style_prompt (Optional[str]): Style instructions

    Returns:
        str: JSON with:
            - status: 'success' or 'error'
            - image_path: Local path to the generated image
            - video_url: URL to the animated video (hosted on fal.ai)
            - video_path: Local path to the downloaded video
            - metadata: Full generation details

    Example:
        prompt: "A cozy cabin in a snowy forest at night, warm light glowing from windows, smoke rising from chimney"
        animation_prompt: "Snow falling gently, smoke drifting from chimney, subtle twinkling of stars"
        duration: 10
        generate_audio: true

    Note:
        Total generation time: 60-180 seconds
        Costs: Image (~$0.02) + Video (~$0.35-1.40 depending on options)
    """
    try:
        # Step 1: Generate the image
        full_prompt = f"Generate an image: {params.prompt}"
        if params.style_prompt:
            full_prompt += f"\n\nStyle: {params.style_prompt}"

        image_result = await generate_image_openrouter(
            prompt=full_prompt,
            aspect_ratio=params.aspect_ratio.value,
            image_size=params.image_size.value
        )

        image_base64 = image_result["image_base64"]

        # Step 2: Save image locally
        image_path = await save_image_to_file(image_base64)

        # Step 3: Upload image to fal.ai CDN for video generation
        print("[INFO] Uploading image to fal.ai CDN...", file=sys.stderr)
        image_bytes = base64.b64decode(image_base64)
        fal_image_url = await upload_image_bytes_to_fal(image_bytes)
        print(f"[INFO] Image uploaded: {fal_image_url[:50]}...", file=sys.stderr)

        # Step 4: Generate animation prompt if not provided
        animation_prompt = params.animation_prompt
        if not animation_prompt:
            # Create a sensible default animation based on the scene
            animation_prompt = f"Subtle cinematic movement bringing the scene to life: {params.prompt[:200]}"

        # Step 5: Animate the image
        print("[INFO] Starting video animation with Kling 2.6...", file=sys.stderr)
        video_result = await animate_image_fal(
            image_url=fal_image_url,
            prompt=animation_prompt,
            duration=params.duration.value,
            generate_audio=params.generate_audio,
            negative_prompt="blur, distort, low quality, jitter"
        )

        video_info = video_result.get("video", {})
        video_url = video_info.get("url")

        # Step 6: Download video locally
        video_path = None
        if video_url:
            try:
                video_path = await download_video_to_file(video_url)
                print(f"[INFO] Video saved to: {video_path}", file=sys.stderr)
            except Exception as e:
                print(f"[WARNING] Could not download video locally: {e}", file=sys.stderr)

        response = {
            "status": "success",
            "image_path": image_path,
            "image_description": image_result.get("description", ""),
            "video_url": video_url,
            "video_path": video_path,
            "video_metadata": {
                "file_name": video_info.get("file_name", "output.mp4"),
                "file_size": video_info.get("file_size"),
                "content_type": video_info.get("content_type", "video/mp4"),
                "duration_seconds": params.duration.value,
                "audio_enabled": params.generate_audio
            },
            "generation_details": {
                "scene_prompt": params.prompt,
                "animation_prompt": animation_prompt,
                "style_prompt": params.style_prompt,
                "aspect_ratio": params.aspect_ratio.value,
                "image_size": params.image_size.value,
                "image_model": OPENROUTER_MODEL,
                "video_model": FAL_MODEL_ID
            },
            "request_id": video_result.get("request_id"),
            "output_dir": str(OUTPUT_DIR)
        }

        return json.dumps(response, indent=2, ensure_ascii=False)

    except Exception as e:
        return _handle_error(e, "Media Generation")


# ============================================================================
# Server Entry Point
# ============================================================================

def main():
    """Entry point for the MCP server."""
    mcp.run()


if __name__ == "__main__":
    main()
