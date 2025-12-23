#!/usr/bin/env python3
"""
Waverse Renderer Benchmark - Simplified

Runs the game in automated mode and measures frame times.
"""

import subprocess
import sys
import time
import re


def run_benchmark(use_modern: bool, duration_sec: int = 30):
    """
    Run the game and collect performance stats.
    
    Returns average FPS if successful, None otherwise.
    """
    cmd = ['python', 'main.py']
    if use_modern:
        cmd.append('--modern')
    
    renderer_name = "Modern" if use_modern else "Legacy"
    print(f"\n  Running {renderer_name} renderer for {duration_sec} seconds...")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  (Press ESC in the game window to stop early)")
    
    try:
        # Run game with timeout
        result = subprocess.run(
            cmd,
            timeout=duration_sec,
            capture_output=True,
            text=True,
            cwd='/Users/kenny/code/waverse'
        )
        output = result.stdout + result.stderr
    except subprocess.TimeoutExpired:
        output = "Timeout - normal termination"
    except Exception as e:
        print(f"  Error: {e}")
        return None
    
    # Parse output for performance data (if any was printed)
    # Look for FPS numbers in output
    fps_matches = re.findall(r'(\d+(?:\.\d+)?)\s*(?:FPS|fps)', output)
    if fps_matches:
        avg_fps = sum(float(f) for f in fps_matches) / len(fps_matches)
        return avg_fps
    
    return None


def quick_gl_test():
    """Quick test to see what OpenGL version is available."""
    try:
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        
        pygame.init()
        screen = pygame.display.set_mode((100, 100), DOUBLEBUF | OPENGL)
        
        from OpenGL.GL import glGetString, GL_VERSION, GL_RENDERER
        version = glGetString(GL_VERSION)
        renderer = glGetString(GL_RENDERER)
        
        pygame.quit()
        
        return version.decode() if version else "Unknown", renderer.decode() if renderer else "Unknown"
    except Exception as e:
        return f"Error: {e}", "Unknown"


def moderngl_test():
    """Test if ModernGL works."""
    try:
        import pygame
        from pygame.locals import DOUBLEBUF, OPENGL
        import moderngl
        
        pygame.init()
        screen = pygame.display.set_mode((100, 100), DOUBLEBUF | OPENGL)
        
        ctx = moderngl.create_context()
        version = ctx.version_code
        
        pygame.quit()
        
        return version
    except Exception as e:
        return f"Error: {e}"


if __name__ == '__main__':
    print("\n" + "="*60)
    print("  WAVERSE RENDERER BENCHMARK")
    print("="*60)
    
    # System info
    print("\n  System Information:")
    gl_version, gl_renderer = quick_gl_test()
    print(f"    OpenGL Version: {gl_version}")
    print(f"    GPU: {gl_renderer}")
    
    mgl_version = moderngl_test()
    print(f"    ModernGL Context: {mgl_version}")
    
    # Note about manual testing
    print("\n" + "="*60)
    print("  MANUAL BENCHMARK INSTRUCTIONS")
    print("="*60)
    print("""
  Since automated benchmarking requires display access, please run
  these commands manually and observe the FPS counter in-game:

  LEGACY RENDERER:
    python main.py
    
  MODERN RENDERER:
    python main.py --modern
    
  The game shows FPS in the top-left corner. Walk around for 30+
  seconds to get a good average. The modern renderer should show:
  
  - Higher FPS (especially with many plants/animals visible)
  - Smoother frame times (less stuttering when loading chunks)
  - Better performance when looking at distant objects
  
  If ModernGL fails on your system (context version: Error), the
  game will automatically fall back to legacy rendering.
""")
    
    print("  Press Enter to exit...")
    input()
