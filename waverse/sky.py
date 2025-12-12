"""
Dynamic Sky System - Day/Night cycle with moving sun and moon.

The sun and moon follow orbital paths that shift over time, so sunrise/sunset
positions change gradually like real seasons.
"""

import math
from dataclasses import dataclass
from OpenGL.GL import *
from OpenGL.GLU import *


@dataclass
class TimeOfDay:
    """Time periods with their characteristics."""
    MIDNIGHT = 0.0      # 00:00
    EARLY_MORNING = 0.2  # 04:48
    DAWN = 0.25         # 06:00
    MORNING = 0.33      # 08:00
    NOON = 0.5          # 12:00
    AFTERNOON = 0.625   # 15:00
    EVENING = 0.75      # 18:00
    DUSK = 0.8          # 19:12
    NIGHT = 0.875       # 21:00


class SkySystem:
    """Manages day/night cycle, sun, moon, and sky colors."""
    
    # Day length in seconds (game time)
    DAY_LENGTH = 120.0  # 2 minutes per full day/night cycle (faster for testing)
    
    # Orbital parameters
    BASE_ORBIT_TILT = 23.5  # Base tilt in degrees (like Earth)
    ORBIT_SHIFT_PERIOD = 20.0  # Days for full orbital shift cycle
    
    def __init__(self):
        self.time = 0.25  # Start at dawn (0-1, 0=midnight)
        self.day_count = 0
        self.accumulated_time = 0.0
        
        # Sun/moon display lists
        self.sun_list = None
        self.moon_list = None
        self.stars_list = None
        
    def init_gl(self):
        """Create display lists for celestial bodies."""
        # Sun - warm orange with layered glow
        self.sun_list = glGenLists(1)
        glNewList(self.sun_list, GL_COMPILE)
        glDisable(GL_LIGHTING)
        # Core - bright yellow-white
        glColor3f(1.0, 0.98, 0.85)
        self._draw_sphere(12, 20)
        # Inner sun - orange
        glColor3f(1.0, 0.75, 0.3)
        self._draw_sphere(15, 18)
        # Outer glow layers
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive blending for glow
        glColor4f(1.0, 0.6, 0.2, 0.4)
        self._draw_sphere(22, 14)
        glColor4f(1.0, 0.5, 0.1, 0.25)
        self._draw_sphere(32, 12)
        glColor4f(1.0, 0.4, 0.05, 0.12)
        self._draw_sphere(45, 10)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_LIGHTING)
        glEndList()
        
        # Moon - pale white sphere with craters
        self.moon_list = glGenLists(1)
        glNewList(self.moon_list, GL_COMPILE)
        glColor3f(0.9, 0.9, 0.95)
        self._draw_sphere(10, 16)
        # Crater marks
        glColor3f(0.7, 0.7, 0.75)
        glPushMatrix()
        glTranslatef(3, 2, 8)
        self._draw_sphere(2, 8)
        glPopMatrix()
        glPushMatrix()
        glTranslatef(-2, -3, 9)
        self._draw_sphere(1.5, 8)
        glPopMatrix()
        glEndList()
        
        # Stars - multiple sizes and colors
        self.stars_list = glGenLists(1)
        glNewList(self.stars_list, GL_COMPILE)
        glDisable(GL_LIGHTING)
        
        import random
        random.seed(12345)
        
        # Bright stars (larger)
        glPointSize(3)
        glBegin(GL_POINTS)
        for _ in range(100):
            theta = random.random() * 2 * math.pi
            phi = random.random() * math.pi * 0.45
            r = 900
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.cos(phi) + 50  # Offset up
            z = r * math.sin(phi) * math.sin(theta)
            
            # Slight color variation (white, blue-white, yellow-white)
            tint = random.choice([(1, 1, 1), (0.8, 0.9, 1), (1, 0.95, 0.8)])
            glColor3f(*tint)
            glVertex3f(x, y, z)
        glEnd()
        
        # Medium stars
        glPointSize(2)
        glBegin(GL_POINTS)
        for _ in range(300):
            theta = random.random() * 2 * math.pi
            phi = random.random() * math.pi * 0.5
            r = 850
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.cos(phi) + 30
            z = r * math.sin(phi) * math.sin(theta)
            
            brightness = 0.6 + random.random() * 0.4
            glColor3f(brightness, brightness, brightness)
            glVertex3f(x, y, z)
        glEnd()
        
        # Dim stars (smallest)
        glPointSize(1)
        glBegin(GL_POINTS)
        for _ in range(600):
            theta = random.random() * 2 * math.pi
            phi = random.random() * math.pi * 0.55
            r = 800
            x = r * math.sin(phi) * math.cos(theta)
            y = r * math.cos(phi) + 20
            z = r * math.sin(phi) * math.sin(theta)
            
            brightness = 0.3 + random.random() * 0.4
            glColor3f(brightness, brightness, brightness)
            glVertex3f(x, y, z)
        glEnd()
        
        glEnable(GL_LIGHTING)
        glEndList()
    
    def _draw_sphere(self, radius: float, slices: int):
        """Draw a simple sphere using triangle fan."""
        for i in range(slices):
            lat0 = math.pi * (-0.5 + float(i) / slices)
            lat1 = math.pi * (-0.5 + float(i + 1) / slices)
            
            glBegin(GL_QUAD_STRIP)
            for j in range(slices + 1):
                lng = 2 * math.pi * float(j) / slices
                
                x0 = math.cos(lat0) * math.cos(lng)
                y0 = math.sin(lat0)
                z0 = math.cos(lat0) * math.sin(lng)
                
                x1 = math.cos(lat1) * math.cos(lng)
                y1 = math.sin(lat1)
                z1 = math.cos(lat1) * math.sin(lng)
                
                glVertex3f(x0 * radius, y0 * radius, z0 * radius)
                glVertex3f(x1 * radius, y1 * radius, z1 * radius)
            glEnd()
    
    def update(self, dt: float):
        """Update time of day."""
        # dt is frame-time normalized (~1.0 at 60fps)
        # Convert to seconds: dt/60 gives rough seconds
        seconds = dt / 60.0
        
        # Progress time (full cycle = DAY_LENGTH seconds)
        time_delta = seconds / self.DAY_LENGTH
        self.time += time_delta
        
        if self.time >= 1.0:
            self.time -= 1.0
            self.day_count += 1
    
    def get_sun_position(self) -> tuple:
        """Get sun position in sky (x, y, z) based on time and orbital shift."""
        # Sun rises at 0.25 (dawn), peaks at 0.5 (noon), sets at 0.75 (dusk)
        # Map time so sun is above horizon from 0.25 to 0.75
        
        # Orbital tilt shifts over days (creates seasonal variation)
        orbit_phase = (self.day_count / self.ORBIT_SHIFT_PERIOD) * 2 * math.pi
        tilt = math.radians(self.BASE_ORBIT_TILT) * math.sin(orbit_phase)
        azimuth_shift = math.sin(orbit_phase * 0.7) * 0.3
        
        r = 600  # Distance to sun
        
        # Sun arc: rises in east, peaks at noon, sets in west
        # time 0.25 = sunrise (east), 0.5 = noon (top), 0.75 = sunset (west)
        sun_progress = (self.time - 0.25) / 0.5  # 0 at sunrise, 1 at sunset
        sun_angle = sun_progress * math.pi  # 0 to PI arc
        
        # Position on arc
        x = r * math.cos(sun_angle + azimuth_shift)  # East to West
        y = r * math.sin(sun_angle)  # Up and down arc
        
        # Apply tilt for north-south seasonal variation  
        z = y * math.sin(tilt) * 0.3 + r * 0.2
        y = y * math.cos(tilt)
        
        return (x, y, z)
    
    def get_moon_position(self) -> tuple:
        """Get moon position - visible at night (opposite of sun)."""
        # Moon rises at 0.75 (dusk), peaks at 0.0 (midnight), sets at 0.25 (dawn)
        # Moon has its own orbital shift
        moon_orbit_phase = (self.day_count / (self.ORBIT_SHIFT_PERIOD * 1.3)) * 2 * math.pi
        moon_tilt = math.radians(self.BASE_ORBIT_TILT * 0.8) * math.cos(moon_orbit_phase)
        
        r = 500
        
        # Moon arc: rises at dusk, peaks at midnight, sets at dawn
        # Normalize time for moon: 0.75->0.25 maps to 0->1 (wrapping around midnight)
        if self.time >= 0.75:
            moon_progress = (self.time - 0.75) / 0.5  # 0.75 to 1.0 -> 0 to 0.5
        elif self.time <= 0.25:
            moon_progress = (self.time + 0.25) / 0.5  # 0 to 0.25 -> 0.5 to 1.0
        else:
            moon_progress = -1  # Moon below horizon during day
        
        if moon_progress < 0 or moon_progress > 1:
            return (0, -500, 0)  # Below horizon
        
        moon_angle = moon_progress * math.pi
        
        x = -r * math.cos(moon_angle)  # Opposite direction from sun
        y = r * math.sin(moon_angle)
        z = y * math.sin(moon_tilt) * 0.3 - r * 0.15
        y = y * math.cos(moon_tilt)
        
        return (x, y, z)
    
    def get_sky_color(self) -> tuple:
        """Get sky background color based on time of day."""
        t = self.time
        
        # Define key colors - vivid sunrise/sunset
        colors = {
            0.0: (0.02, 0.02, 0.08),    # Midnight - deep blue/black
            0.18: (0.05, 0.03, 0.12),   # Pre-dawn - very dark blue
            0.22: (0.15, 0.08, 0.25),   # Early dawn - purple hint
            0.25: (0.6, 0.25, 0.35),    # Dawn - rose/magenta
            0.28: (0.9, 0.45, 0.25),    # Sunrise - vivid orange
            0.32: (0.95, 0.65, 0.35),   # Early morning - golden
            0.38: (0.6, 0.75, 0.95),    # Morning - light blue
            0.5: (0.45, 0.68, 0.98),    # Noon - bright blue
            0.62: (0.5, 0.72, 0.95),    # Afternoon - blue
            0.70: (0.85, 0.6, 0.35),    # Late afternoon - golden
            0.75: (0.95, 0.45, 0.2),    # Sunset - vivid orange
            0.78: (0.7, 0.25, 0.35),    # Dusk - magenta/red
            0.82: (0.35, 0.15, 0.35),   # Twilight - purple
            0.88: (0.1, 0.05, 0.18),    # Night begins - dark purple
            1.0: (0.02, 0.02, 0.08),    # Back to midnight
        }
        
        # Interpolate between key colors
        keys = sorted(colors.keys())
        for i in range(len(keys) - 1):
            if keys[i] <= t < keys[i + 1]:
                t0, t1 = keys[i], keys[i + 1]
                c0, c1 = colors[t0], colors[t1]
                blend = (t - t0) / (t1 - t0)
                return (
                    c0[0] + (c1[0] - c0[0]) * blend,
                    c0[1] + (c1[1] - c0[1]) * blend,
                    c0[2] + (c1[2] - c0[2]) * blend,
                )
        
        return colors[0.0]
    
    def get_light_color(self) -> tuple:
        """Get sunlight color for terrain lighting."""
        t = self.time
        
        if 0.25 <= t < 0.35:  # Dawn
            blend = (t - 0.25) / 0.1
            return (0.8 + blend * 0.2, 0.6 + blend * 0.35, 0.4 + blend * 0.5)
        elif 0.35 <= t < 0.75:  # Day
            return (1.0, 0.95, 0.9)
        elif 0.75 <= t < 0.85:  # Dusk
            blend = (t - 0.75) / 0.1
            return (1.0 - blend * 0.7, 0.95 - blend * 0.75, 0.9 - blend * 0.75)
        else:  # Night
            return (0.15, 0.15, 0.25)  # Moonlight tint
    
    def get_ambient_level(self) -> float:
        """Get ambient light level (0-1)."""
        t = self.time
        
        if 0.3 <= t < 0.7:  # Full day
            return 0.8
        elif 0.25 <= t < 0.3:  # Dawn transition
            return 0.3 + (t - 0.25) / 0.05 * 0.5
        elif 0.7 <= t < 0.8:  # Dusk transition
            return 0.8 - (t - 0.7) / 0.1 * 0.5
        else:  # Night
            return 0.2
    
    def get_fog_color(self) -> tuple:
        """Get fog color based on time."""
        sky = self.get_sky_color()
        # Fog is slightly lighter than sky
        return (
            min(1.0, sky[0] + 0.1),
            min(1.0, sky[1] + 0.1),
            min(1.0, sky[2] + 0.1),
        )
    
    def is_night(self) -> bool:
        """Check if it's currently night."""
        return self.time < 0.25 or self.time > 0.8
    
    def get_time_name(self) -> str:
        """Get human-readable time of day."""
        t = self.time
        if t < 0.2:
            return "Midnight"
        elif t < 0.25:
            return "Early Morning"
        elif t < 0.33:
            return "Dawn"
        elif t < 0.45:
            return "Morning"
        elif t < 0.55:
            return "Noon"
        elif t < 0.7:
            return "Afternoon"
        elif t < 0.8:
            return "Evening"
        elif t < 0.875:
            return "Dusk"
        else:
            return "Night"
    
    def render(self, camera_x: float, camera_y: float, camera_z: float):
        """Render sky, sun, moon, and stars."""
        if self.sun_list is None:
            self.init_gl()
        
        # Disable depth writing for sky
        glDepthMask(GL_FALSE)
        glDisable(GL_LIGHTING)
        
        # Render stars (visible at night and twilight)
        star_alpha = 0.0
        if self.time < 0.28:  # Before sunrise
            star_alpha = 1.0 - (self.time / 0.28)
        elif self.time > 0.72:  # After sunset
            star_alpha = (self.time - 0.72) / 0.28
        
        if star_alpha > 0.05:
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE)  # Additive for stars
            glPushMatrix()
            glTranslatef(camera_x, camera_y, camera_z)
            # Modulate star brightness
            glColor4f(1, 1, 1, star_alpha)
            glCallList(self.stars_list)
            glPopMatrix()
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        
        # Render sun (only during day: 0.20 to 0.80)
        if 0.20 <= self.time <= 0.80:
            sun_x, sun_y, sun_z = self.get_sun_position()
            if sun_y > 0:  # Only when above horizon
                glPushMatrix()
                glTranslatef(camera_x + sun_x, camera_y + sun_y, camera_z + sun_z)
                glCallList(self.sun_list)
                glPopMatrix()
        
        # Render moon (only during night)
        if self.time >= 0.70 or self.time <= 0.30:
            moon_x, moon_y, moon_z = self.get_moon_position()
            if moon_y > 0:  # Only when above horizon
                glPushMatrix()
                glTranslatef(camera_x + moon_x, camera_y + moon_y, camera_z + moon_z)
                glCallList(self.moon_list)
                glPopMatrix()
        
        glDepthMask(GL_TRUE)
        glEnable(GL_LIGHTING)
    
    def apply_lighting(self):
        """Apply lighting based on sun position and time."""
        sun_pos = self.get_sun_position()
        light_color = self.get_light_color()
        ambient = self.get_ambient_level()
        
        # Normalize sun direction
        length = math.sqrt(sun_pos[0]**2 + sun_pos[1]**2 + sun_pos[2]**2)
        if length > 0:
            light_dir = (sun_pos[0]/length, sun_pos[1]/length, sun_pos[2]/length, 0.0)
        else:
            light_dir = (0, 1, 0, 0)
        
        glLightfv(GL_LIGHT0, GL_POSITION, light_dir)
        glLightfv(GL_LIGHT0, GL_DIFFUSE, (*light_color, 1.0))
        glLightfv(GL_LIGHT0, GL_AMBIENT, (ambient * 0.4, ambient * 0.4, ambient * 0.5, 1.0))
    
    def apply_fog(self):
        """Update fog color based on time."""
        fog_color = self.get_fog_color()
        glFogfv(GL_FOG_COLOR, (*fog_color, 1.0))

