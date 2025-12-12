"""
WaveDNA - The genetic code of a Waverse world.

Everything in the world is deterministically generated from this DNA structure.
The DNA is JSON-serializable for saving/loading/sharing worlds.
"""

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Tuple, Dict, Any, Literal
import json
import hashlib


# Wave function types
WaveType = Literal["sin", "cos", "triangle", "sawtooth", "square", "perlin", "simplex", "ridged"]

# Falloff function types for localized features  
FalloffType = Literal["linear", "gaussian", "cosine", "smooth", "sharp"]


@dataclass
class Wave:
    """
    A single wave component in the Fourier-style terrain generation.
    
    Waves can be 2D (affect height based on x,z) or 3D (affect density in volume).
    """
    # Frequency in x and z directions (lower = larger features)
    freq_x: float = 0.01
    freq_z: float = 0.01
    
    # Optional frequency in y for 3D waves (caves/overhangs)
    freq_y: Optional[float] = None
    
    # Amplitude (height contribution)
    amplitude: float = 10.0
    
    # Phase offset (shifts the wave pattern)
    phase: float = 0.0
    
    # Direction angle in radians (rotates the wave pattern)
    direction: float = 0.0
    
    # Wave function type
    wave_type: WaveType = "sin"
    
    # Optional: harmonic number for Fourier series
    harmonic: int = 1
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Wave":
        return cls(**data)


@dataclass
class WaveLayer:
    """
    A layer of waves that are summed together.
    
    Layers can have different purposes:
    - "continental": Very low frequency, large landmasses
    - "regional": Medium frequency, hills and valleys  
    - "detail": High frequency, small bumps and texture
    - "cave": 3D waves for cave generation
    """
    name: str
    waves: List[Wave] = field(default_factory=list)
    
    # Blend mode with other layers
    blend_mode: Literal["add", "multiply", "max", "min", "average"] = "add"
    
    # Overall layer weight
    weight: float = 1.0
    
    # Optional: only apply below/above certain heights
    height_min: Optional[float] = None
    height_max: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "waves": [w.to_dict() for w in self.waves],
            "blend_mode": self.blend_mode,
            "weight": self.weight,
            "height_min": self.height_min,
            "height_max": self.height_max,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WaveLayer":
        waves = [Wave.from_dict(w) for w in data.get("waves", [])]
        return cls(
            name=data["name"],
            waves=waves,
            blend_mode=data.get("blend_mode", "add"),
            weight=data.get("weight", 1.0),
            height_min=data.get("height_min"),
            height_max=data.get("height_max"),
        )


@dataclass
class Feature:
    """
    A localized feature with a center point and radius.
    
    Features have their own wave patterns that blend with the global terrain
    based on distance from center and falloff function.
    """
    # Feature type identifier
    feature_type: str = "mountain"
    
    # Center position in world coordinates
    center_x: float = 0.0
    center_z: float = 0.0
    
    # Radius of influence
    radius: float = 100.0
    
    # Falloff type (how influence decreases with distance)
    falloff: FalloffType = "gaussian"
    
    # Waves that define this feature's shape
    waves: List[Wave] = field(default_factory=list)
    
    # Height offset (raises/lowers the feature)
    height_offset: float = 0.0
    
    # Optional: vertical extent for 3D features
    y_min: Optional[float] = None
    y_max: Optional[float] = None
    
    # Optional: seed for deterministic variation within feature
    local_seed: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_type": self.feature_type,
            "center_x": self.center_x,
            "center_z": self.center_z,
            "radius": self.radius,
            "falloff": self.falloff,
            "waves": [w.to_dict() for w in self.waves],
            "height_offset": self.height_offset,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "local_seed": self.local_seed,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Feature":
        waves = [Wave.from_dict(w) for w in data.get("waves", [])]
        return cls(
            feature_type=data.get("feature_type", "mountain"),
            center_x=data.get("center_x", 0.0),
            center_z=data.get("center_z", 0.0),
            radius=data.get("radius", 100.0),
            falloff=data.get("falloff", "gaussian"),
            waves=waves,
            height_offset=data.get("height_offset", 0.0),
            y_min=data.get("y_min"),
            y_max=data.get("y_max"),
            local_seed=data.get("local_seed"),
        )


@dataclass 
class CaveLayer:
    """
    Defines cave/tunnel generation using 3D wave carving.
    
    Caves are generated by creating regions where density < 0 (air).
    """
    name: str = "caves"
    
    # 3D waves that carve out caves
    waves: List[Wave] = field(default_factory=list)
    
    # Threshold: values below this become air
    threshold: float = 0.0
    
    # Depth range where caves can appear
    y_min: float = -50.0
    y_max: float = 20.0
    
    # Cave density (probability of cave existing)
    density: float = 0.3
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "waves": [w.to_dict() for w in self.waves],
            "threshold": self.threshold,
            "y_min": self.y_min,
            "y_max": self.y_max,
            "density": self.density,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CaveLayer":
        waves = [Wave.from_dict(w) for w in data.get("waves", [])]
        return cls(
            name=data.get("name", "caves"),
            waves=waves,
            threshold=data.get("threshold", 0.0),
            y_min=data.get("y_min", -50.0),
            y_max=data.get("y_max", 20.0),
            density=data.get("density", 0.3),
        )


@dataclass
class WaveDNA:
    """
    The complete DNA of a Waverse world.
    
    This is the master structure that defines everything about terrain generation.
    It's JSON-serializable for saving, loading, and sharing worlds.
    """
    # Version for compatibility
    version: str = "1.0"
    
    # World seed for deterministic generation
    seed: int = 42
    
    # World name (optional)
    name: str = "Untitled World"
    
    # Base water level (y coordinate)
    water_level: float = 0.0
    
    # Bedrock level (minimum terrain depth)
    bedrock_level: float = -100.0
    
    # Sky level (maximum terrain height)  
    sky_level: float = 200.0
    
    # Global wave layers (applied everywhere)
    layers: List[WaveLayer] = field(default_factory=list)
    
    # Localized features (mountains, craters, etc.)
    features: List[Feature] = field(default_factory=list)
    
    # Cave generation layers
    cave_layers: List[CaveLayer] = field(default_factory=list)
    
    # Chunk size in world units
    chunk_size: int = 64
    
    # Tile size within chunks (grid resolution)
    tile_size: float = 1.0
    
    def __post_init__(self):
        """Generate world hash from DNA content."""
        self._hash = None
    
    @property
    def world_hash(self) -> str:
        """Generate a unique hash for this world DNA."""
        if self._hash is None:
            content = json.dumps(self.to_dict(), sort_keys=True)
            self._hash = hashlib.sha256(content.encode()).hexdigest()[:16]
        return self._hash
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dictionary."""
        return {
            "version": self.version,
            "seed": self.seed,
            "name": self.name,
            "water_level": self.water_level,
            "bedrock_level": self.bedrock_level,
            "sky_level": self.sky_level,
            "layers": [layer.to_dict() for layer in self.layers],
            "features": [feat.to_dict() for feat in self.features],
            "cave_layers": [cave.to_dict() for cave in self.cave_layers],
            "chunk_size": self.chunk_size,
            "tile_size": self.tile_size,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "WaveDNA":
        """Create WaveDNA from dictionary."""
        layers = [WaveLayer.from_dict(l) for l in data.get("layers", [])]
        features = [Feature.from_dict(f) for f in data.get("features", [])]
        cave_layers = [CaveLayer.from_dict(c) for c in data.get("cave_layers", [])]
        
        dna = cls(
            version=data.get("version", "1.0"),
            seed=data.get("seed", 42),
            name=data.get("name", "Untitled World"),
            water_level=data.get("water_level", 0.0),
            bedrock_level=data.get("bedrock_level", -100.0),
            sky_level=data.get("sky_level", 200.0),
            layers=layers,
            features=features,
            cave_layers=cave_layers,
            chunk_size=data.get("chunk_size", 64),
            tile_size=data.get("tile_size", 1.0),
        )
        return dna
    
    def to_json(self, indent: int = 2) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)
    
    @classmethod
    def from_json(cls, json_str: str) -> "WaveDNA":
        """Deserialize from JSON string."""
        return cls.from_dict(json.loads(json_str))
    
    def save(self, filepath: str):
        """Save DNA to a JSON file."""
        with open(filepath, 'w') as f:
            f.write(self.to_json())
    
    @classmethod
    def load(cls, filepath: str) -> "WaveDNA":
        """Load DNA from a JSON file."""
        with open(filepath, 'r') as f:
            return cls.from_json(f.read())
    
    @classmethod
    def create_default(cls) -> "WaveDNA":
        """Create a default world with interesting terrain."""
        return cls(
            name="Default Waverse World",
            seed=42,
            water_level=0.0,
            layers=[
                # Continental scale - huge rolling landmasses
                WaveLayer(
                    name="continental",
                    waves=[
                        Wave(freq_x=0.002, freq_z=0.002, amplitude=40, phase=0.0, wave_type="sin"),
                        Wave(freq_x=0.003, freq_z=0.001, amplitude=25, phase=1.5, wave_type="sin"),
                        Wave(freq_x=0.001, freq_z=0.004, amplitude=30, phase=0.8, wave_type="cos"),
                    ],
                    weight=1.0,
                ),
                # Regional scale - hills and valleys
                WaveLayer(
                    name="regional", 
                    waves=[
                        Wave(freq_x=0.01, freq_z=0.01, amplitude=15, phase=0.0, wave_type="sin"),
                        Wave(freq_x=0.015, freq_z=0.008, amplitude=10, phase=2.1, wave_type="sin"),
                        Wave(freq_x=0.008, freq_z=0.02, amplitude=12, phase=0.5, wave_type="perlin"),
                    ],
                    weight=1.0,
                ),
                # Detail scale - small bumps and texture
                WaveLayer(
                    name="detail",
                    waves=[
                        Wave(freq_x=0.05, freq_z=0.05, amplitude=3, phase=0.0, wave_type="perlin"),
                        Wave(freq_x=0.1, freq_z=0.1, amplitude=1.5, phase=1.0, wave_type="simplex"),
                        Wave(freq_x=0.08, freq_z=0.06, amplitude=2, phase=0.3, wave_type="ridged"),
                    ],
                    weight=1.0,
                ),
            ],
            features=[
                # A big mountain
                Feature(
                    feature_type="mountain",
                    center_x=200,
                    center_z=200,
                    radius=150,
                    falloff="gaussian",
                    waves=[
                        Wave(freq_x=0.02, freq_z=0.02, amplitude=50, wave_type="cos"),
                    ],
                    height_offset=30,
                ),
                # A crater/valley
                Feature(
                    feature_type="crater",
                    center_x=-150,
                    center_z=100,
                    radius=80,
                    falloff="cosine",
                    waves=[
                        Wave(freq_x=0.03, freq_z=0.03, amplitude=-30, wave_type="cos"),
                    ],
                    height_offset=-15,
                ),
            ],
            cave_layers=[
                CaveLayer(
                    name="primary_caves",
                    waves=[
                        Wave(freq_x=0.05, freq_z=0.05, freq_y=0.08, amplitude=1.0, wave_type="perlin"),
                        Wave(freq_x=0.1, freq_z=0.08, freq_y=0.1, amplitude=0.5, wave_type="simplex"),
                    ],
                    threshold=0.3,
                    y_min=-60,
                    y_max=10,
                    density=0.25,
                ),
            ],
        )
    
    @classmethod
    def create_psychedelic(cls) -> "WaveDNA":
        """Create a wild, trippy world with extreme wave patterns."""
        return cls(
            name="Psychedelic Realm",
            seed=420,
            water_level=-5.0,
            layers=[
                # Crazy continental with mixed wave types
                WaveLayer(
                    name="continental_chaos",
                    waves=[
                        Wave(freq_x=0.003, freq_z=0.002, amplitude=50, wave_type="sin"),
                        Wave(freq_x=0.002, freq_z=0.005, amplitude=40, wave_type="triangle"),
                        Wave(freq_x=0.004, freq_z=0.003, amplitude=35, wave_type="ridged", phase=1.2),
                    ],
                ),
                # Interference patterns
                WaveLayer(
                    name="interference",
                    waves=[
                        Wave(freq_x=0.02, freq_z=0.02, amplitude=20, wave_type="sin", direction=0.0),
                        Wave(freq_x=0.02, freq_z=0.02, amplitude=20, wave_type="sin", direction=0.5),
                        Wave(freq_x=0.02, freq_z=0.02, amplitude=20, wave_type="sin", direction=1.0),
                    ],
                ),
                # High frequency ripples
                WaveLayer(
                    name="ripples",
                    waves=[
                        Wave(freq_x=0.15, freq_z=0.15, amplitude=5, wave_type="sin"),
                        Wave(freq_x=0.12, freq_z=0.18, amplitude=4, wave_type="cos", phase=0.7),
                    ],
                ),
            ],
        )


# Convenience function
def create_wave_dna(**kwargs) -> WaveDNA:
    """Quick factory for creating WaveDNA with custom parameters."""
    return WaveDNA(**kwargs)

