# Waverse Development Roadmap

This document tracks planned features and future development directions.

## 🛠️ Active Development

### HUD System (v0.9)
- [x] Centered crosshair with gap
- [x] Compass bar with cardinal directions (N, S, E, W)
- [x] Coordinate display (top left) with shadow for visibility
- [x] FPS counter
- [x] Minecraft-style item bar (bottom center)
- [x] Tool icons (pickaxe, shovel, camera, axe, magnifier, microscope)
- [x] L1/R1 (or `,`/`.`) tool cycling
- [x] Status text display with shadow
- [ ] Biome name display
- [ ] Time of day indicator
- [ ] Weather indicator

### Current Tools (v0.9)
| Tool | Hotkey | Description |
|------|--------|-------------|
| Pickaxe | MINE | Lower/mine terrain |
| Shovel | FILL | Raise/fill terrain |
| Camera | SCAN | Scan and log plant/animal DNA |
| Axe | CUT | (Future) Cut down trees/wood |
| Magnifier | ZOOM | (Future) Examine surfaces for micro-biomes |
| Microscope | MICRO | (Future) View micro-life |

---

## 🚀 Future Features

### 1. Microbiome System (Planned)
**Magnifying Glass Tool**
- Point at a surface (plant, ground, water, rock)
- "Zoom in" to reveal a procedurally generated micro-biome
- Insect-like life: ants, beetles, mites, worms
- Plant details: pollen, spores, seeds, fungal networks
- Terrain details: mineral crystals, soil layers, fossils

**Microscope Tool**
- Even smaller scale than magnifying glass
- View micro-organisms: amoeba, bacteria, tardigrades
- Cell-like structures with procedural generation
- Microlife can also be scanned and logged!

### 2. Expanded Life Systems

**Insects & Small Creatures**
- Ants (colony formations, trails)
- Bees (hive structures, pollination)
- Butterflies (migration patterns)
- Beetles (various species)
- Spiders (web building)
- Worms (soil aeration)
- Snails (shell genetics)

**Micro-organisms**
- Amoeba (movement patterns)
- Tardigrades (extreme survival)
- Paramecium (cilia movement)
- Algae colonies
- Plankton in water

### 3. Resource System

**Wood Cutting (Axe Tool)**
- Cut down trees for wood resources
- Trees regenerate over time
- Different wood types from different tree DNA
- Wood used for building/crafting

**Mining System (Pickaxe Tool)**
- Mine terrain to reveal mineral layers
- Different ores/crystals in different biomes
- Depth-based resource generation

**Building Materials**
- Stone (from mining)
- Wood (from cutting)
- Crystal (special biomes)
- Obsidian (volcanic biomes)

### 4. Building System Expansion

**Player Structures**
- Place blocks from inventory
- Snap-to-grid building
- Copy/paste structures
- Blueprint system

**NPC Structures**
- Villages with population
- Procedural interior layouts
- NPCs with daily routines

### 5. Biome Enhancements

**Exotic Biomes (Completed ✓)**
- [x] Psychedelic (rainbow colors, dancing plants)
- [x] Hellfire (volcanoes, fire plants, lava)
- [x] Shadow (dark, eerie, shadow creatures)
- [x] Crystal (prismatic, geometric)
- [x] Void (empty, mysterious)

**Future Biomes**
- Underwater kingdoms (submersible gameplay)
- Floating islands (aerial exploration)
- Underground caverns (spelunking)
- Bioluminescent forests (night beauty)
- Coral reefs (underwater diversity)

### 6. Multiplayer (Long-term)
- Local co-op (split screen)
- Network co-op (shared world)
- Shared DNA catalog
- Building collaboration

---

## 🐛 Known Issues / Polish

### Performance
- [ ] Further optimize chunk loading
- [ ] LOD system for distant objects
- [ ] Occlusion culling
- [ ] GPU instancing optimization

### Visuals
- [ ] Shadow mapping
- [ ] Screen-space ambient occlusion
- [ ] Better water reflections
- [ ] Plant detail textures

### Gameplay
- [ ] Better collision detection
- [ ] Swimming mechanics
- [ ] Day/night cycle effects on wildlife
- [ ] Seasonal changes

---

## 📊 Technical Debt

### Code Quality
- [ ] Unit tests for DNA system
- [ ] Unit tests for all plant/animal types
- [ ] Integration tests for chunk generation
- [ ] Performance benchmarks

### Documentation
- [ ] API documentation for modding
- [ ] Asset creation guide
- [ ] Biome creation tutorial

---

## 🎮 Controls Reference

### Keyboard
| Key | Action |
|-----|--------|
| WASD | Move |
| H/F | Fly up/down |
| Space | Jump |
| Right Click + Mouse | Look around |
| IJKL | Keyboard look |
| Tab | Faster look |
| , / . | Prev/Next tool |
| T | Cycle tool radius |
| R | Use tool |
| E | Screenshot |
| X | Toggle auto-fly/tour mode |
| ESC | Menu |
| Ctrl+1-5 | Warp to exotic biome |
| Shift+N | Random exotic biome |

### Gamepad
| Button | Action |
|--------|--------|
| Left Stick | Move |
| Right Stick | Look |
| A | Action |
| B | Jump |
| X | Toggle fly/walk |
| Y | Screenshot |
| L2/R2 | Speed down/up |
| L1/R1 | Prev/Next tool (in menu: tabs) |
| R1 | Use tool (out of menu) |
| D-PAD ←→ | Cycle tools |
| D-PAD ↑↓ | Tool radius |
| START | Menu |
| SELECT | Warp to new universe |

---

*Last updated: December 2024*

