"""A small custom room: floor + 4 walls, with some character.

Standalone scene so we can look at an enclosed room before dropping the arm/
desk inside it. Built from the same primitives as so101_scene (fixed `Box`
morphs + `Rough` surfaces) so it composes cleanly later.

The walls are NOT pure white: each gets a different muted, designer-ish hue
(clay / sage / slate / greige) so the room has depth and the cameras get a
non-flat background, while staying calm enough not to confuse a policy.

Run (open the window from a normal macOS Terminal, not automation):
    .venv/bin/python scripts/room_scene.py
"""

import numpy as np
import genesis as gs

import so101_scene as S  # reuse the real desk geometry the arm mounts to

# --- Room geometry (meters) ---------------------------------------------------
# Centered a little forward (+x) of the world origin so a desk/arm sitting near
# x in [0, 0.6] lands roughly in the middle of the room rather than against the
# back wall.
ROOM_CENTER = (0.3, 0.0)
ROOM_HALF_X = 1.5            # half depth (x): room spans 3.0 m front-to-back
ROOM_HALF_Y = 1.5            # half width (y): room spans 3.0 m side-to-side
ROOM_HEIGHT = 2.4           # wall height (z)
WALL_THICKNESS = 0.05

# --- Palette (RGB 0..1), muted so it has character without shouting -----------
FLOOR_COLOR = (0.72, 0.60, 0.45)   # warm light oak
BACK_COLOR = (0.74, 0.56, 0.48)    # dusty clay (the -x wall the arm faces away from)
LEFT_COLOR = (0.62, 0.69, 0.58)    # soft sage green (+y)
RIGHT_COLOR = (0.56, 0.62, 0.70)   # muted slate blue (-y)
FRONT_COLOR = (0.86, 0.83, 0.78)   # warm greige (+x); lightest, faces the cameras


def add_room(scene):
    """Add a floor slab + 4 enclosing walls. All fixed (static geometry)."""
    cx, cy = ROOM_CENTER
    hx, hy = ROOM_HALF_X, ROOM_HALF_Y
    h, t = ROOM_HEIGHT, WALL_THICKNESS

    # Warm floor slab covering the room interior (sits flush on the ground plane).
    scene.add_entity(
        gs.morphs.Box(
            pos=(cx, cy, 0.01),
            size=(2 * hx, 2 * hy, 0.02),
            fixed=True,
        ),
        surface=gs.surfaces.Rough(color=FLOOR_COLOR),
    )

    z_center = h / 2.0
    # Back (-x) and front (+x) walls span the full width plus the wall thickness
    # on each side so the corners close cleanly against the side walls.
    span_y = 2 * hy + 2 * t
    for sx, color in ((-1.0, BACK_COLOR), (1.0, FRONT_COLOR)):
        scene.add_entity(
            gs.morphs.Box(
                pos=(cx + sx * hx, cy, z_center),
                size=(t, span_y, h),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(color=color),
        )

    # Left (+y) and right (-y) walls run along the depth (x).
    for sy, color in ((1.0, LEFT_COLOR), (-1.0, RIGHT_COLOR)):
        scene.add_entity(
            gs.morphs.Box(
                pos=(cx, cy + sy * hy, z_center),
                size=(2 * hx, t, h),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(color=color),
        )


# --- Bed (composed from primitives; no furniture asset ships with Genesis) -----
# Tucked into the back-left corner (-x, +y), long axis along x, clear of the
# desk footprint (desk is y in [-0.6, 0.6]; the bed sits at y >= 0.6).
BED_CENTER = (-0.3, 1.0)     # (x, y) center of the bed footprint
BED_LENGTH = 1.8            # along x (head at -x, against the back wall)
BED_WIDTH = 0.8            # along y
BED_FRAME_H = 0.25         # frame/base height
BED_MATTRESS_H = 0.18

BED_FRAME_COLOR = (0.30, 0.22, 0.16)    # dark walnut
BED_MATTRESS_COLOR = (0.90, 0.88, 0.82)  # warm cream
BED_PILLOW_COLOR = (0.95, 0.94, 0.92)    # soft white
BED_BLANKET_COLOR = (0.40, 0.45, 0.60)   # muted indigo, folded over the foot


def add_bed(scene):
    """Add a simple bed: walnut frame + mattress + pillow + folded blanket."""
    bx, by = BED_CENTER
    L, W = BED_LENGTH, BED_WIDTH
    fh, mh = BED_FRAME_H, BED_MATTRESS_H

    def box(pos, size, color):
        scene.add_entity(
            gs.morphs.Box(pos=pos, size=size, fixed=True),
            surface=gs.surfaces.Rough(color=color),
        )

    # Frame/base: a low solid block on the floor.
    box((bx, by, fh / 2.0), (L, W, fh), BED_FRAME_COLOR)

    # Mattress: slightly inset, resting on the frame.
    mattress_z = fh + mh / 2.0
    mattress_top = fh + mh
    box((bx, by, mattress_z), (L * 0.98, W * 0.96, mh), BED_MATTRESS_COLOR)

    # Pillow at the head end (-x), raised a touch above the mattress.
    pillow_x = bx - L / 2.0 + 0.28
    box((pillow_x, by, mattress_top + 0.05), (0.42, W * 0.85, 0.10), BED_PILLOW_COLOR)

    # Blanket folded over the foot two-thirds (+x), a thin slab on the mattress.
    blanket_len = L * 0.6
    blanket_x = bx + L / 2.0 - blanket_len / 2.0 - 0.05
    box((blanket_x, by, mattress_top + 0.025), (blanket_len, W * 0.99, 0.05),
        BED_BLANKET_COLOR)


def build_scene(show_viewer: bool = True):
    """Create a scene with ground + the enclosing room. Returns the scene."""
    cx, cy = ROOM_CENTER
    scene = gs.Scene(
        viewer_options=gs.options.ViewerOptions(
            # Stand inside near the front-right corner, look across the room at
            # roughly desk height so all four walls are visible.
            camera_pos=(cx + 1.2, -1.2, 1.6),
            camera_lookat=(cx, cy, 0.7),
            camera_fov=50,
        ),
        vis_options=gs.options.VisOptions(
            # Same soft, even overhead lighting as so101_scene.
            ambient_light=(0.6, 0.6, 0.6),
            shadow=False,
            lights=[
                gs.options.vis.DirectionalLight(
                    dir=(0.0, 0.0, -1.0), color=(1.0, 1.0, 1.0), intensity=3.0),
                gs.options.vis.DirectionalLight(
                    dir=(-0.3, -0.2, -1.0), color=(1.0, 1.0, 1.0), intensity=1.5),
                gs.options.vis.DirectionalLight(
                    dir=(0.3, 0.2, -1.0), color=(1.0, 1.0, 1.0), intensity=1.5),
            ],
        ),
        show_viewer=show_viewer,
    )

    scene.add_entity(gs.morphs.Plane())
    add_room(scene)
    S._add_table(scene)   # the real desk the SO101 mounts to
    add_bed(scene)
    return scene


def main() -> None:
    gs.init(backend=gs.cpu)
    scene = build_scene(show_viewer=True)
    scene.build()
    print("Room viewer open — close the window to exit.")
    while scene.viewer.is_alive():
        scene.step()
    print("Room viewer closed cleanly.")


if __name__ == "__main__":
    main()
