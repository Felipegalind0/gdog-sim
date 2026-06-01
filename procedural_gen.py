import tempfile

import genesis as gs
from jinja2 import Template
import numpy as np


def generate_random_robot_urdf(rng):
    """Reads the Jinja template, randomizes parameters, and returns a temporary URDF path and params."""
    with open("gdog.urdf.jinja", "r") as f:
        template = Template(f.read())

    # Generate Random Parameters (Domain Randomization)
    body_length = float(rng.uniform(0.4, 0.5))
    body_width = float(rng.uniform(0.15, 0.2))
    body_height = float(rng.uniform(0.08, 0.15))
    face_width = float( body_width + rng.uniform(0.01, 0.02) )
    face_height = float( body_height + rng.uniform(0.01, 0.02) )

    robot_params = {
        "body_length": body_length,
        "body_width": body_width,
        "body_height": body_height,
        "body_mass": rng.uniform(3.0, 4.0),
        "face_width": face_width,
        "face_height": face_height,
        "face_depth": float(face_width * rng.uniform(0.3, 0.4)),
        "leg_thickness": rng.uniform(0.04, 0.06),
        "thigh_length": rng.uniform(0.18, 0.25),
        "calf_thickness": rng.uniform(0.03, 0.05),
        "calf_length": rng.uniform(0.18, 0.25),
        "wheel_radius": rng.uniform(0.04, 0.06),
        "wheel_width": rng.uniform(0.03, 0.04),
    }

    # Render URDF XML string
    urdf_string = template.render(**robot_params)

    # Write to a temporary file
    temp_urdf = tempfile.NamedTemporaryFile(delete=False, suffix=".urdf", mode="w")
    temp_urdf.write(urdf_string)
    temp_urdf.close()

    print("Generated random robot with parameters:")
    for k, v in robot_params.items():
        print(f"  {k}: {v:.3f}")

    return temp_urdf.name, robot_params


def generate_random_terrain_morph(rng):
    n_subterrains = (3, 3)
    subterrain_size = (8.0, 8.0)

    terrain_pool = [
        "flat_terrain",
        "random_uniform_terrain",
        "wave_terrain",
        "sloped_terrain",
        "pyramid_sloped_terrain",
    ]
    terrain_weights = np.array([0.35, 0.25, 0.15, 0.15, 0.10], dtype=np.float64)
    terrain_weights /= terrain_weights.sum()

    center_i = n_subterrains[0] // 2
    center_j = n_subterrains[1] // 2
    subterrain_types = []
    for i in range(n_subterrains[0]):
        row = []
        for j in range(n_subterrains[1]):
            # Keep the center patch flat to make spawn behavior stable.
            if i == center_i and j == center_j:
                row.append("flat_terrain")
            else:
                row.append(str(rng.choice(terrain_pool, p=terrain_weights)))
        subterrain_types.append(row)

    vertical_scale = float(rng.uniform(0.004, 0.01))
    horizontal_scale = 0.25

    subterrain_parameters = {
        "random_uniform_terrain": {
            "min_height": float(-rng.uniform(0.02, 0.08)),
            "max_height": float(rng.uniform(0.02, 0.08)),
            "step": float(rng.choice(np.array([0.02, 0.03, 0.04, 0.05]))),
            "downsampled_scale": float(rng.uniform(0.2, 0.6)),
        },
        "wave_terrain": {
            "num_waves": float(rng.integers(1, 4)),
            "amplitude": float(rng.uniform(0.03, 0.12)),
        },
        "sloped_terrain": {
            "slope": float(rng.uniform(-0.25, 0.25)),
        },
        "pyramid_sloped_terrain": {
            "slope": float(rng.uniform(-0.2, 0.2)),
        },
    }

    terrain_pos = (
        -0.5 * n_subterrains[0] * subterrain_size[0],
        -0.5 * n_subterrains[1] * subterrain_size[1],
        0.0,
    )

    terrain_morph = gs.morphs.Terrain(
        pos=terrain_pos,
        randomize=False,
        n_subterrains=n_subterrains,
        subterrain_size=subterrain_size,
        horizontal_scale=horizontal_scale,
        vertical_scale=vertical_scale,
        subterrain_types=subterrain_types,
        subterrain_parameters=subterrain_parameters,
    )

    return terrain_morph, {
        "n_subterrains": n_subterrains,
        "subterrain_size": subterrain_size,
        "horizontal_scale": horizontal_scale,
        "vertical_scale": vertical_scale,
        "subterrain_types": subterrain_types,
    }


def generate_moon_albedo_texture(rng, size=512):
    # Multi-scale noise creates low-cost moon-like albedo variation.
    coarse_1 = rng.normal(0.0, 1.0, size=(64, 64))
    coarse_2 = rng.normal(0.0, 1.0, size=(128, 128))
    noise_1 = np.repeat(np.repeat(coarse_1, 8, axis=0), 8, axis=1)[:size, :size]
    noise_2 = np.repeat(np.repeat(coarse_2, 4, axis=0), 4, axis=1)[:size, :size]
    h = 0.7 * noise_1 + 0.3 * noise_2

    # Add crater-like bowl/rim marks in the albedo to improve readability.
    crater_field = np.zeros((size, size), dtype=np.float32)
    n_craters = 70
    for _ in range(n_craters):
        cx = int(rng.integers(0, size))
        cy = int(rng.integers(0, size))
        radius = float(rng.uniform(8.0, 36.0))

        x0 = max(0, int(cx - 1.6 * radius))
        x1 = min(size, int(cx + 1.6 * radius) + 1)
        y0 = max(0, int(cy - 1.6 * radius))
        y1 = min(size, int(cy + 1.6 * radius) + 1)
        if x1 - x0 < 2 or y1 - y0 < 2:
            continue

        yy, xx = np.mgrid[y0:y1, x0:x1]
        dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
        d = dist / max(radius, 1e-6)

        bowl = -0.30 * np.exp(-2.0 * d * d)
        rim = 0.18 * np.exp(-((d - 1.0) / 0.22) ** 2)
        crater_field[y0:y1, x0:x1] += (bowl + rim).astype(np.float32)

    h = h + crater_field
    h = (h - h.min()) / max(h.max() - h.min(), 1e-8)

    base = 92.0 + 110.0 * h
    bands = 0.94 + 0.06 * np.sin(2.0 * np.pi * (8.0 * h + float(rng.uniform(0.0, 1.0))))
    albedo = np.clip(base * bands, 0.0, 255.0).astype(np.uint8)
    return np.stack([albedo, albedo, albedo], axis=-1)
