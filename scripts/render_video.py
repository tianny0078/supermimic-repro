"""Record fresh simulator rollouts and render policy/reference videos inside Slurm.

All clips start from frame zero; excerpt start times only trim the saved video.
Reference XY/yaw are never aligned to the robot. Terminal poses come from replay's
pre-reset terminal row, so an auto-reset can never appear as recovery in the video.
"""
from __future__ import annotations

import json
from collections import deque
import os
from pathlib import Path
import shutil
import subprocess
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import torch
import tyro

from supermimic.artifacts import write_json
from supermimic.data.motion_convert import g1_xml_path
from supermimic.envs import SuperMimicEnvCfgArgs, make_env
from supermimic.envs.mdp.motion_library import discover_npz
from supermimic.envs.state_io import StateIO
from supermimic.eval.benchmark import load_policy_checkpoint
from supermimic.training.provenance import run_manifest, sha256_file


DEFAULT_SHOTS = [
    {"name": "dance2_subject1", "start": 26, "seconds": 12, "label": "DANCE"},
    {"name": "jumps1_subject1", "start": 34, "seconds": 12, "label": "JUMPS"},
    {"name": "walk1_subject1", "start": 10, "seconds": 10, "label": "WALK"},
    {"name": "fallAndGetUp1_subject1", "start": 30, "seconds": 14, "label": "FALL AND GET UP"},
    {"name": "first_tracking_failure", "start": 0, "seconds": 10, "label": "FAILURE DIAGNOSTIC", "diagnostic": True, "auto_failure": True},
]


def make_renderer(width, height):
    path = g1_xml_path()
    xml = ET.parse(path).getroot()
    xml.find("compiler").set("meshdir", str(path.parent / "assets"))
    visual = ET.SubElement(xml, "visual")
    ET.SubElement(visual, "global", offwidth=str(width), offheight=str(height))
    ET.SubElement(visual, "quality", shadowsize="2048")
    asset = xml.find("asset")
    ET.SubElement(asset, "texture", name="video_grid", type="2d", builtin="checker", rgb1="0.87 0.89 0.92", rgb2="0.73 0.77 0.82", width="512", height="512")
    ET.SubElement(asset, "material", name="video_floor", texture="video_grid", texrepeat="1 1", texuniform="true", rgba="0.45 0.48 0.52 1", reflectance="0.05")
    ET.SubElement(asset, "texture", name="video_sky", type="skybox", builtin="gradient", rgb1="0.92 0.95 0.98", rgb2="0.66 0.74 0.84", width="256", height="256")
    world = xml.find("worldbody")
    ET.SubElement(world, "geom", name="video_floor", type="plane", size="0 0 0.1", material="video_floor", contype="0", conaffinity="0")
    ET.SubElement(world, "light", pos="0 0 8", dir="0 0 -1", directional="true", diffuse="0.8 0.8 0.8")
    model = mujoco.MjModel.from_xml_string(ET.tostring(xml, encoding="unicode"))
    robot_geoms = model.geom_bodyid != 0
    model.geom_matid[robot_geoms] = -1
    model.geom_rgba[robot_geoms] = [0.93, 0.49, 0.15, 1]
    model.vis.headlight.ambient[:] = 0.4
    model.vis.headlight.diffuse[:] = 0.6
    data, reference = mujoco.MjData(model), mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=height, width=width)
    options = mujoco.MjvOption()
    options.geomgroup[3:] = 0
    options.sitegroup[:] = 0
    camera = mujoco.MjvCamera()
    camera.azimuth, camera.elevation = 135, -17
    return model, data, reference, renderer, options, camera


@torch.no_grad()
def collect(policy, config, files, shots, fps):
    # Same 40-world batch and contact pool as the full-motion benchmark.
    env = make_env(SuperMimicEnvCfgArgs(
        npz_files=[str(p) for p in files], num_envs=len(files),
        future_frames=config["env"]["future_frames"], episode_length_s=1e5,
        action_noise_std=0, obs_noise=False, reset_noise=False,
        domain_randomization=False, sampling_mode="start", replay_capacity=4,
        record_transitions=True, seed=0,
        anchor_obs=config["env"].get("anchor_obs", False),
        anchor_orientation=config["env"].get("anchor_orientation", False),
    ), device="cuda:0")
    try:
        cmd = env.command_manager.get_term("motion")
        lib = cmd.library
        names = [c.name for c in lib.clips]
        if len(names) != 40:
            raise ValueError("Video parity currently requires the 40-clip LAFAN1 benchmark")
        regular = [s for s in shots if not s.get("auto_failure")]
        auto_failure = next((s for s in shots if s.get("auto_failure")), None)
        ids = [names.index(s["name"]) for s in regular]
        cmd.forced_frame[:] = lib.clip_start.clone()
        obs, _ = env.reset()
        cmd.forced_frame[:] = lib.clip_start[-1]  # Same idle-world reset as benchmark's first batch.
        io = StateIO(env)
        buffer = env.recorder_manager.get_term("replay").buffer
        active = torch.ones(len(files), dtype=torch.bool, device=env.device)
        records = {s["name"]: {"qpos": [], "ref": [], "time": [], "terminated": [], "fail_step": None} for s in regular}
        regular_limit = round(max(s["start"] + s["seconds"] for s in regular) * 50)
        step_limit = int(lib.clip_len.max()) if auto_failure else regular_limit
        tail = deque(maxlen=10 * fps + 2)
        found_failure = False
        stride = 50 // fps

        def capture(step, state, terminal_ids=()):
            for shot, i in zip(regular, ids):
                if not active[i] and i not in terminal_ids:
                    continue
                if step % stride != 0 and i not in terminal_ids:
                    continue
                if not round(shot["start"] * 50) <= step <= round((shot["start"] + shot["seconds"]) * 50):
                    continue
                record = records[shot["name"]]
                frame = min(int(lib.clip_start[i]) + step, int(lib.clip_last[i]))
                qref, _ = lib.ref_qpos_qvel(torch.tensor([frame], device=env.device))
                record["qpos"].append(state[i, :36].cpu().numpy().copy())
                record["ref"].append(qref[0].cpu().numpy().copy())
                record["time"].append(step / 50)
                record["terminated"].append(i in terminal_ids)

        capture(0, io.read())
        for step in range(1, step_limit + 1):
            obs, _, terminated, timed_out, _ = env.step(policy(obs["policy"], obs["goal"]))
            done = (terminated | timed_out) & active
            state = io.read()
            ended = done.nonzero().flatten()
            terminal_ids = ended.tolist()
            if terminal_ids:
                terminal_rows = (buffer.ptr[ended] - 1) % buffer.T
                assert buffer.terminal[ended, terminal_rows].all(), "Missing true pre-reset terminal state"
                state[ended] = buffer.s[ended, terminal_rows]
                for shot, i in zip(regular, ids):
                    if i in terminal_ids:
                        records[shot["name"]]["fail_step"] = step if bool(terminated[i]) else None
            capture(step, state, terminal_ids)
            if auto_failure and not found_failure and (step % stride == 0 or terminal_ids):
                ref_frame = torch.minimum(lib.clip_start + step, lib.clip_last)
                ref_q, _ = lib.ref_qpos_qvel(ref_frame)
                tail.append((step / 50, state[:, :36].cpu().numpy().copy(), ref_q.cpu().numpy().copy()))
                failure_ids = (terminated & active).nonzero().flatten().tolist()
                if failure_ids:
                    i = failure_ids[0]
                    auto_failure.update(name=names[i], record_key="failure_" + names[i], start=tail[0][0], seconds=step / 50 - tail[0][0])
                    records[auto_failure["record_key"]] = {
                        "qpos": [q[i] for _, q, _ in tail], "ref": [q[i] for _, _, q in tail],
                        "time": [t for t, _, _ in tail], "terminated": [False] * (len(tail) - 1) + [True], "fail_step": step,
                    }
                    found_failure = True
                    print(f"[record] first actual tracking failure: {names[i]} at {step / 50:.2f}s", flush=True)
            active &= ~done
            if step % 500 == 0:
                print(f"[record] {step}/{step_limit} simulator steps", flush=True)
            if step >= regular_limit and (not auto_failure or found_failure):
                break
        if auto_failure and not found_failure:
            shots.remove(auto_failure)
            print("[record] No termination in this fresh replay; diagnostic video omitted.", flush=True)
        for shot in regular:
            if not records[shot["name"]]["qpos"]:
                raise RuntimeError(f"{shot['name']} ended before the selected excerpt; inspect fail_step instead of hiding failure")
        return records
    finally:
        env.close()


def encode(path, width, height, fps):
    return subprocess.Popen([
        "ffmpeg", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24",
        "-s", f"{width}x{height}", "-r", str(fps), "-i", "-", "-an", "-c:v", "libx264",
        "-preset", "fast", "-threads", "4", "-crf", "21", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(path),
    ], stdin=subprocess.PIPE)


def render_shot(scene, record, shot, fps, width, height, movie, poster=None):
    model, data, reference, renderer, options, camera = scene
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    bold = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", 23)
    count = 0
    for t, (q, qr, time, terminal) in enumerate(zip(record["qpos"], record["ref"], record["time"], record["terminated"])):
        data.qpos[:] = q
        reference.qpos[:] = qr
        mujoco.mj_forward(model, data)
        mujoco.mj_forward(model, reference)
        camera.lookat[:] = [(q[0] + qr[0]) / 2, (q[1] + qr[1]) / 2, 0.68]
        drift = float(np.linalg.norm(q[:2] - qr[:2]))
        camera.distance = max(3.3, 2.6 + 1.1 * drift)
        renderer.update_scene(data, camera=camera, scene_option=options)
        first_ghost = renderer.scene.ngeom
        mujoco.mjv_addGeoms(model, reference, options, mujoco.MjvPerturb(), mujoco.mjtCatBit.mjCAT_DYNAMIC.value, renderer.scene)
        for i in range(first_ghost, renderer.scene.ngeom):
            geom = renderer.scene.geoms[i]
            geom.rgba[:] = [0.12, 0.58, 1.0, 0.32]
            geom.matid = -1
        image = Image.fromarray(renderer.render())
        draw = ImageDraw.Draw(image)
        draw.rectangle((0, 0, width, 88), fill=(20, 30, 45))
        draw.text((18, 9), f"SUPERMIMIC  /  {shot['label']}", font=bold, fill="white")
        draw.text((18, 46), f"{shot['name']}   |   motion {time:.2f}s   |   XY error {drift:.2f} m", font=font, fill=(205, 218, 235))
        draw.rectangle((0, height - 39, width, height), fill=(20, 30, 45))
        draw.text((18, height - 32), "Orange: policy    Blue: reference    |    Global poses, no alignment    |    1x speed", font=font, fill=(240, 240, 240))
        if terminal:
            draw.rectangle((18, 104, width - 18, 146), fill=(160, 32, 38))
            draw.text((30, 112), f"TRACKING TERMINATED at {time:.2f}s  /  terminal pose held", font=font, fill="white")
        rgb = np.asarray(image).tobytes()
        movie.stdin.write(rgb)
        count += 1
        if poster is not None and t == min(50, len(record["time"]) - 1):
            image.save(poster)
        if terminal:
            for _ in range(fps):
                movie.stdin.write(rgb)
                count += 1
    return count


def main(ckpt: Path, out_dir: Path, shots_json: Path | None = None, fps: int = 25, width: int = 960, height: int = 540):
    if not os.getenv("SLURM_JOB_ID"):
        raise RuntimeError("Simulation and EGL rendering must run in a Slurm allocation")
    if fps < 1 or 50 % fps:
        raise ValueError("fps must divide the 50 Hz control frequency")
    if width % 2 or height % 2:
        raise ValueError("H264 dimensions must be even")
    if out_dir.exists():
        raise FileExistsError(f"Choose a new video output directory: {out_dir}")
    out_dir.mkdir(parents=True)
    shots = json.loads(shots_json.read_text()) if shots_json else DEFAULT_SHOTS
    original_checkpoint = ckpt.resolve()
    staged = Path(os.environ["SUPERMIMIC_SCRATCH"]) / "video_checkpoint.pt"
    shutil.copyfile(original_checkpoint, staged)
    policy, config, checkpoint_info = load_policy_checkpoint(staged, "cuda:0")
    files = discover_npz(Path("data/npz/lafan1"))
    provenance = run_manifest(files, "cuda:0")
    if provenance["dataset_sha256"] != checkpoint_info["manifest"]["dataset_sha256"]:
        raise ValueError("Video data differs from training data")
    scene = make_renderer(width, height)  # Validate EGL before collecting the rollout.
    try:
        records = collect(policy, config, files, shots, fps)
        metadata = {"checkpoint": original_checkpoint, "checkpoint_sha256": sha256_file(staged),
                    "checkpoint_info": checkpoint_info, "source": provenance, "shots": shots,
                    "protocol": "fresh deterministic mjlab/MuJoCo Warp rollout from each first frame; same 40-world batch; true terminal state; global reference without XY/yaw alignment",
                    "selection": "Selected illustrative excerpts, not random samples or a full-motion evaluation; failure example chosen from the previous benchmark.",
                    "fps": fps, "width": width, "height": height, "videos": []}
        for diagnostic in (False, True):
            selected = [s for s in shots if bool(s.get("diagnostic")) == diagnostic]
            if not selected:
                continue
            name = "failure_diagnostic" if diagnostic else "highlights"
            output = out_dir / (name + ".mp4")
            movie = encode(output, width, height, fps)
            frame_count = 0
            try:
                for i, shot in enumerate(selected):
                    record = records[shot.get("record_key", shot["name"])]
                    frame_count += render_shot(scene, record, shot, fps, width, height, movie,
                                               out_dir / (name + ".jpg") if i == 0 else None)
                    np.savez_compressed(out_dir / (shot.get("record_key", shot["name"]) + ".npz"), qpos=np.asarray(record["qpos"]), reference_qpos=np.asarray(record["ref"]),
                                        time=np.asarray(record["time"]), terminated=np.asarray(record["terminated"]))
                    print(f"[video] {shot['name']}: {len(record['time'])} sampled states, fail_step={record['fail_step']}", flush=True)
            finally:
                movie.stdin.close()
                code = movie.wait()
            if code:
                raise RuntimeError(f"ffmpeg exited with {code}")
            metadata["videos"].append({"path": output, "frames": frame_count, "duration_s": frame_count / fps, "sha256": sha256_file(output)})
        metadata["rollouts"] = {name: {"sampled_frames": len(r["time"]), "first_time": r["time"][0], "last_time": r["time"][-1], "fail_step": r["fail_step"]} for name, r in records.items()}
        write_json(out_dir / "manifest.json", metadata)
    finally:
        scene[3].close()
    print(f"[DONE] videos and provenance in {out_dir}", flush=True)


if __name__ == "__main__":
    tyro.cli(main)
