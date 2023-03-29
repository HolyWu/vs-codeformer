from __future__ import annotations

import os
from threading import Lock

import cv2
import numpy as np
import torch
import vapoursynth as vs
from torchvision.transforms.functional import normalize

from .codeformer_arch import CodeFormer
from .face_restoration_helper import FaceRestoreHelper
from .img_util import img2tensor, tensor2img

__version__ = "1.0.0"

os.environ["CUDA_MODULE_LOADING"] = "LAZY"

model_dir = os.path.join(os.path.dirname(os.path.realpath(__file__)), "models")


@torch.inference_mode()
def codeformer(
    clip: vs.VideoNode,
    device_index: int | None = None,
    num_streams: int = 1,
    upscale: int = 2,
    detector: int = 0,
    only_center_face: bool = False,
    weight: float = 0.5,
    bg_clip: vs.VideoNode | None = None,
) -> vs.VideoNode:
    """Towards Robust Blind Face Restoration with Codebook Lookup TransFormer

    :param clip:                Clip to process. Only RGB24 format is supported.
    :param device_index:        Device ordinal of the GPU.
    :param num_streams:         Number of CUDA streams to enqueue the kernels.
    :param upscale:             Final upsampling scale.
    :param detector:            Face detector.
                                0 = retinaface_resnet50
                                1 = dlib
    :param only_center_face:    Only restore the center face.
    :param weight:              Balance the quality and fidelity. Generally, smaller weight tends to produce a
                                higher-quality result, while larger weight yields a higher-fidelity result.
    :param bg_clip:             Background clip that has been upsampled to final scale. If None, bilinear will be used.
    """
    if not isinstance(clip, vs.VideoNode):
        raise vs.Error("codeformer: this is not a clip")

    if clip.format.id != vs.RGB24:
        raise vs.Error("codeformer: only RGB24 format is supported")

    if not torch.cuda.is_available():
        raise vs.Error("codeformer: CUDA is not available")

    if num_streams < 1:
        raise vs.Error("codeformer: num_streams must be at least 1")

    if num_streams > vs.core.num_threads:
        raise vs.Error("codeformer: setting num_streams greater than `core.num_threads` is useless")

    if upscale < 1:
        raise vs.Error("codeformer: upscale must be at least 1")

    if detector not in range(2):
        raise vs.Error("codeformer: detector must be 0 or 1")

    if weight < 0 or weight > 1:
        raise vs.Error("codeformer: weight must be between 0.0 and 1.0 (inclusive)")

    if bg_clip is not None:
        if not isinstance(bg_clip, vs.VideoNode):
            raise vs.Error("codeformer: bg_clip is not a clip")

        if bg_clip.format.id != vs.RGB24:
            raise vs.Error("codeformer: bg_clip must be of RGB24 format")

        if bg_clip.width != clip.width * upscale or bg_clip.height != clip.height * upscale:
            raise vs.Error("codeformer: dimensions of bg_clip must match final upsampling scale")

        if bg_clip.num_frames != clip.num_frames:
            raise vs.Error("codeformer: bg_clip must have the same number of frames as main clip")

    if os.path.getsize(os.path.join(model_dir, "codeformer.pth")) == 0:
        raise vs.Error("codeformer: model files have not been downloaded. run 'python -m vscodeformer' first")

    torch.set_float32_matmul_precision("high")

    device = torch.device("cuda", device_index)

    stream = [torch.cuda.Stream(device=device) for _ in range(num_streams)]
    stream_lock = [Lock() for _ in range(num_streams)]

    model_path = os.path.join(model_dir, "codeformer.pth")

    module = CodeFormer()
    module.load_state_dict(torch.load(model_path, map_location="cpu")["params_ema"])
    module.eval().to(device)

    detection_model = "retinaface_resnet50" if detector == 0 else "dlib"
    face_helper = [
        FaceRestoreHelper(upscale, det_model=detection_model, use_parse=True, device=device) for _ in range(num_streams)
    ]

    index = -1
    index_lock = Lock()

    @torch.inference_mode()
    def inference(n: int, f: list[vs.VideoFrame]) -> vs.VideoFrame:
        nonlocal index
        with index_lock:
            index = (index + 1) % num_streams
            local_index = index

        with stream_lock[local_index], torch.cuda.stream(stream[local_index]):
            img = frame_to_ndarray(f[0])
            bg_img = frame_to_ndarray(f[2]) if bg_clip is not None else None

            face_helper[local_index].clean_all()
            face_helper[local_index].read_image(img)
            face_helper[local_index].get_face_landmarks_5(
                only_center_face=only_center_face, resize=640, eye_dist_threshold=5
            )
            face_helper[local_index].align_warp_face()

            for cropped_face in face_helper[local_index].cropped_faces:
                cropped_face_t = img2tensor(cropped_face / 255.0)
                normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
                cropped_face_t = cropped_face_t.unsqueeze(0).to(device)
                output = module(cropped_face_t, w=weight, adain=True)[0]
                restored_face = tensor2img(output, min_max=(-1, 1))
                face_helper[local_index].add_restored_face(restored_face, cropped_face)

            face_helper[local_index].get_inverse_affine()
            restored_img = face_helper[local_index].paste_faces_to_input_image(upsample_img=bg_img)
            restored_img = cv2.cvtColor(restored_img, cv2.COLOR_BGR2RGB)
            return ndarray_to_frame(restored_img, f[1].copy())

    pad_w = 512 - clip.width if clip.width < 512 else 0
    pad_h = 512 - clip.height if clip.height < 512 else 0

    if pad_w > 0 or pad_h > 0:
        clip = clip.resize.Point(
            clip.width + pad_w, clip.height + pad_h, src_width=clip.width + pad_w, src_height=clip.height + pad_h
        )

    new_clip = clip.std.BlankClip(width=clip.width * upscale, height=clip.height * upscale, keep=True)

    if bg_clip is None:
        ret = new_clip.std.FrameEval(
            lambda n: new_clip.std.ModifyFrame([clip, new_clip], inference), clip_src=[clip, new_clip]
        )
    else:
        bg_pad_w = new_clip.width - bg_clip.width
        bg_pad_h = new_clip.height - bg_clip.height

        if bg_pad_w > 0 or bg_pad_h > 0:
            bg_clip = bg_clip.resize.Point(
                bg_clip.width + bg_pad_w,
                bg_clip.height + bg_pad_h,
                src_width=bg_clip.width + bg_pad_w,
                src_height=bg_clip.height + bg_pad_h,
            )

        ret = new_clip.std.FrameEval(
            lambda n: new_clip.std.ModifyFrame([clip, new_clip, bg_clip], inference), clip_src=[clip, new_clip, bg_clip]
        )

    return ret.std.Crop(right=pad_w * upscale, bottom=pad_h * upscale)


def frame_to_ndarray(frame: vs.VideoFrame) -> np.ndarray:
    return np.stack([np.asarray(frame[plane]) for plane in range(frame.format.num_planes - 1, -1, -1)], axis=2)


def ndarray_to_frame(array: np.ndarray, frame: vs.VideoFrame) -> vs.VideoFrame:
    for plane in range(frame.format.num_planes):
        np.copyto(np.asarray(frame[plane]), array[:, :, plane])
    return frame
